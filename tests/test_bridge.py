"""Offline bridge accounting and image transport tests; no SSH or provider calls."""
import base64
import copy
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from bvi import APIBudget, ImageFrame, ProtocolError, VLMRequest, VLMResponse
from bvi.bridge import BridgeProcessor, FileBridgeTransport, atomic_json, decode_request, encode_request

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5e0AAAAASUVORK5CYII=")


class FakeProvider:
    provider = "fake-offline"
    model = "fixture-model"

    def __init__(self):
        self.requests = []
        self.error = None
        self.reset_count = 0

    def generate(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return VLMResponse('{"fixture":true}', "fake-provider-response", {"input_tokens": 9}, "completed")

    def reset_connection(self):
        self.reset_count += 1


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.request = VLMRequest("system", "look at this camera", (ImageFrame("head", PNG),),
                                  {"type": "object"}, 512, "a" * 32)
        self.envelope = encode_request(self.request, "fake-offline", "fixture-model", "offline-auth")
        self.provider = FakeProvider()
        self.budget = APIBudget("offline-auth", 2, 512, 0.2, 0.1, 10000)
        self.processor = BridgeProcessor(self.provider, self.budget, self.directory / "local")

    def test_round_trip_preserves_actual_image_bytes_and_attempt_identity(self):
        decoded = decode_request(json.loads(json.dumps(self.envelope)))
        self.assertEqual(decoded, self.request)
        self.assertEqual(decoded.images[0].data, PNG)

    def test_cached_response_replays_after_restart_without_second_api_call(self):
        first = self.processor.process(self.envelope)
        snapshot = self.processor.directory / ("a" * 32 + ".request.json")
        self.assertEqual(json.loads(snapshot.read_text()), self.envelope)
        other_provider = FakeProvider()
        restarted = BridgeProcessor(other_provider, self.budget, self.directory / "local")
        self.assertEqual(restarted.process(self.envelope), first)
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(other_provider.requests, [])
        records = [json.loads(line) for line in self.processor.logger.path.read_text().splitlines()]
        self.assertEqual({row["accounting_role"] for row in records},
                         {"mirror_of_remote_attempt", "physical_provider_attempt"})
        self.assertEqual(records[-1]["bridge_id"], "a" * 32)

    def test_unknown_started_attempt_is_never_reissued(self):
        self.processor.process(self.envelope)
        (self.processor.directory / f"{'a' * 32}.response.json").unlink()
        with self.assertRaisesRegex(ProtocolError, "charge unknown"):
            self.processor.process(self.envelope)
        self.assertEqual(len(self.provider.requests), 1)

    def test_timeout_error_is_cached_and_does_not_retry(self):
        self.provider.error = TimeoutError("private provider details")
        result = self.processor.process(self.envelope)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(self.processor.process(self.envelope), result)
        self.assertEqual(len(self.provider.requests), 1)

    def test_connection_error_retries_are_individually_claimed_and_bounded(self):
        ConnectionErrorType=type('APIConnectionError',(Exception,),{})
        class Flaky(FakeProvider):
            def generate(self,request):
                self.requests.append(request)
                if len(self.requests)<3:raise ConnectionErrorType('private')
                return VLMResponse('{}','ok',{},'completed')
        provider=Flaky();processor=BridgeProcessor(provider,
            replace(self.budget,max_calls=4,max_cost_usd=.4),self.directory/'retry',max_provider_retries=10,
            retry_delay_seconds=0)
        result=processor.process(self.envelope)
        self.assertTrue(result['ok']);self.assertEqual(result['provider_attempts'],3)
        self.assertEqual(len(list(processor.directory.glob('*.claim.json'))),3)
        self.assertEqual(provider.reset_count,2)
        events=[json.loads(x) for x in processor.logger.path.read_text().splitlines()]
        failures=[x for x in events if x['event']=='bridge_provider_attempt_failed']
        self.assertEqual([x['exception_chain'] for x in failures],[['APIConnectionError']]*2)

    def test_retry_never_exceeds_global_physical_attempt_budget(self):
        self.provider.error=type('APIConnectionError',(Exception,),{})('private')
        processor=BridgeProcessor(self.provider,self.budget,self.directory/'retry-budget',
                                  max_provider_retries=10,retry_delay_seconds=0)
        result=processor.process(self.envelope)
        self.assertFalse(result['ok']);self.assertEqual(result['error_type'],'APIBudgetExhausted')
        self.assertEqual(len(self.provider.requests),2)
        self.assertEqual(len(list(processor.directory.glob('*.claim.json'))),2)

    def test_changed_payload_with_same_id_cannot_reuse_or_resend(self):
        self.processor.process(self.envelope)
        changed = copy.deepcopy(self.envelope)
        changed["request"]["prompt"] = "a different instruction"
        with self.assertRaisesRegex(ProtocolError, "different request"):
            self.processor.process(changed)
        self.assertEqual(len(self.provider.requests), 1)

    def test_provider_model_and_authorization_cannot_be_changed_remotely(self):
        for field in ("provider", "model", "authorization_id"):
            envelope = {**self.envelope, field: "unapproved"}
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                self.processor.process(envelope)
        self.assertEqual(self.provider.requests, [])

    def test_local_caps_reject_request_before_api(self):
        processor = BridgeProcessor(self.provider, replace(self.budget, max_calls=1),
                                    self.directory / "limited")
        processor.process(self.envelope)
        next_envelope = encode_request(replace(self.request, attempt_id="b" * 32),
                                      "fake-offline", "fixture-model", "offline-auth")
        with self.assertRaisesRegex(ProtocolError, "budget exhausted"):
            processor.process(next_envelope)
        self.assertEqual(len(self.provider.requests), 1)

    def test_invalid_images_or_ids_cannot_enter_bridge(self):
        invalid = copy.deepcopy(self.envelope)
        invalid["bridge_id"] = "../escape"
        with self.assertRaises(ProtocolError):
            decode_request(invalid)
        invalid = copy.deepcopy(self.envelope)
        invalid["request"]["images"][0]["data_base64"] = base64.b64encode(b"filename.png").decode()
        with self.assertRaises(ProtocolError):
            decode_request(invalid)

    def test_file_transport_delivers_matching_response_without_credentials(self):
        remote = self.directory / "remote"
        transport = FileBridgeTransport(remote, "fake-offline", "fixture-model", "offline-auth",
                                        timeout_seconds=2)
        received = []
        def local_worker():
            request_file = remote / ("a" * 32) / "request.json"
            deadline = time.monotonic() + 2
            while not request_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            envelope = json.loads(request_file.read_text())
            received.append(decode_request(envelope))
            atomic_json(request_file.parent / "response.json", {
                "bridge_id": "a" * 32, "ok": True,
                "response": {"text": "{}", "request_id": "offline-id", "usage": {"input_tokens": 3},
                             "finish_reason": "completed"}})
        worker = threading.Thread(target=local_worker)
        worker.start()
        response = transport.generate(self.request)
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received[0].images[0].data, PNG)
        self.assertEqual(response.request_id, "offline-id")

    def test_remote_wait_expires_without_new_attempt(self):
        transport = FileBridgeTransport(self.directory / "remote", "fake-offline", "fixture-model",
                                        "offline-auth", timeout_seconds=0.01)
        with self.assertRaises(TimeoutError):
            transport.generate(self.request)
        expired = self.directory / "remote" / ("a" * 32) / "expired.json"
        self.assertTrue(expired.is_file())
        self.assertEqual(len(list(expired.parent.parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()

