import base64
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bvi import (APIBudget, AllowedCall, ImageFrame, JsonlLogger, Observation,
                 ProtocolError, SkillSpec, Target, VLMCoordinator, VLMResponse,
                 parse_request)
from bvi.providers import AnthropicTransport, OpenAITransport


# Tiny synthetic PNG fixture for transport tests. This is not a camera rollout.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5e0AAAAASUVORK5CYII=")


class FakeTransport:
    provider = "fake-offline"
    model = "fixture-model"

    def __init__(self, text):
        self.text = text
        self.requests = []
        self.error = None

    def generate(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return VLMResponse(self.text, "fixture-request", {"input_tokens": 10, "output_tokens": 5})


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logger = JsonlLogger(Path(self.temp.name) / "events.jsonl", "offline-fixture")
        self.specs = {"pick": SkillSpec("pick"), "place": SkillSpec("place")}
        self.obs = Observation("f0", 0, policy={"private_pose": "not_for_vlm"},
            images=(ImageFrame("head", PNG),),
            targets=(Target("cup", "a cup", "fake_fixture"),
                     Target("table", "a table", "fake_fixture")),
            allowed_calls=(AllowedCall("pick", "cup"), AllowedCall("place", "table")),
            task="Put the cup on the table.", metadata={"hidden": "not_for_vlm"})
        self.payload = {"call_id": "c1", "skill": "pick", "target_id": "cup",
                        "observation_id": "f0", "requirements": [
                            {"id": "r1", "predicate": "benchmark_success"}],
                        "max_steps": 10, "timeout_seconds": 30.0}
        self.transport = FakeTransport(json.dumps(self.payload))
        # This fake budget is fixture data, never a real research authorization.
        self.budget = APIBudget("offline-test-only", 2, 500, 0.1, 0.05)
        self.coordinator = VLMCoordinator(self.transport, self.specs, self.logger, self.budget)

    def records(self):
        return [json.loads(line) for line in self.logger.path.read_text().splitlines()]

    def test_explicit_text_only_view_removes_wire_images(self):
        from bvi.feedback import FeedbackView
        self.coordinator.feedback_view = FeedbackView(images=False)
        self.coordinator.decide(self.obs)
        request = self.transport.requests[-1]
        self.assertEqual(request.images, ())
        self.assertEqual(json.loads(request.prompt)['image_order'], [])

    def test_raw_prompt_context_matches_legacy_bytes(self):
        from bvi.coordinator import json_default
        history = [dict(skill='pick', target_id='cup', steps=2)]
        expected = dict(task=self.obs.task, frame_id=self.obs.frame_id,
            image_order=[x.camera for x in self.obs.images], targets=self.obs.targets,
            allowed_calls=self.obs.allowed_calls, skill_contracts=list(self.specs.values()),
            feedback_history=list(history))
        self.coordinator.decide(self.obs, history)
        self.assertEqual(self.transport.requests[-1].prompt,
                         json.dumps(expected, default=json_default, ensure_ascii=False, allow_nan=False))

    def test_only_returned_invalid_response_has_model_error_type(self):
        from bvi.coordinator import ModelResponseError
        self.transport.text = '{}{}'
        with self.assertRaises(ModelResponseError):
            self.coordinator.decide(self.obs)
        self.assertEqual(self.coordinator.calls_reserved, 1)
        self.transport.error = ProtocolError('transport failure')
        with self.assertRaises(ProtocolError) as caught:
            self.coordinator.decide(self.obs)
        self.assertNotIsInstance(caught.exception, ModelResponseError)

    def test_identical_repeated_call_is_executed_once_and_audited(self):
        raw = self.transport.text
        self.transport.text = raw + raw
        request = self.coordinator.decide(self.obs)
        self.assertEqual(request.call_id, 'c1')
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.coordinator.calls_reserved, 1)
        records = self.records()
        self.assertEqual(records[1]['response_text'], raw + raw)
        normalized = next(r for r in records if r['event'] == 'coordinator_output_normalized')
        self.assertEqual(normalized['copies'], 2)

    def test_semantically_identical_repetition_allows_only_whitespace_difference(self):
        raw = self.transport.text
        second = json.dumps(json.loads(raw), separators=(',', ':'))
        self.transport.text = raw + '\n' + second
        request = self.coordinator.decide(self.obs)
        self.assertEqual(request.call_id, 'c1')
        normalized = next(r for r in self.records()
                          if r['event'] == 'coordinator_output_normalized')
        self.assertEqual(normalized['copies'], 2)
        self.assertEqual(normalized['rule'], 'semantically_identical_json_repetition/2')

    def test_conflicting_or_partial_repeated_call_is_rejected(self):
        from bvi.coordinator import normalize_repeated_response
        raw = json.dumps(self.payload)
        for suffix in (json.dumps({**self.payload, 'skill': 'place'}), raw[:-1], ' explanation'):
            text = raw + suffix
            self.assertEqual(normalize_repeated_response(text), (text, 1))
            with self.assertRaises(ProtocolError):
                parse_request(text, self.obs, self.specs)

    def test_normalization_does_not_bypass_contract_validation(self):
        from bvi.coordinator import ModelResponseError
        self.transport.text = json.dumps({**self.payload, 'target_id': 'unknown'}) * 2
        with self.assertRaises(ModelResponseError):
            self.coordinator.decide(self.obs)

    def test_actual_image_bytes_pass_without_policy_privileges(self):
        request = self.coordinator.decide(self.obs)
        self.assertEqual(request.target_id, "cup")
        self.assertEqual(self.transport.requests[0].images[0].data, PNG)
        self.assertNotIn("private_pose", self.transport.requests[0].prompt)
        self.assertNotIn("not_for_vlm", self.transport.requests[0].prompt)
        self.assertEqual([r["event"] for r in self.records()],
                         ["api_request_started", "api_usage", "coordinator_decision"])
        self.assertIsNone(self.records()[1]["amount"])
        self.assertEqual(self.records()[1]["usage"]["input_tokens"], 10)

    def test_executor_owned_horizon_is_not_model_output(self):
        payload = {key: value for key, value in self.payload.items() if key != "max_steps"}
        self.transport.text = json.dumps(payload)
        horizons = {name: spec.max_steps for name, spec in self.specs.items()}
        coordinator = VLMCoordinator(self.transport, self.specs, self.logger, self.budget,
                                     executor_horizons=horizons)
        request = coordinator.decide(self.obs)
        self.assertEqual(request.max_steps, self.specs["pick"].max_steps)
        schema = self.transport.requests[-1].schema
        self.assertNotIn("max_steps", schema["properties"])
        self.assertIn("executor, not you", self.transport.requests[-1].system)

    def test_executor_owned_horizon_rejects_model_override(self):
        horizons = {name: spec.max_steps for name, spec in self.specs.items()}
        coordinator = VLMCoordinator(self.transport, self.specs, self.logger, self.budget,
                                     executor_horizons=horizons)
        with self.assertRaises(ProtocolError):
            coordinator.decide(self.obs)

    def test_missing_or_invalid_image_is_rejected_without_request(self):
        for images in ((), (ImageFrame("head", b"/tmp/image.png"),)):
            with self.subTest(images=images):
                with self.assertRaises(ProtocolError):
                    self.coordinator.decide(replace(self.obs, images=images))
        self.assertFalse(self.transport.requests)

    def test_unknown_target_skill_pair_and_extra_fields_rejected(self):
        variants = [
            {**self.payload, "target_id": "imagined-bottle"},
            {**self.payload, "skill": "teleport"},
            {**self.payload, "target_id": "table"},
            {**self.payload, "observation_id": "old-frame"},
            {**self.payload, "task_pointer": 4},
            {**self.payload, "max_steps": True},
            {**self.payload, "max_steps": 50000},
            {**self.payload, "requirements": []},
            {**self.payload, "requirements": [{"id": "r1", "predicate": "grasp_neck"}]},
            {**self.payload, "timeout_seconds": float("nan")},
        ]
        for payload in variants:
            with self.subTest(payload=payload):
                with self.assertRaises(ProtocolError):
                    parse_request(json.dumps(payload), self.obs, self.specs)

    def test_duplicate_fields_not_silently_overwritten(self):
        payload = json.dumps(self.payload).replace('"target_id": "cup"',
                                                 '"target_id": "table", "target_id": "cup"')
        with self.assertRaises(ProtocolError):
            parse_request(payload, self.obs, self.specs)

    def test_response_parse_failure_still_records_usage_no_retry(self):
        self.transport.text = "I would probably navigate."
        with self.assertRaises(ProtocolError):
            self.coordinator.decide(self.obs)
        self.assertEqual(len(self.transport.requests), 1)
        records = self.records()
        self.assertEqual(records[1]["event"], "api_usage")
        self.assertEqual(records[2]["event"], "coordinator_rejected")

    def test_unknown_transport_failure_retains_reservation_no_retry(self):
        self.transport.error = TimeoutError("sensitive details")
        with self.assertRaises(TimeoutError):
            self.coordinator.decide(self.obs)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.coordinator.calls_reserved, 1)
        failure = self.records()[-1]
        self.assertEqual(failure["status"], "charge_unknown_pending_reconciliation")
        self.assertNotIn("sensitive details", json.dumps(failure))

    def test_restart_preserves_call_and_cost_reservations(self):
        self.coordinator.decide(self.obs)
        restarted = VLMCoordinator(self.transport, self.specs, self.logger, self.budget)
        restarted.decide(self.obs)
        with self.assertRaises(ProtocolError):
            restarted.decide(self.obs)
        self.assertEqual(len(self.transport.requests), 2)
        self.assertAlmostEqual(restarted.cost_reserved_usd, 0.1)

    def test_cost_cap_can_stop_before_call_count_cap(self):
        budget = APIBudget("offline-small-budget", 10, 500, 0.05, 0.05)
        coordinator = VLMCoordinator(self.transport, self.specs, self.logger, budget)
        coordinator.decide(self.obs)
        with self.assertRaises(ProtocolError):
            coordinator.decide(self.obs)
        self.assertEqual(len(self.transport.requests), 1)

    def test_large_input_is_rejected_before_spending(self):
        coordinator = VLMCoordinator(self.transport, self.specs, self.logger,
            APIBudget("offline-byte-budget", 2, 500, 0.1, 0.05, max_input_bytes=100))
        with self.assertRaises(ProtocolError):
            coordinator.decide(self.obs)
        self.assertEqual(len(self.transport.requests), 0)

    def test_incomplete_but_parseable_response_is_not_dispatched(self):
        self.transport.generate = lambda request: VLMResponse(
            json.dumps(self.payload), "fixture-incomplete", {"output_tokens": 500}, "incomplete")
        with self.assertRaises(ProtocolError):
            self.coordinator.decide(self.obs)
        self.assertEqual(self.records()[1]["event"], "api_usage")
        self.assertEqual(self.records()[2]["event"], "coordinator_rejected")

    def test_no_admissible_calls_never_spends(self):
        with self.assertRaises(ProtocolError):
            self.coordinator.decide(replace(self.obs, allowed_calls=()))
        self.assertEqual(len(self.transport.requests), 0)

    def test_openai_sdk_payload_contains_data_url_and_strict_schema(self):
        self.coordinator.decide(self.obs)
        request = self.transport.requests[0]
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
            output_text=json.dumps(self.payload), id="fake-openai-id",
            usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10}), status="completed"))))
        response = OpenAITransport("fake-explicit-model", client=client,
                                   reasoning_effort="none").generate(request)
        kwargs = client.responses.create.call_args.kwargs
        image = kwargs["input"][0]["content"][1]
        self.assertEqual(base64.b64decode(image["image_url"].split(",")[1]), PNG)
        self.assertEqual(image["detail"], "low")
        self.assertEqual(kwargs["reasoning"], {"effort": "none"})
        self.assertTrue(kwargs["text"]["format"]["strict"])
        self.assertFalse(kwargs["store"])
        self.assertEqual(response.request_id, "fake-openai-id")

    def test_anthropic_sdk_payload_forces_one_skill_schema(self):
        self.coordinator.decide(self.obs)
        request = self.transport.requests[0]
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="request_skill", input=self.payload)],
            id="fake-anthropic-id", usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10}),
            stop_reason="tool_use"))))
        response = AnthropicTransport("fake-explicit-model", client=client).generate(request)
        kwargs = client.messages.create.call_args.kwargs
        source = kwargs["messages"][0]["content"][1]["source"]
        self.assertEqual(base64.b64decode(source["data"]), PNG)
        self.assertEqual(kwargs["tool_choice"]["name"], "request_skill")
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(json.loads(response.text), self.payload)


if __name__ == "__main__":
    unittest.main()
