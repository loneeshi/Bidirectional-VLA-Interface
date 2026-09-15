import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("bridge_delivery", scripts / "serve_vlm_bridge.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeliveryTests(unittest.TestCase):
    def test_retries_cached_delivery_only(self):
        spool = module.SSHSpool(Path("ssh_config"), "tapt", "/workspace/bridge")
        with patch.object(spool, "_reply_once", side_effect=[RuntimeError("rename"), None]) as send, patch.object(module.time, "sleep"):
            spool.reply("a" * 32, Path("cached.json"))
            self.assertEqual(send.call_count, 2)
            self.assertEqual(send.call_args_list[0], send.call_args_list[1])

    def test_exhaustion_is_bounded(self):
        spool = module.SSHSpool(Path("ssh_config"), "tapt", "/workspace/bridge")
        with patch.object(spool, "_reply_once", side_effect=RuntimeError("offline")) as send, patch.object(module.time, "sleep"):
            with self.assertRaises(RuntimeError):
                spool.reply("a" * 32, Path("cached.json"))
            self.assertEqual(send.call_count, 3)
