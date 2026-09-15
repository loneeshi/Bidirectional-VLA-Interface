import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "family_language",
    Path(__file__).resolve().parents[1] / "scripts/family_language.py",
)
language = importlib.util.module_from_spec(spec)
spec.loader.exec_module(language)


class FamilyLanguageTests(unittest.TestCase):
    def test_preserves_original_and_target_without_spatial_hallucinations(self):
        for family in ("reach", "grasp", "move", "release"):
            for target in ("butter", "cream cheese box"):
                variants = language.instruction_variants(family, target, "original")
                self.assertEqual(variants[0], "original")
                for text in variants[1:]:
                    self.assertIn(target, text)
                    self.assertLess(len(text), 120)
                    self.assertNotIn("left", text)
                    self.assertNotIn("right", text)

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            language.instruction_variants("reach", "unreviewed", "original")
