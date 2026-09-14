"""Offline coverage for archive traversal safety and mounted-filesystem cost."""

import importlib.util
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_assets.py"
SPEC = importlib.util.spec_from_file_location("download_assets", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class ArchiveValidationTest(unittest.TestCase):
    def test_absent_subtree_does_not_stat_every_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            members = [zipfile.ZipInfo(f"fixture/plans/{i}/plan.json") for i in range(33_345)]
            original = Path.lstat
            with mock.patch.object(Path, "lstat", autospec=True, side_effect=original) as probe:
                downloader.validate_zip_members(members, parent, "fixture")
            # One missing ancestor proves there are no existing descendant links.
            self.assertEqual(probe.call_count, 1)

    def test_archive_traversal_and_symlink_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ("../outside", "fixture/../../outside", "/fixture/file",
                             "fixture\\outside", "fixture/file:stream"):
                member = zipfile.ZipInfo(filename)
                member.filename = filename  # Preserve raw separator on Windows too.
                with self.subTest(filename=filename), self.assertRaisesRegex(RuntimeError, "Unsafe ZIP"):
                    downloader.validate_zip_members([member], Path(tmp), "fixture")
            symlink = zipfile.ZipInfo("fixture/link")
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            with self.assertRaisesRegex(RuntimeError, "Unsafe ZIP"):
                downloader.validate_zip_members([symlink], Path(tmp), "fixture")

    def test_existing_leaf_and_ancestor_symlink_escapes_are_rejected(self):
        # Mock only filesystem metadata/resolution so this also runs on Windows
        # hosts that do not grant the privilege required to create symlinks.
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp).resolve()
            (parent / "fixture").mkdir()
            original_lstat, original_resolve = Path.lstat, Path.resolve
            for name in ("fixture/link", "fixture/link/child.txt"):
                link = parent / "fixture/link"
                outside = parent.parent / "outside-archive-root"

                def lstat(path):
                    if path == link:
                        return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
                    return original_lstat(path)

                def resolve(path, *args, **kwargs):
                    return outside if path == link else original_resolve(path, *args, **kwargs)

                with self.subTest(name=name), \
                        mock.patch.object(Path, "lstat", autospec=True, side_effect=lstat), \
                        mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve), \
                        self.assertRaisesRegex(RuntimeError, "Unsafe ZIP"):
                    downloader.validate_zip_members([zipfile.ZipInfo(name)], parent, "fixture")

    def test_verified_extraction_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("fixture/a.txt", "hello")
            source = dict(name="fixture", directory="fixture", url="unused",
                          size=archive.stat().st_size, extracted_size=5,
                          sha256=downloader.sha256(archive))
            downloader.extract_asset(source, root / "assets", cache)
            self.assertEqual((root / "assets/fixture/a.txt").read_text(), "hello")
            self.assertFalse(archive.exists())
            with mock.patch.object(downloader, "download_zip", side_effect=AssertionError("Unexpected download")):
                downloader.extract_asset(source, root / "assets", cache)


if __name__ == "__main__":
    unittest.main()
