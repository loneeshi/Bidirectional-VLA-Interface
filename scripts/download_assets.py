#!/usr/bin/env python3
"""Download pinned MS-HAB inference assets, with resumable verified files.

--root is MS_ASSET_DIR (ManiSkill appends /data), not the data directory itself.
Requires huggingface_hub in the active simulation environment; no GPU is used.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.request
import zipfile


YCB = {
    "name": "ycb",
    "directory": "mani_skill2_ycb",
    "url": "https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/"
    "a1988e0918a9a1f2a9cb46ad7984efc3d422ac63/data/mani_skill2_ycb.zip",
    "size": 26_229_037,
    "extracted_size": 87_217_479,
    "sha256": "1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3",
}
REARRANGE = {
    "name": "rearrange",
    "directory": "rearrange",
    "url": "https://huggingface.co/datasets/haosulab/ReplicaCADRearrange/resolve/"
    "c5921276ddece53607cd7dd55e7ed54385c9e49d/rearrange.zip",
    "size": 1_486_211_499,
    "extracted_size": 2_949_738_875,
    "sha256": "6ab9a0ceece6e0fe176ffc67a6aab1bad1efb7a4bf70808089e19673cd59b15b",
}
REPLICA_REVISION = "88bdca74dd4ec1f8c994904f7985392fc3f2b4c4"
CHECKPOINT_REVISION = "91e96be85128df43728a7511355c3fa999bd2c94"


def log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_space(path: Path, needed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    reserve = 1_000_000_000
    if free < needed + reserve:
        raise RuntimeError(
            f"Insufficient free disk at {path}: {free} B; need {needed} B "
            f"plus {reserve} B reserve. No automatic resize or deletion is performed."
        )


def download_zip(source: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / (source["name"] + ".zip")
    partial = final.with_suffix(".zip.part")
    if final.exists():
        if final.stat().st_size == source["size"] and sha256(final) == source["sha256"]:
            return final
        raise RuntimeError(f"Unverified existing archive {final}; inspect it before retrying")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source["size"]:
        raise RuntimeError(f"Oversized partial archive {partial}; inspect it before retrying")
    require_space(cache, source["size"] - offset)
    if offset < source["size"]:
        headers = {"User-Agent": "Bidirectional-VLA-Interface/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        log("download", asset=source["name"], offset=offset, expected_bytes=source["size"])
        request = urllib.request.Request(source["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Server returned an inconsistent download range")
            if response.status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status {response.status}")
            if not append:
                offset = 0  # Server may ignore Range; replace only the partial file.
            received = offset
            reported_at = time.monotonic()
            with partial.open("ab" if append else "wb") as stream:
                while block := response.read(4 * 1024 * 1024):
                    received += len(block)
                    if received > source["size"]:
                        raise RuntimeError("Download exceeds the pinned file size")
                    stream.write(block)
                    if time.monotonic() - reported_at > 15:
                        log("download_progress", asset=source["name"], bytes=received)
                        reported_at = time.monotonic()
    if partial.stat().st_size != source["size"]:
        raise RuntimeError(f"Incomplete {partial}; rerun to resume")
    if sha256(partial) != source["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {partial}; inspect before removing/retrying")
    partial.replace(final)
    return final


def extraction_complete(marker: Path, source: dict, destination: Path) -> bool:
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("sha256") != source["sha256"]:
        return False
    # Manifest is saved only after all files pass ZIP CRC checks during extraction.
    return all(
        (destination / entry["path"]).is_file()
        and (destination / entry["path"]).stat().st_size == entry["size"]
        for entry in record.get("files", [])
    ) and record.get("extracted_bytes") == source["extracted_size"]


def extract_asset(source: dict, parent: Path, cache: Path) -> dict:
    destination = parent / source["directory"]
    marker = destination / ".bvi-extraction.json"
    if extraction_complete(marker, source, destination):
        log("asset_cached", asset=source["name"], directory=str(destination))
        return {**source, "destination": str(destination), "status": "verified"}
    archive = download_zip(source, cache)
    require_space(parent, source["extracted_size"])
    files = []
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if sum(member.file_size for member in members) != source["extracted_size"]:
            raise RuntimeError("ZIP expanded size differs from the pinned manifest")
        for member in members:
            pure = PurePosixPath(member.filename)
            target = (parent / member.filename).resolve()
            mode = member.external_attr >> 16
            if (pure.is_absolute() or ".." in pure.parts or "\\" in member.filename
                    or not pure.parts or pure.parts[0] != source["directory"]
                    or not target.is_relative_to(parent.resolve())
                    or stat.S_ISLNK(mode)):
                raise RuntimeError(f"Unsafe ZIP member {member.filename}")
        log("extract", asset=source["name"], entries=len(members))
        # Paths are validated first; zipfile verifies CRC while reading each member.
        # Interrupted extractions can resume by replacing only pinned source members.
        for member in members:
            z.extract(member, parent)
            if not member.is_dir():
                files.append({"path": str(PurePosixPath(member.filename).relative_to(source["directory"])),
                              "size": member.file_size})
    record = {"sha256": source["sha256"], "extracted_bytes": source["extracted_size"], "files": files}
    atomic_json(marker, record)
    archive.unlink()  # Delete only this verified temporary ZIP after successful extraction.
    return {**source, "destination": str(destination), "status": "verified"}


def snapshot(repo_id: str, repo_type: str, revision: str, destination: Path,
             patterns: list[str] | None, workers: int) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Unexpected revision for {repo_id}: {info.sha}")
    siblings = [s for s in info.siblings
                if patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    if not siblings or any(s.size is None for s in siblings):
        raise RuntimeError(f"Missing pinned file metadata for {repo_id}")
    missing_bytes = sum(s.size for s in siblings if not (destination / s.rfilename).is_file())
    require_space(destination, missing_bytes)
    log("snapshot_download", repository=repo_id, revision=revision, files=len(siblings),
        bytes=sum(s.size for s in siblings), destination=str(destination))
    snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision,
                      allow_patterns=patterns, local_dir=str(destination), max_workers=workers)
    manifest = []
    for entry in siblings:
        path = destination / entry.rfilename
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"File size mismatch: {path}")
        expected_sha = (entry.lfs.get("sha256") if isinstance(entry.lfs, dict)
                        else getattr(entry.lfs, "sha256", None)) if entry.lfs else None
        if expected_sha and sha256(path) != expected_sha:
            raise RuntimeError(f"File SHA-256 mismatch: {path}")
        git_blob = getattr(entry, "blob_id", None)
        if not expected_sha and git_blob:
            digest = hashlib.sha1(f"blob {entry.size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != git_blob:
                raise RuntimeError(f"Git blob hash mismatch: {path}")
        manifest.append({"path": entry.rfilename, "bytes": entry.size,
                         "sha256": expected_sha, "git_blob": git_blob})
    result = {"repository": repo_id, "repository_type": repo_type, "revision": revision,
              "destination": str(destination), "files": manifest, "status": "verified"}
    atomic_json(destination / ".bvi-snapshot.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="MS_ASSET_DIR; /data is appended")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Defaults to ROOT/data/mshab_checkpoints")
    parser.add_argument("--full-tidy", action="store_true",
                        help="All TidyHouse RL checkpoints; required by unmodified mshab.evaluate")
    parser.add_argument("--only", choices=("all", "assets", "checkpoints"), default="all")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    root = args.root.expanduser().resolve()
    data = root / "data"
    checkpoints = (args.checkpoint_root or data / "mshab_checkpoints").expanduser().resolve()
    cache = root / ".downloads"
    os.environ.setdefault("HF_HOME", str(root / ".hf-cache"))
    manifest = {"schema_version": 1, "root": str(root), "checkpoint_root": str(checkpoints),
                "full_tidy": args.full_tidy, "started_at_unix": time.time(), "assets": []}
    manifest_path = root / "download-manifest.json"
    atomic_json(manifest_path, manifest)

    def record(value):
        manifest["assets"].append(value)
        atomic_json(manifest_path, manifest)

    if args.only in ("all", "assets"):
        record(extract_asset(YCB, data / "assets", cache))
        rc = data / "scene_datasets/replica_cad_dataset"
        record(snapshot("haosulab/ReplicaCAD", "dataset", REPLICA_REVISION, rc, None, args.max_workers))
        record(extract_asset(REARRANGE, rc, cache))
    if args.only in ("all", "checkpoints"):
        patterns = (["rl/tidy_house/**"] if args.full_tidy else
                    [f"rl/tidy_house/{skill}/all/*" for skill in ("navigate", "pick", "place")])
        record(snapshot("arth-shukla/mshab_checkpoints", "model", CHECKPOINT_REVISION,
                        checkpoints, patterns, args.max_workers))
    manifest["completed_at_unix"] = time.time()
    atomic_json(manifest_path, manifest)
    log("complete", manifest=str(manifest_path), ms_asset_dir=str(root),
        checkpoint_root=str(checkpoints), free_bytes=shutil.disk_usage(root).free)


if __name__ == "__main__":
    main()
