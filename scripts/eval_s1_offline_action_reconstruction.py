"""Teacher-forced S1-IA action reconstruction on recorded MS-HAB observations.

This is a level-1 diagnostic: it never creates or steps a simulator and never
updates model parameters.  The selected recorded observation at time ``t`` is
sent to the frozen policy, then the sampled action chunk is compared with the
recorded expert actions from the same invocation.  The first action is the
strict teacher-forced comparison; later chunk positions are reported
separately because they are open-loop predictions from the observation at
``t``.

The S1 head is stochastic.  Every observation therefore receives multiple
samples.  Metrics include both the distribution-mean prediction and the
expected error of an individual sample.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time

import numpy as np


GPU1_UUID = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
HEAD_CHANNELS = (8, 9)

# Pinned ManiSkill Fetch ``pd_joint_delta_pos`` composite-controller order.
# Controller-normalized [-1, 1] commands are converted to the actual target
# delta/target/velocity passed to the sub-controller.  These are physical
# command units, not measured joint displacement after physics integration.
ACTION_CHANNELS = (
    dict(name="shoulder_pan", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="shoulder_lift", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="upperarm_roll", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="elbow_flex", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="forearm_roll", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="wrist_flex", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="wrist_roll", component="arm", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="gripper", component="gripper", unit="m_joint_target", lower=-0.01, upper=0.05),
    dict(name="head_pan", component="head", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="head_tilt", component="head", unit="rad_delta", lower=-0.1, upper=0.1),
    dict(name="torso_lift", component="torso", unit="m_delta", lower=-0.1, upper=0.1),
    dict(name="base_forward", component="base", unit="m_per_s", lower=-1.0, upper=1.0),
    dict(name="base_yaw", component="base", unit="rad_per_s", lower=-3.14, upper=3.14),
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_to_physical(actions):
    """Map normalized Fetch13 commands to sub-controller physical commands."""
    values = np.asarray(actions, dtype=np.float64)
    if values.shape[-1] != len(ACTION_CHANNELS) or not np.isfinite(values).all():
        raise ValueError("Expected finite Fetch13 controller commands")
    lower = np.asarray([row["lower"] for row in ACTION_CHANNELS], dtype=np.float64)
    upper = np.asarray([row["upper"] for row in ACTION_CHANNELS], dtype=np.float64)
    return lower + (values + 1.0) * (upper - lower) / 2.0


def applied_controller_actions(raw):
    """Apply the same controller clipping and stationary-head wrapper as S1."""
    values = np.asarray(raw, dtype=np.float64)
    if values.shape[-1] != len(ACTION_CHANNELS) or not np.isfinite(values).all():
        raise ValueError("Expected finite Fetch13 policy actions")
    result = np.clip(values, -1.0, 1.0)
    result[..., list(HEAD_CHANNELS)] = 0.0
    return result


def _spread_order(length: int) -> list[int]:
    """Deterministic farthest-point order over a one-dimensional sequence."""
    if length < 1:
        return []
    chosen = [0]
    if length > 1:
        chosen.append(length - 1)
    while len(chosen) < length:
        remaining = [i for i in range(length) if i not in chosen]
        chosen.append(max(remaining, key=lambda i: (min(abs(i - j) for j in chosen), -i)))
    return chosen


def _row_key(row):
    return (str(row["parent_id"]), int(row["call_index"]), int(row["observation_index"]))


def balanced_selection(rows, limit: int, target_frames_per_call: int = 4):
    """Select a spread of calls and several spread frames within each call."""
    if limit < 1 or target_frames_per_call < 1:
        raise ValueError("Positive selection limits required")
    groups = defaultdict(list)
    for row in sorted(rows, key=_row_key):
        groups[(str(row["parent_id"]), int(row["call_index"]))].append(row)
    group_keys = sorted(groups)
    # Covering every call with only one frame produces no temporal signal.  The
    # default instead reserves roughly four positions per selected call while
    # spreading those calls across the complete stable roster.
    minimum_groups = 2 if limit >= 2 and len(group_keys) >= 2 else 1
    group_budget = min(len(group_keys), max(minimum_groups, limit // target_frames_per_call))
    ordered_groups = [group_keys[i] for i in _spread_order(len(group_keys))[:group_budget]]
    candidates = {
        key: [groups[key][i] for i in _spread_order(len(groups[key]))]
        for key in ordered_groups
    }
    selected = []
    rank = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for key in ordered_groups:
            if rank < len(candidates[key]):
                selected.append(candidates[key][rank])
                added = True
                if len(selected) == min(limit, len(rows)):
                    break
        if not added:
            break
        rank += 1
    return sorted(selected, key=_row_key)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return _finite_or_none(np.corrcoef(x, y)[0, 1])


def channel_statistics(sample_predictions, expert):
    """Return per-channel statistics for [target, sample, channel] arrays."""
    samples = np.asarray(sample_predictions, np.float64)
    target = np.asarray(expert, np.float64)
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("Metric shapes must be [target,sample,channel] and [target,channel]")
    mean = samples.mean(axis=1)
    records = []
    for channel in range(samples.shape[-1]):
        prediction, truth = mean[:, channel], target[:, channel]
        error = prediction - truth
        expert_std = float(np.std(truth))
        stochastic = np.std(samples[:, :, channel], axis=1, ddof=1) if samples.shape[1] > 1 else np.zeros(len(samples))
        records.append(dict(
            n_targets=len(truth),
            samples_per_target=samples.shape[1],
            expert_mean=float(np.mean(truth)),
            expert_std=expert_std,
            expert_min=float(np.min(truth)),
            expert_max=float(np.max(truth)),
            prediction_mean=float(np.mean(prediction)),
            prediction_std=float(np.std(prediction)),
            bias=float(np.mean(error)),
            mae=float(np.mean(np.abs(error))),
            rmse=float(np.sqrt(np.mean(np.square(error)))),
            pearson=pearson(prediction, truth),
            normalized_rmse_by_expert_std=(float(np.sqrt(np.mean(np.square(error))) / expert_std)
                                           if expert_std > 1e-12 else None),
            expected_sample_rmse=float(np.sqrt(np.mean(np.square(samples[:, :, channel] - truth[:, None])))),
            mean_stochastic_std=float(np.mean(stochastic)),
        ))
    return records


def _scope_arrays(predictions, expert, valid, scope):
    # predictions: [observation, sample, horizon, channel]
    if scope == "first_action":
        return predictions[:, :, 0, :], expert[:, 0, :]
    if scope != "valid_chunk":
        raise ValueError(scope)
    reordered = np.transpose(predictions, (0, 2, 1, 3))
    return reordered[valid], expert[valid]


def per_channel_records(raw_predictions, expert, valid):
    applied = applied_controller_actions(raw_predictions)
    expert_applied = applied_controller_actions(expert)
    physical = controller_to_physical(applied)
    expert_physical = controller_to_physical(expert_applied)
    records = []
    for scope in ("first_action", "valid_chunk"):
        raw_scope, expert_controller = _scope_arrays(raw_predictions, expert_applied, valid, scope)
        applied_scope, _ = _scope_arrays(applied, expert_applied, valid, scope)
        physical_scope, expert_si = _scope_arrays(physical, expert_physical, valid, scope)
        raw_stats = channel_statistics(raw_scope, expert_controller)
        controller_stats = channel_statistics(applied_scope, expert_controller)
        physical_stats = channel_statistics(physical_scope, expert_si)
        for channel, contract in enumerate(ACTION_CHANNELS):
            row = dict(scope=scope, channel=channel, **contract,
                       stationary_head_masked=channel in HEAD_CHANNELS)
            row.update({f"physical_{key}": value for key, value in physical_stats[channel].items()})
            zero_physical = controller_to_physical(np.zeros(13))[channel]
            zero_rmse = float(np.sqrt(np.mean(np.square(expert_si[:, channel] - zero_physical))))
            row["physical_zero_controller_rmse"] = zero_rmse
            row["physical_rmse_over_zero_controller"] = (
                physical_stats[channel]["rmse"] / zero_rmse if zero_rmse > 1e-12 else None)
            for key in ("bias", "mae", "rmse", "pearson", "normalized_rmse_by_expert_std",
                        "expected_sample_rmse", "mean_stochastic_std"):
                row[f"controller_applied_{key}"] = controller_stats[channel][key]
                row[f"controller_raw_{key}"] = raw_stats[channel][key]
            raw_values = raw_scope[:, :, channel]
            row["raw_out_of_bounds_fraction"] = float(np.mean(np.abs(raw_values) > 1.0))
            row["wrapper_changed_fraction"] = float(np.mean(
                raw_values != applied_scope[:, :, channel]))
            records.append(row)
    return records


def cross_channel_records(raw_predictions, expert, valid):
    applied = applied_controller_actions(raw_predictions).mean(axis=1)
    expert_applied = applied_controller_actions(expert)
    rows = []
    for scope in ("first_action", "valid_chunk"):
        if scope == "first_action":
            prediction, truth = applied[:, 0], expert_applied[:, 0]
        else:
            prediction, truth = applied[valid], expert_applied[valid]
        for predicted in range(13):
            for target in range(13):
                rows.append(dict(scope=scope, predicted_channel=predicted,
                    predicted_name=ACTION_CHANNELS[predicted]["name"], expert_channel=target,
                    expert_name=ACTION_CHANNELS[target]["name"],
                    pearson=pearson(prediction[:, predicted], truth[:, target])))
    return rows


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timeline_records(selected, predictions, expert, valid, group_positions):
    applied = applied_controller_actions(predictions)
    physical = controller_to_physical(applied)
    expert_applied = applied_controller_actions(expert)
    expert_physical = controller_to_physical(expert_applied)
    rows = []
    for i, item in enumerate(selected):
        position, length = group_positions[_row_key(item)]
        progress = position / max(length - 1, 1)
        for offset in np.flatnonzero(valid[i]):
            for channel, contract in enumerate(ACTION_CHANNELS):
                values = physical[i, :, offset, channel]
                raw = predictions[i, :, offset, channel]
                rows.append(dict(parent_id=item["parent_id"], call_index=item["call_index"],
                    tool_family=item["tool_family"], observation_index=item["observation_index"],
                    call_frame_index=position, call_frames=length, call_progress=progress,
                    horizon_offset=int(offset), source_action_index=item["action_source_indices"][offset],
                    channel=channel, channel_name=contract["name"], unit=contract["unit"],
                    expert_controller=float(expert_applied[i, offset, channel]),
                    expert_physical=float(expert_physical[i, offset, channel]),
                    prediction_mean_raw_controller=float(np.mean(raw)),
                    prediction_mean_applied_controller=float(np.mean(applied[i, :, offset, channel])),
                    prediction_mean_physical=float(np.mean(values)),
                    physical_signed_error=float(np.mean(values) - expert_physical[i, offset, channel]),
                    physical_absolute_error=float(abs(np.mean(values) - expert_physical[i, offset, channel])),
                    physical_sample_std=float(np.std(values, ddof=1)),
                    raw_out_of_bounds_fraction=float(np.mean(np.abs(raw) > 1.0))))
    return rows


def time_curve_records(timeline, bins=10):
    first = [row for row in timeline if row["horizon_offset"] == 0]
    groups = defaultdict(list)
    for row in first:
        index = min(int(float(row["call_progress"]) * bins), bins - 1)
        for family in ("all", row["tool_family"]):
            groups[(family, row["channel"], index)].append(row)
    result = []
    for (family, channel, index), values in sorted(groups.items()):
        result.append(dict(tool_family=family, channel=channel,
            channel_name=ACTION_CHANNELS[channel]["name"], unit=ACTION_CHANNELS[channel]["unit"],
            progress_bin=index, progress_center=(index + 0.5) / bins, n=len(values),
            physical_mae=float(np.mean([v["physical_absolute_error"] for v in values])),
            physical_signed_error=float(np.mean([v["physical_signed_error"] for v in values]))))
    return result


def write_curve_svg(path: Path, curves):
    """Dependency-free 13-panel MAE curve over invocation progress."""
    width, height, columns, rows = 1120, 900, 4, 4
    panel_w, panel_h = width / columns, height / rows
    colors = {"all": "#111827", "reach": "#2563eb", "grasp": "#dc2626",
              "move": "#059669", "release": "#9333ea"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.line{fill:none;stroke-width:1.8}</style>']
    for channel, contract in enumerate(ACTION_CHANNELS):
        x0 = (channel % columns) * panel_w + 48
        y0 = (channel // columns) * panel_h + 28
        plot_w, plot_h = panel_w - 65, panel_h - 58
        values = [r for r in curves if r["channel"] == channel]
        ymax = max([r["physical_mae"] for r in values] + [1e-12])
        parts.extend([
            f'<text x="{x0}" y="{y0 - 8}" font-size="12">{channel}: {html.escape(contract["name"])} ({html.escape(contract["unit"])})</text>',
            f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>',
            f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>',
            f'<text x="{x0}" y="{y0 + plot_h + 14}" font-size="9">0</text>',
            f'<text x="{x0 + plot_w - 8}" y="{y0 + plot_h + 14}" font-size="9">1</text>',
            f'<text x="{x0 + 3}" y="{y0 + 10}" font-size="9">{ymax:.3g}</text>',
        ])
        for family in sorted({r["tool_family"] for r in values}):
            points = sorted((r for r in values if r["tool_family"] == family),
                            key=lambda r: r["progress_center"])
            coordinates = " ".join(
                f'{x0 + r["progress_center"] * plot_w:.1f},{y0 + plot_h - r["physical_mae"] / ymax * plot_h:.1f}'
                for r in points)
            if coordinates:
                parts.append(f'<polyline class="line" stroke="{colors.get(family, "#6b7280")}" points="{coordinates}"/>')
    parts.append('<text x="20" y="892" font-size="10">x: normalized invocation progress; y: teacher-forced first-action physical-command MAE. Black=all calls.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _artifact_manifest(output: Path):
    rows = []
    for path in sorted(output.rglob("*")):
        if (not path.is_file() or path.name in {"socket-auth.local", "model.sock", "artifact-manifest.json"}):
            continue
        rows.append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=sha256(path)))
    (output / "artifact-manifest.json").write_text(json.dumps(dict(files=rows), indent=2), encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset-manifest", "checkpoint", "normalizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--parent-id", action="append", type=int, default=[])
    parser.add_argument("--max-observations", type=int, default=128)
    parser.add_argument("--samples-per-observation", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=18092026)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--gpu-uuid", default=GPU1_UUID)
    parser.add_argument("--training-stage", choices=("S1_ordinary_target_domain_SFT_not_TAPT",
                        "S1_IA_single_bank_no_progress"), default="S1_IA_single_bank_no_progress")
    parser.add_argument("--controller-source", type=Path,
        default=Path.home() / "bvi-research/src/AC-DiT/third_party/ManiSkill/mani_skill/agents/robots/fetch/fetch.py")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not 1 <= args.max_observations <= 500:
        raise ValueError("--max-observations must be in [1,500]")
    if not 2 <= args.samples_per_observation <= 8:
        raise ValueError("Stochastic S1 requires 2-8 samples per observation")
    if not 60 <= args.max_seconds <= 7200 or not 10 <= args.request_timeout_seconds <= 300:
        raise ValueError("Invalid bounded runtime")
    if not 0 <= args.base_seed <= 2**32 - 1:
        raise ValueError("--base-seed must be uint32")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = dict(status="preflight", stage="level1_offline_action_reconstruction",
        question="Can frozen S1-IA reconstruct expert Fetch13 actions from recorded observations?",
        scope="teacher-forced recorded observations; no simulator; no rollout; no training; no checkpoint selection",
        split=args.split, samples_per_observation=args.samples_per_observation,
        max_observations=args.max_observations, max_seconds=args.max_seconds,
        training_updates=0, simulator_steps=0, rollout_episodes=0, api_calls=0,
        checkpoint=str(args.checkpoint.resolve()), normalizer=str(args.normalizer.resolve()),
        source=str(args.source.resolve()), dataset_manifest=str(args.dataset_manifest.resolve()),
        physical_metric_definition=("post-quantile-unnormalization, then controller clipping and stationary-head mask; "
            "mapped to sub-controller target delta/target/velocity units, not measured post-physics motion"),
        stochastic_treatment=("independent sequential flow samples at each identical observation; report sample-mean "
            "reconstruction, expected single-sample RMSE, and sample standard deviation"),
        first_action_interpretation="strict teacher-forced reconstruction at recorded observation t",
        chunk_interpretation="open-loop horizon prediction from observation t; reported separately",
        channel_contract=ACTION_CHANNELS, head_mask_indices=list(HEAD_CHANNELS),
        controller_contract_source=str(args.controller_source.resolve()))
    error = None
    server = connection = None
    auth = output / "socket-auth.local"
    server_dir = output / "server"
    try:
        # Imports below belong to the pinned lab model environment, not CPU unit tests.
        from multiprocessing.connection import Client
        import h5py
        from audit_s1_teacher_actions import audit_provenance
        from bvi.ia_fetch_data import invocation_rows, invocation_sample

        manifest, provenance = audit_provenance(
            args.source, args.dataset_manifest, args.normalizer)
        rows = [r for r in invocation_rows(manifest, horizon=10) if r["split"] == args.split]
        if args.parent_id:
            requested = set(args.parent_id)
            rows = [r for r in rows if int(r["parent_id"]) in requested]
            found = {int(r["parent_id"]) for r in rows}
            if found != requested:
                raise ValueError(f"Requested parent ids missing from {args.split}: {sorted(requested - found)}")
        if not rows:
            raise ValueError("No eligible action-supervised rows")
        selected = balanced_selection(rows, args.max_observations)
        calls = len(selected) * args.samples_per_observation
        if calls > 2000:
            raise ValueError("Selected stochastic inference calls exceed frozen server cap 2000")
        grouped = defaultdict(list)
        for row in sorted(rows, key=_row_key):
            grouped[(str(row["parent_id"]), int(row["call_index"]))].append(row)
        group_positions = {}
        for values in grouped.values():
            for position, row in enumerate(values):
                group_positions[_row_key(row)] = (position, len(values))
        summary.update(provenance=provenance, eligible_observations=len(rows),
            eligible_parent_calls=len(grouped), selected_observations=len(selected),
            planned_inference_calls=calls, parent_filter=args.parent_id or None,
            selection=("farthest-point call subset with target four temporally spread frames per call; "
                "round-robin allocation; fixed before inference"),
            selected_roster=[dict(parent_id=r["parent_id"], call_index=r["call_index"],
                tool_family=r["tool_family"], observation_index=r["observation_index"])
                for r in selected],
            controller_contract_source_sha256=(sha256(args.controller_source)
                if args.controller_source.is_file() else None))
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")

        auth.write_bytes(secrets.token_bytes(32))
        auth.chmod(0o600)
        server_dir.mkdir()
        socket_path = server_dir / "model.sock"
        command = [os.sys.executable, str(Path(__file__).with_name("serve_native_s1.py")),
            "--checkpoint", str(args.checkpoint), "--normalizer", str(args.normalizer),
            "--output", str(server_dir), "--socket", str(socket_path), "--auth-file", str(auth),
            "--gpu-uuid", args.gpu_uuid, "--max-inference-calls", str(calls),
            "--training-stage", args.training_stage]
        with (server_dir / "stdout.log").open("x", encoding="utf-8") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while not (server_dir / "ready.json").exists():
            if server.poll() is not None:
                raise RuntimeError("Frozen model server failed before readiness")
            if time.monotonic() - started > min(300, args.max_seconds):
                raise TimeoutError("Frozen model loading timeout")
            time.sleep(1)
        connection = Client(str(socket_path), family="AF_UNIX", authkey=auth.read_bytes())

        predicted, experts, masks, rngs, latencies = [], [], [], [], []
        with h5py.File(args.source, "r") as source:
            for row_index, row in enumerate(selected):
                if time.monotonic() - started >= args.max_seconds:
                    raise TimeoutError("Offline reconstruction wall limit reached")
                sample = invocation_sample(source[row["trajectory"]], row)
                expert = np.asarray(sample["actions"], np.float32)
                valid = ~np.asarray(sample["actions_is_pad"], bool)
                row_seed_material = json.dumps([args.base_seed, row["parent_id"], row["call_index"],
                    row["observation_index"]], separators=(",", ":")).encode()
                seed = (args.base_seed + int.from_bytes(hashlib.sha256(row_seed_material).digest()[:4], "little")) % 2**32
                connection.send(dict(op="reset", seed=seed))
                if not connection.poll(min(10, args.request_timeout_seconds)):
                    raise TimeoutError("Model reset timeout")
                reset = connection.recv()
                if reset.get("status") != "reset":
                    raise RuntimeError(f"Model reset failed: {reset}")
                samples, sample_rngs, sample_latencies = [], [], []
                for _ in range(args.samples_per_observation):
                    remaining = args.max_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Offline reconstruction wall limit reached")
                    request = dict(op="predict", seed=seed, head_rgb=sample["image"],
                        wrist_rgb=sample["wrist_image"], state=sample["state"], prompt=sample["prompt"])
                    connection.send(request)
                    if not connection.poll(min(args.request_timeout_seconds, remaining)):
                        raise TimeoutError("Offline model prediction timeout")
                    response = connection.recv()
                    if response.get("status") != "ok":
                        raise RuntimeError(f"Model inference failed: {response}")
                    actions = np.asarray(response["actions"], np.float32)
                    if actions.shape != (10, 13) or not np.isfinite(actions).all():
                        raise ValueError("Frozen server returned invalid action chunk")
                    samples.append(actions)
                    sample_rngs.append(response["rng"])
                    sample_latencies.append(response["inference_seconds"])
                predicted.append(samples)
                experts.append(expert)
                masks.append(valid)
                rngs.append(sample_rngs)
                latencies.append(sample_latencies)
                summary.update(status="running", completed_observations=row_index + 1,
                    completed_inference_calls=(row_index + 1) * args.samples_per_observation,
                    wall_seconds=time.monotonic() - started)
                (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")

        predictions = np.asarray(predicted, np.float32)
        expert = np.asarray(experts, np.float32)
        valid = np.asarray(masks, bool)
        np.savez_compressed(output / "reconstruction-arrays.npz", predictions_raw_controller=predictions,
            predictions_applied_controller=applied_controller_actions(predictions).astype(np.float32),
            predictions_physical=controller_to_physical(applied_controller_actions(predictions)).astype(np.float32),
            expert_controller=expert, expert_physical=controller_to_physical(expert).astype(np.float32),
            action_valid=valid, rng=np.asarray(rngs, np.uint32), inference_seconds=np.asarray(latencies, np.float32))

        per_channel = per_channel_records(predictions, expert, valid)
        cross_channel = cross_channel_records(predictions, expert, valid)
        timeline = timeline_records(selected, predictions, expert, valid, group_positions)
        curves = time_curve_records(timeline)
        write_csv(output / "per-channel.csv", per_channel)
        write_csv(output / "cross-channel-correlation.csv", cross_channel)
        write_csv(output / "timeline.csv", timeline)
        write_csv(output / "time-curve.csv", curves)
        write_curve_svg(output / "time-curves.svg", curves)

        best_cross = {}
        for scope in ("first_action", "valid_chunk"):
            for predicted_channel in range(13):
                values = [r for r in cross_channel if r["scope"] == scope
                          and r["predicted_channel"] == predicted_channel and r["pearson"] is not None]
                if values:
                    best = max(values, key=lambda r: abs(r["pearson"]))
                    best_cross[f"{scope}:{predicted_channel}"] = best
        server_metadata = json.loads((server_dir / "metadata.json").read_text())
        summary.update(status="completed", completed_observations=len(selected),
            completed_inference_calls=calls, wall_seconds=time.monotonic() - started,
            mean_inference_seconds=float(np.mean(latencies)),
            p95_inference_seconds=float(np.quantile(latencies, 0.95)),
            first_action_physical_metrics=[r for r in per_channel if r["scope"] == "first_action"],
            best_cross_channel_correlations=best_cross,
            model_identity={key: server_metadata[key] for key in ("checkpoint", "pretrained_parameters_sha256",
                "normalizer_sha256", "state_contract_sha256", "rng_contract")},
            outputs=["per-channel.csv", "cross-channel-correlation.csv", "timeline.csv",
                "time-curve.csv", "time-curves.svg", "reconstruction-arrays.npz"])
    except Exception as exc:
        error = exc
        summary.update(status="failed", error=repr(exc), wall_seconds=time.monotonic() - started)
    finally:
        if connection is not None:
            connection.close()
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        auth.unlink(missing_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
        _artifact_manifest(output)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
