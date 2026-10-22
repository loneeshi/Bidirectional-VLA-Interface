"""Evaluation-only cross-process snapshot primitives (not a release runner).

Raw GPU row numbers are process-local. Callers must supply complete, verified
stable row identities; historical snapshots without this evidence fail closed.
These helpers do not approve restoration or weaken the native equality gate.
"""
from __future__ import annotations

import copy
import json

CONTRACT = "bvi-eval-snapshot-transport-v2"


def capture_transport_reference(uenv):
    """Evaluator reference with parent-qualified link identities, no ordinals."""
    from export_c2_collision_bundle import capture_reference, _body_state
    result = capture_reference(uenv)
    bodies = {}
    def add(identity, body, dynamic):
        stable = json.dumps(identity, separators=(',', ':'))
        if stable in bodies:
            raise ValueError('ambiguous stable scene reference identity')
        bodies[stable] = _body_state(body, dynamic=dynamic)
    for actor in uenv.scene.actors.values():
        add(['actor', actor.name], actor, actor.px_body_type in ('dynamic','kinematic'))
    for art in uenv.scene.articulations.values():
        if art is uenv.agent.robot or art.name == uenv.agent.robot.name:
            continue
        for link in art.links:
            add(['other_link', art.name, link.name], link, True)
    if len(bodies) != len(result['scene_bodies']):
        raise ValueError('stable reference body coverage differs')
    result['scene_bodies'] = bodies
    return result


def clone_snapshot_value(value):
    """Copy state without invoking dataclass constructors or __post_init__.

    deepcopy preserves init=False fields, shared references and cycles. Native
    checkpoint tensors are detached leaves; unsupported deepcopy types raise
    instead of being reconstructed with potentially different semantics.
    """
    from collections import deque
    from dataclasses import fields, is_dataclass
    from enum import Enum
    import numpy as np
    seen = set()
    def validate(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if item is None or isinstance(item, (str, bytes, int, float, bool, Enum,
                                              np.ndarray, np.generic, np.random.RandomState,
                                              np.random.Generator)):
            return
        if type(item).__name__ in ('Tensor', 'Generator', 'BatchedRNG', 'Pose'):
            return
        if isinstance(item, dict):
            for k, v in item.items():
                validate(k); validate(v)
        elif isinstance(item, (tuple, list, set, deque)):
            for child in item: validate(child)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item): validate(getattr(item, field.name))
        else:
            raise TypeError(type(item).__name__)
    validate(value)
    return copy.deepcopy(value)


def remap_buffer_rows(value, source_ids, destination_ids):
    """Reorder a full buffer using evaluator-recorded stable body identities.

    IDs must identify rows, including articulation links and environment index,
    rather than just object classes. No nearest-position identity inference is
    permitted here. Duplicate, missing, extra and partial coverage are errors.
    The original buffer is never mutated, including for an identity mapping.
    """
    source_ids, destination_ids = tuple(source_ids), tuple(destination_ids)
    for ids in (source_ids, destination_ids):
        if any(not isinstance(key, str) or not key for key in ids):
            raise ValueError("nonempty stable row identities required")
        if len(set(ids)) != len(ids):
            raise ValueError("ambiguous duplicate row identity")
    if not hasattr(value, "shape") or len(value.shape) < 1:
        raise ValueError("row-indexed buffer required")
    if value.shape[0] != len(source_ids):
        raise ValueError("incomplete source row coverage")
    if set(source_ids) != set(destination_ids):
        raise ValueError("source/destination row identities differ")
    index = {key: i for i, key in enumerate(source_ids)}
    order = [index[key] for key in destination_ids]
    return clone_snapshot_value(value[order])


def native_buffer_layout(scene):
    """Read every native row's identity from the live scene, not its position."""
    from .executor_diagnostics import GPU_BUFFERS
    layouts = {key: [None] * getattr(scene.px, key).torch().shape[0]
               for key in GPU_BUFFERS}

    def bind(key, indices, envs, identity):
        indices, envs = indices.tolist(), envs.tolist()
        if len(indices) != len(envs):
            raise ValueError('row/environment binding length differs')
        for row, env in zip(indices, envs):
            if type(row) is not int or not 0 <= row < len(layouts[key]):
                raise ValueError('native row index outside buffer')
            name = json.dumps([env, *identity], separators=(',', ':'))
            if layouts[key][row] is not None:
                raise ValueError('native buffer row bound twice')
            layouts[key][row] = name

    for actor in scene.actors.values():
        if actor.px_body_type != 'static':
            bind('cuda_rigid_body_data', actor._body_data_index,
                 actor._scene_idxs, ['actor', actor.name])
    for art in scene.articulations.values():
        for key in GPU_BUFFERS:
            if key != 'cuda_rigid_body_data':
                bind(key, art._data_index, art._scene_idxs, ['articulation', art.name])
        for link in art.links:
            bind('cuda_rigid_body_data', link._body_data_index,
                 link._scene_idxs, ['link', art.name, link.name])
    for key, names in layouts.items():
        # The pinned native allocator includes zero-filled alignment holes.
        # Only rows not claimed by any scene body AND exactly zero may be padding.
        # They remain in the full comparison, with equal padding counts required
        # across processes. A nonzero unbound row still fails closed.
        missing = [i for i, name in enumerate(names) if name is None]
        if missing and not bool((getattr(scene.px,key).torch()[missing] != 0).any()):
            for ordinal, row in enumerate(missing):
                names[row] = json.dumps(['zero_padding', ordinal], separators=(',', ':'))
        if any(name is None for name in names) or len(set(names)) != len(names):
            error = ValueError('native buffer identity coverage incomplete: ' + key)
            missing = [i for i, name in enumerate(names) if name is None]
            error.receipt = {'buffer':key, 'row_count':len(names),
                             'bound_rows':len(names)-len(missing), 'missing_rows':missing,
                             'unbound_values':getattr(scene.px,key).torch()[missing].tolist()}
            raise error
    return layouts


def capture_transport(base):
    """New explicit contract; legacy capture defaults remain unchanged."""
    from check_c2_snapshot_restore import capture
    return {'contract': CONTRACT, 'layout': native_buffer_layout(base.uenv.scene),
            'state': capture(base, clone_fn=clone_snapshot_value)}


def prepare_transport_restore(checkpoint, destination_layout):
    """Validate all bindings before any native write; never adapt v1 silently."""
    from .executor_diagnostics import validate_snapshot, GPU_BUFFERS
    if set(checkpoint) != {'contract', 'layout', 'state'} or checkpoint['contract'] != CONTRACT:
        raise ValueError('explicit v2 checkpoint required; historical rows unbound')
    validate_snapshot(checkpoint['state'])
    if set(checkpoint['layout']) != set(GPU_BUFFERS) or set(destination_layout) != set(GPU_BUFFERS):
        raise ValueError('all native buffer layouts required')
    expected = clone_snapshot_value(checkpoint['state'])
    expected['gpu_buffers'] = {
        key: remap_buffer_rows(checkpoint['state']['gpu_buffers'][key],
                               checkpoint['layout'][key], destination_layout[key])
        for key in GPU_BUFFERS}
    return expected


def restore_transport(base, checkpoint, *, observer=None):
    """Apply v2 and require complete state equivalence, without a physics step."""
    from check_c2_snapshot_restore import restore, capture
    from .executor_diagnostics import snapshot_comparison
    layout = native_buffer_layout(base.uenv.scene)
    expected = prepare_transport_restore(checkpoint, layout)
    if observer is not None:
        observer('restore_before', dict(layout=layout, state=capture(base, clone_fn=clone_snapshot_value)))
        observer('restore_expected', dict(layout=layout, state=expected))
    try:
        restore(base, expected, mode='buffers', clone_fn=clone_snapshot_value)
    except BaseException as error:
        if observer is not None:
            try:observer('restore_partial', dict(layout=layout, state=capture(base, clone_fn=clone_snapshot_value)))
            except BaseException as secondary:error.diagnostic_error=repr(secondary)
        raise
    actual = capture(base, clone_fn=clone_snapshot_value)
    diagnostic_error = None
    if observer is not None:
        try:observer('restore_actual', dict(layout=layout, state=actual))
        except BaseException as secondary:diagnostic_error=secondary
    try:
        receipt = snapshot_comparison(expected, actual)
    except BaseException as error:
        if diagnostic_error is not None:error.diagnostic_error=repr(diagnostic_error)
        raise
    receipt['contract'] = CONTRACT
    error = None
    if not receipt['passed']:
        error = ValueError('v2 zero-action full state equivalence failed')
        error.receipt = receipt
    if observer is not None:
        try:
            if diagnostic_error is not None:raise diagnostic_error
            observer('restore_comparison', dict(receipt=receipt, expected=expected, actual=actual))
        except BaseException as secondary:
            if error is None:raise
            error.diagnostic_error=repr(secondary)
    if error is not None:raise error
    return receipt
