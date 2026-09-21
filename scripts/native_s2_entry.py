"""Admission for NEW native24 S2 only; never edits historical training holds."""

from bvi.s1_capability_gate import validate_s1_capability
from bvi.s1_evidence import build_report
from bvi.native24_handoff import validate_manifest


def validate_entry(panel, best_record, handoff_manifest, training_manifest=None):
    if any(x is None for x in (panel, best_record, handoff_manifest, training_manifest)):
        raise ValueError(
            'New S1 panel, heldout best record and native24 handoff evidence required; '
            'training manifest required'
        )
    native = build_report(panel, best_record)
    admission = validate_s1_capability(native, native['checkpoint_sha256'])
    handoff = validate_manifest(handoff_manifest, training_manifest)
    return dict(native=native, admission=admission,
                handoff_sha256=handoff['manifest_sha256'],
                handoff_parents=handoff['handoff_parents'],
                handoff_input_gate=handoff,
                budget_increase=False, historical_pipeline_resume=False)
