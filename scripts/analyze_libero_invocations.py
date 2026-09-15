"""Join learned events to logged simulator diagnostics without changing decisions."""

import argparse, json, pathlib


def analyze(file):
    rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
    actions = {}
    result = []
    for row in rows:
        if row["event"] == "action":
            actions.setdefault(row["call_id"], []).append(row)
        if row["event"] != "invocation_finished":
            continue
        events = actions.get(row["call_id"], [])
        last = events[-1] if events else None
        contacts = [
            r["diagnostic_only"]["dual_finger_contact_grasp_candidates"] for r in events
        ]
        result.append(
            dict(
                row,
                episode=file.stem,
                last_simulator_rule_diagnostic=last["simulator_rule"] if last else None,
                requested_object_contact_steps=sum(
                    row["target"] in c for c in contacts
                ),
                other_object_contact_steps=sum(
                    any(x != row["target"] for x in c) for c in contacts
                ),
                last_object_positions=last["diagnostic_only"]["object_positions"]
                if last
                else None,
                diagnostic_caveat="Contact and rule checks are imperfect diagnostics; these values were not fed to the learned-condition planner.",
            )
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    records = [
        row
        for file in sorted(pathlib.Path(a.directory).glob("episode*.jsonl"))
        for row in analyze(file)
    ]
    pathlib.Path(a.output).write_text(json.dumps(records, indent=2))
    print(
        {
            "invocations": len(records),
            "learned_events": sum(r["reason"].startswith("learned_") for r in records),
            "threshold_events_with_false_rule_diagnostic": sum(
                r["reason"] == "learned_threshold"
                and r["last_simulator_rule_diagnostic"] is False
                for r in records
            ),
        }
    )
