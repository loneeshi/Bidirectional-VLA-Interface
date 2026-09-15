"""Summarize fixed evaluations and link every attempted GPT call to usage."""

import argparse, collections, csv, json, pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evidence = pathlib.Path(a.evidence)
    bridge = pathlib.Path(a.bridge)
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    calls = []
    for mode in ["baseline", "vlm-standard", "vlm-tapt"]:
        summary = json.loads((evidence / mode / "summary.json").read_text(encoding='utf-8'))
        assert len(summary) == 5 and [r["episode"] for r in summary] == list(range(5))
        for r in summary:
            trace = [
                json.loads(x)
                for x in (evidence / mode / f"episode{r['episode']:03d}.jsonl")
                .read_text(encoding='utf-8')
                .splitlines()
            ]
            ids = [e["attempt_id"] for e in trace if e["event"] == "vlm_request"]
            cost = 0.0
            unknown = 0
            for aid in ids:
                f = bridge / (aid + ".response.json")
                data = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
                response = data.get("response", {})
                usage = response.get("usage")
                amount = None
                if usage:
                    d = usage.get("input_tokens_details") or {}
                    cache = d.get("cached_tokens", 0)
                    write = d.get("cache_write_tokens", 0)
                    amount = (
                        (usage["input_tokens"] - cache - write) * 0.2
                        + cache * 0.02
                        + write * 0.25
                        + usage["output_tokens"] * 1.2
                    ) / 1e6
                    cost += amount
                else:
                    unknown += 1
                calls.append(
                    {
                        "mode": mode,
                        "episode": r["episode"],
                        "attempt_id": aid,
                        "request_id": response.get("request_id"),
                        "usage": usage,
                        "estimated_usd": amount,
                        "provider_bill_reconciled": False,
                    }
                )
            action = [e for e in trace if e["event"] == "action"]
            wrong = None
            if mode != "baseline":
                wrong = sum(
                    any(
                        obj != e["diagnostic_only"]["requested_object"]
                        for obj in e["diagnostic_only"][
                            "dual_finger_contact_grasp_candidates"
                        ]
                    )
                    for e in action
                )
            row = dict(
                r,
                configuration=mode,
                api_estimated_usd=cost,
                api_unknown_attempts=unknown,
                diagnostic_wrong_object_contact_steps=wrong,
            )
            assert len(action) == r["steps"] and len(ids) == r["vlm_calls"]
            assert not any(e.get("native_success") for e in action[:-1])
            rows.append(row)
    baseline = [
        r["initial_state_sha256"] for r in rows if r["configuration"] == "baseline"
    ]
    for mode in ["vlm-standard", "vlm-tapt"]:
        assert baseline == [
            r["initial_state_sha256"] for r in rows if r["configuration"] == mode
        ]
    result = {
        "episodes": rows,
        "api_calls": calls,
        "known_api_estimated_usd": sum(
            c["estimated_usd"] for c in calls if c["estimated_usd"] is not None
        ),
        "outcomes": {
            m: {
                "successes": sum(r["success"] for r in rows if r["configuration"] == m),
                "episodes": 5,
            }
            for m in ["baseline", "vlm-standard", "vlm-tapt"]
        },
        "note": "Rule-vs-learned combined-method comparison, not a causal TAPT isolation. Contact diagnostic is not grasp-force ground truth.",
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    fields = [
        "configuration",
        "episode",
        "success",
        "steps",
        "vlm_calls",
        "api_estimated_usd",
        "api_unknown_attempts",
        "termination_reason",
        "switch_reasons",
        "diagnostic_wrong_object_contact_steps",
        "video",
        "error",
    ]
    with (out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(r, switch_reasons=json.dumps(r.get("switch_reasons", {}), sort_keys=True)) for r in rows)
    print(json.dumps(result["outcomes"], indent=2))


if __name__ == "__main__":
    main()
