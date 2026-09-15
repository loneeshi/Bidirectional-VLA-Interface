"""Audit actual evaluation traces without trusting a video or summary alone."""

import argparse, collections, json, pathlib


def audit(path):
    rows = [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf-8').splitlines()]
    current = None
    queue = collections.deque()
    finished = set()
    instructions = {}
    families = {}
    hashes = {}
    calls = 0
    actions = 0
    learned = 0
    rejected = 0
    for row in rows:
        event = row["event"]
        if event == "vlm_response":
            call = row["attempt_id"]
            inv = row["invocation"]
            assert call not in instructions and row["request_id"]
            assert not queue and current is None
            current = call
            instructions[call] = inv["instruction"]
            families[call] = inv["tool_family"]
            calls += 1
        elif event == "learned_progress":
            assert (
                row["call_id"] == current
                and row["instruction"] == instructions[current]
                and row["family"] == families[current]
            )
            h = row["adapter_sha256"]
            assert len(h) == 64
            if row["family"] in hashes:
                assert hashes[row["family"]] == h
            hashes[row["family"]] = h
        elif event == "prediction":
            assert (
                row["call_id"] == current
                and row["instruction"] == instructions[current]
                and not queue
            )
            queue.extend(row["actions"][:5])
        elif event == "action":
            assert current not in finished and row["call_id"] == current and queue
            assert row["action"] == queue.popleft()
            actions += 1
        elif event == "invocation_finished":
            assert row["call_id"] == current and row["discarded_actions"] == len(queue)
            finished.add(current)
            queue.clear()
            current = None
            if row["reason"].startswith("learned_"):
                learned += 1
        elif event == "error":
            # A rejected request is an episode failure, not an executed invalid action.
            assert row["error"] == "ValueError('Instruction too long')"
            assert current is None and not queue
            rejected += 1
    assert not queue and current is None and calls <= 20 and actions <= 520
    return {
        "file": str(path),
        "calls": calls,
        "actions": actions,
        "learned_events": learned,
        "rejected_requests": rejected,
        "adapter_hashes": hashes,
        "passed": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    a = p.parse_args()
    print(
        json.dumps(
            [
                audit(x)
                for x in sorted(pathlib.Path(a.directory).glob("episode*.jsonl"))
            ],
            indent=2,
        )
    )
