import concurrent.futures, hashlib, json, pathlib, urllib.request

ROOT = pathlib.Path("/workspace/tapt/data")
ROOT.mkdir(exist_ok=True)
BASE = "https://huggingface.co/datasets/physical-intelligence/libero/resolve/a4336d589d589045d1c56423ffdf3b88a0e19b1f/"


def read(path):
    return urllib.request.urlopen(BASE + path).read()


episodes = [json.loads(x) for x in read("meta/episodes.jsonl").decode().splitlines()]
task = "put both the cream cheese box and the butter in the basket"
selected = sorted(
    [e for e in episodes if task in e["tasks"]], key=lambda e: e["episode_index"]
)
assert len(selected) >= 25, len(selected)
selected = selected[:25]


def fetch(e):
    i = e["episode_index"]
    p = f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet"
    payload = read(p)
    out = ROOT / pathlib.Path(p).name
    out.write_bytes(payload)
    return dict(
        e, path=str(out), sha256=hashlib.sha256(payload).hexdigest(), source=BASE + p
    )


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    result = list(pool.map(fetch, selected))
(ROOT / "manifest.json").write_text(
    json.dumps(dict(task=task, train=result[:20], validation=result[20:]), indent=2)
)
print("downloaded", len(result), flush=True)
