# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Fetch ISLES'24 parquet shards from the Hugging Face mirror (hugging-science/isles24-stroke), one at a
time, resumably. Each shard is one subject (150-260 MB; 149 shards, ~99 GB in all).

    python scripts/fetch_isles24.py /path/to/isles24 50        # make sure shards 0..49 are under <root>/data/

Skips shards already on disk, stops cleanly when free space drops below 3 GB, retries a flaky transfer three
times, and can simply be re-run to resume. Needs `pip install huggingface_hub`.
"""
import os, shutil, sys, time


def main(root: str, n: int):
    data = os.path.join(root, "data"); os.makedirs(data, exist_ok=True)
    from huggingface_hub import hf_hub_download
    names = [f"train-{i:05d}-of-00149.parquet" for i in range(n)]
    todo = [m for m in names if not os.path.exists(os.path.join(data, m))]
    print(f"{n - len(todo)} of {n} shards already present; fetching {len(todo)}", flush=True)
    t0 = time.time()
    for k, name in enumerate(todo, 1):
        free_gb = shutil.disk_usage(root).free / 1e9
        if free_gb < 3:
            print(f"stopping: only {free_gb:.1f} GB free. Free some space and re-run.", flush=True); sys.exit(1)
        print(f"[{k}/{len(todo)}] {name}  ({free_gb:.0f} GB free)", flush=True)
        for attempt in range(3):
            try:
                hf_hub_download("hugging-science/isles24-stroke", filename=f"data/{name}", repo_type="dataset", local_dir=root)
                break
            except Exception as e:
                print(f"   retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True); time.sleep(10)
        else:
            print(f"   giving up on {name} for now; re-run later", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; {len(os.listdir(data))} files in {data}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
