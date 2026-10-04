# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Unpack ISLES'24 parquet shards (the Hugging Face mirror hugging-science/isles24-stroke stores one subject per
shard, each NIfTI as a {bytes, path} struct) into per-subject folders of NIfTI files plus subjects.csv with the
clinical fields (age, sex, NIHSS, mRS).

    python scripts/unpack_isles24.py isles24/data isles24/unpacked

Download shards with huggingface_hub, e.g. the first ten subjects:
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('hugging-science/isles24-stroke',
        repo_type='dataset', allow_patterns=['data/train-%05d-of-00149.parquet' % i for i in range(10)], local_dir='isles24')"
"""
import sys, os, glob, csv
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(root, "train-*.parquet"))):
    try:
        t = pq.read_table(f)
    except Exception as e:
        print("skip", f, e); continue
    for r in t.to_pylist():
        sid = r["subject_id"]; d = os.path.join(out, sid); os.makedirs(d, exist_ok=True)
        rec = {"subject_id": sid}; n = 0
        for k, v in r.items():
            if isinstance(v, dict) and "bytes" in v:
                if v["bytes"] is None:
                    continue
                name = os.path.basename(v.get("path") or f"{sid}_{k}.nii.gz")
                p = os.path.join(d, name)
                if not os.path.exists(p):
                    with open(p, "wb") as fh:
                        fh.write(v["bytes"])
                rec[k] = name; n += 1
            elif not isinstance(v, dict):
                rec[k] = v
        rows.append(rec); print(sid, "->", n, "files", flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "subject_id", k))
with open(os.path.join(out, "subjects.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", os.path.join(out, "subjects.csv"))
