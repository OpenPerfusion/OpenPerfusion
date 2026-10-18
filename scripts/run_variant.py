# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Run the ISLES 2018 (or ISLES'24) agreement harness with one PipelineConfig variant.

    python scripts/run_variant.py isles2018 /path/to/TRAINING results/variants/snr3 --cfg snr_min_tissue=3
    python scripts/run_variant.py isles2024 /path/to/shards results/variants/snr3_24 --cfg snr_min_tissue=3 [--max N]

`--cfg` takes comma-separated key=value pairs over PipelineConfig's defaults (numbers and strings).
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_isles2018 import run_config
from openperfusion.pipeline import PipelineConfig


def parse_cfg(s: str) -> dict:
    """key=value pairs separated by commas; a value may be a number, true/false/none, a string, or a
    parenthesised tuple such as aif_weights=(2,-1,-1)."""
    import ast, re
    out = {}
    for kv in filter(None, re.split(r",(?![^()]*\))", s)):
        k, v = kv.split("=", 1); v = v.strip()
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = {"true": True, "false": False, "none": None}.get(v.lower(), v)
        out[k.strip()] = v
    return out


def main(kind: str, root: str, out: Path, cfg: dict, max_n: int):
    out.mkdir(parents=True, exist_ok=True)
    if kind == "isles2018":
        from openperfusion.io_isles2018 import find_cases
        items, loader = find_cases(root), None
    else:
        from openperfusion.io_isles2024 import find_shards, load_subject_from_parquet
        items, loader = find_shards(root), load_subject_from_parquet
    t0 = time.time()
    summ, df = run_config(items[:max_n], PipelineConfig(**cfg), **({"loader": loader} if loader else {}))
    df.to_csv(out / "cases.csv", index=False)
    json.dump({"config": cfg, **summ, "seconds": round(time.time() - t0, 1)}, open(out / "summary.json", "w"), indent=2, default=str)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], Path(a[2]), parse_cfg(a[a.index("--cfg") + 1]) if "--cfg" in a else {}, int(a[a.index("--max") + 1]) if "--max" in a else 10**6)
