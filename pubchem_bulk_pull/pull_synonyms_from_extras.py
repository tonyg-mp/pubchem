from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pandas as pd


def read_cids(path: Path) -> set[int]:
    out: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.replace(",", ";").split(";"):
            p = part.strip()
            if p.isdigit():
                out.add(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)
    ap.add_argument("--cid-synonym-gz", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--log-every", type=int, default=5_000_000)
    args = ap.parse_args()

    wanted = read_cids(Path(args.cid_list))
    gz_path = Path(args.cid_synonym_gz)

    print("[stage] inputs")
    print("wanted_cids:", len(wanted))
    print("gz_path:", str(gz_path))

    rows = []
    seen = set()
    lines = 0
    matched = 0
    t0 = time.time()

    print("[stage] streaming")
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines += 1
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            cid_str, syn = parts[0].strip(), parts[1].strip()
            if not cid_str.isdigit() or not syn:
                continue
            cid = int(cid_str)
            if cid not in wanted:
                if args.log_every and lines % args.log_every == 0:
                    rate = lines / max(1e-9, time.time() - t0)
                    print(f"lines={lines} matched={matched} rate={rate:.0f} lines/s")
                continue

            key = (cid, syn)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"cid": cid, "synonym": syn})
            matched += 1

            if args.log_every and lines % args.log_every == 0:
                rate = lines / max(1e-9, time.time() - t0)
                print(f"lines={lines} matched={matched} rate={rate:.0f} lines/s")

    df = pd.DataFrame(rows)
    outp = Path(args.out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    print("[done] wrote", str(outp))
    print("rows:", len(df))
    print("unique_cids:", df["cid"].nunique() if not df.empty else 0)


if __name__ == "__main__":
    main()
