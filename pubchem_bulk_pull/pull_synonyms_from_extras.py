from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pandas as pd


def read_cids(path: Path) -> set[int]:
    """
    Read a set of CIDs from a text file.
    
    Supports comma-separated, semicolon-separated, or newline-separated formats.
    Returns deduplicated integer CIDs.
    
    Args:
        path: Path to the CID list file
    Returns:
        Set of unique integer CIDs
    """
    out: set[int] = set()
    # Read all lines from file
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Handle both comma and semicolon separators
        for part in line.replace(",", ";").split(";"):
            p = part.strip()
            # Only add numeric values
            if p.isdigit():
                out.add(int(p))
    return out


def main() -> None:
    """
    Stream a gzip-compressed PubChem CID-Synonym file and extract synonyms for wanted CIDs.
    
    The bulk file (CID-Synonym.gz) contains millions of lines in format: CID<tab>Synonym
    This function streams it line-by-line to avoid loading everything in memory,
    filters for only the wanted CIDs, and deduplicates (CID, synonym) pairs.
    
    Output: Parquet file with columns [cid, synonym]
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)  # File with CIDs to extract
    ap.add_argument("--cid-synonym-gz", required=True)  # Gzip-compressed CID-Synonym bulk file
    ap.add_argument("--out-parquet", required=True)  # Output parquet file
    ap.add_argument("--log-every", type=int, default=5_000_000)  # Log progress every N lines
    args = ap.parse_args()

    # Load the set of CIDs we want to extract
    wanted = read_cids(Path(args.cid_list))
    gz_path = Path(args.cid_synonym_gz)

    print("[stage] inputs")
    print("wanted_cids:", len(wanted))
    print("gz_path:", str(gz_path))

    # Accumulators for output data
    rows = []
    # Track seen (CID, synonym) pairs to avoid duplicates
    seen = set()
    lines = 0  # Total lines processed
    matched = 0  # Lines that matched a wanted CID
    t0 = time.time()

    print("[stage] streaming")
    # Stream through the gzip file line by line (memory efficient)
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines += 1
            line = line.rstrip("\n")
            if not line:
                continue
            # Split on first tab only (synonym might contain tabs)
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            cid_str, syn = parts[0].strip(), parts[1].strip()
            # Validate CID is numeric and synonym is not empty
            if not cid_str.isdigit() or not syn:
                continue
            cid = int(cid_str)
            # Skip if this CID is not in our wanted list
            if cid not in wanted:
                if args.log_every and lines % args.log_every == 0:
                    rate = lines / max(1e-9, time.time() - t0)
                    print(f"lines={lines} matched={matched} rate={rate:.0f} lines/s")
                continue

            # Deduplicate (CID, synonym) pairs
            key = (cid, syn)
            if key in seen:
                continue
            seen.add(key)
            # Add to output
            rows.append({"cid": cid, "synonym": syn})
            matched += 1

            # Log progress
            if args.log_every and lines % args.log_every == 0:
                rate = lines / max(1e-9, time.time() - t0)
                print(f"lines={lines} matched={matched} rate={rate:.0f} lines/s")

    # Write results to parquet
    df = pd.DataFrame(rows)
    outp = Path(args.out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    print("[done] wrote", str(outp))
    print("rows:", len(df))
    print("unique_cids:", df["cid"].nunique() if not df.empty else 0)


if __name__ == "__main__":
    main()
