from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# New headings to pull
HEADINGS = [
    "ClinicalTrials.gov",
    "IUPHAR/BPS Guide to PHARMACOLOGY Target Classification",
]


def _sha1(s: str) -> str:
    """
    Calculate SHA-1 hash of a string.
    
    Used to create cache filenames for API responses.
    Args:
        s: String to hash
    Returns:
        Hexadecimal SHA-1 hash of the input string
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _read_id_list(path: Path, max_ids: int | None) -> list[int]:
    """
    Read a list of CIDs (compound IDs) from a text file.
    
    Supports multiple formats: comma-separated, semicolon-separated, or one per line.
    Validates that entries are numeric and deduplicates results.
    
    Args:
        path: Path to the CID list file
        max_ids: Optional maximum number of IDs to return (truncates if exceeded)
    Returns:
        Sorted list of unique integer CIDs
    """
    raw = path.read_text(encoding="utf-8").splitlines()
    vals: list[int] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        for part in line.replace(",", ";").split(";"):
            p = part.strip()
            if not p:
                continue
            if not p.isdigit():
                continue
            vals.append(int(p))
    vals = sorted(set(vals))
    if max_ids is not None:
        vals = vals[:max_ids]
    return vals


def _pugview_url(cid: int, heading: str) -> str:
    """
    Construct a PubChem PugView API URL for a specific compound heading.
    
    Args:
        cid: Compound ID (CID) to fetch
        heading: Section heading name to retrieve
    Returns:
        Full URL string for the PugView API endpoint
    """
    h = requests.utils.quote(heading, safe="")
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={h}"


def _extract_clinicaltrials_data(j: dict[str, Any]) -> dict[str, Any]:
    """
    Extract ClinicalTrials.gov data from PugView JSON.
    
    ClinicalTrials data is usually stored as an ExternalTableName reference
    with a count of associated trials.
    
    Args:
        j: PugView JSON response for ClinicalTrials.gov heading
    Returns:
        Dictionary with trial_count and raw_json fields
    """
    trial_count = None
    
    # Navigate nested structure to find the Information section
    try:
        sections = j.get("Record", {}).get("Section", [])
        for sec in sections:
            for subsec in sec.get("Section", []):
                for subsubsec in subsec.get("Section", []):
                    if subsubsec.get("TOCHeading") == "ClinicalTrials.gov":
                        infos = subsubsec.get("Information", [])
                        for info in infos:
                            val = info.get("Value", {})
                            if "ExternalTableNumRows" in val:
                                trial_count = val["ExternalTableNumRows"]
                                break
    except Exception:
        pass
    
    return {
        "clinicaltrials_count": trial_count,
        "clinicaltrials_json": json.dumps(j, ensure_ascii=False) if j else None,
    }


def _extract_iuphar_data(j: dict[str, Any]) -> dict[str, Any]:
    """
    Extract IUPHAR/BPS Target Classification data from PugView JSON.
    
    IUPHAR data is stored as a hierarchical ID (HID) reference.
    
    Args:
        j: PugView JSON response for IUPHAR heading
    Returns:
        Dictionary with HID and raw_json fields
    """
    hid = None
    
    # Navigate nested structure to find the HID
    try:
        sections = j.get("Record", {}).get("Section", [])
        for sec in sections:
            for subsec in sec.get("Section", []):
                if "IUPHAR" in subsec.get("TOCHeading", ""):
                    infos = subsec.get("Information", [])
                    for info in infos:
                        if info.get("Name") == "HID":
                            val = info.get("Value", {})
                            nums = val.get("Number", [])
                            if nums:
                                hid = nums[0]
                                break
    except Exception:
        pass
    
    return {
        "iuphar_hid": hid,
        "iuphar_json": json.dumps(j, ensure_ascii=False) if j else None,
    }


def main():
    """
    Pull additional PubChem headings (ClinicalTrials.gov and IUPHAR).
    
    Fetches two new headings for each CID and combines into a single table.
    Supports caching, resume capability, and progress tracking.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)  # File with CID list to process
    ap.add_argument("--outdir", required=True)  # Directory for output parquet file
    ap.add_argument("--max-cids", type=int, default=None)  # Optional limit on number of CIDs
    ap.add_argument("--flush-every", type=int, default=1000)  # Write parquet every N CIDs
    ap.add_argument("--sleep", type=float, default=0.1)  # Sleep seconds between API requests
    ap.add_argument("--cache-dir", default="pubchem_bulk_pull/cache_json")  # Cache directory
    ap.add_argument("--resume", action="store_true")  # Resume from last checkpoint
    args = ap.parse_args()
    
    # Set up output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Set up cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Track processed CIDs for resume functionality
    processed_file = outdir / "processed_cids.txt"
    processed: set[int] = set()
    if args.resume and processed_file.exists():
        for line in processed_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit():
                processed.add(int(line))
    
    print("[stage] setup")
    print("outdir:", str(outdir))
    print("cache_dir:", str(cache_dir) if cache_dir else "None")
    
    # Load CID list
    cids_all = _read_id_list(Path(args.cid_list), args.max_cids)
    cids = [c for c in cids_all if c not in processed]
    print("cids_total_in_list:", len(cids_all))
    print("already_processed:", len(processed))
    print("cids_to_process:", len(cids))
    
    print("[stage] pulling additional headings per CID")
    rows: list[dict[str, Any]] = []
    processed_now: list[int] = []
    part_idx = 0
    
    t0 = time.time()
    for i, cid in enumerate(cids, start=1):
        # Initialize row for this CID
        row = {"cid": cid}
        
        # Fetch both headings
        for heading in HEADINGS:
            url = _pugview_url(cid, heading)
            cache_path = (cache_dir / f"{_sha1(url)}.json") if cache_dir else None
            
            # Check cache first
            j: dict[str, Any] | None = None
            if cache_path and cache_path.exists():
                try:
                    j = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            # Fetch from API if not cached
            if j is None:
                try:
                    r = requests.get(url, timeout=60)
                    if r.status_code == 200:
                        j = r.json()
                    else:
                        j = {"error": {"message": f"HTTP {r.status_code}"}}
                except Exception as e:
                    j = {"error": {"message": str(e)}}
                
                # Cache the response
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
            
            # Extract data based on heading type
            if heading == "ClinicalTrials.gov":
                data = _extract_clinicaltrials_data(j)
                row.update(data)
            elif "IUPHAR" in heading:
                data = _extract_iuphar_data(j)
                row.update(data)
            
            # Sleep between requests to be respectful to API
            if args.sleep:
                time.sleep(args.sleep)
        
        # Add completed row
        rows.append(row)
        processed_now.append(cid)
        
        # Log progress every 25 CIDs
        if i % 25 == 0 or i == len(cids):
            rate = i / max(1e-9, (time.time() - t0))
            print(f"processed {i}/{len(cids)} | rate={rate:.2f} cid/s")
        
        # Periodically flush to disk
        if args.flush_every and (i % args.flush_every == 0 or i == len(cids)):
            part_idx += 1
            df = pd.DataFrame(rows)
            
            # Write parquet file
            part_file = outdir / f"part-{part_idx:05d}.parquet"
            df.to_parquet(part_file, index=False)
            print(f"[flush] wrote {part_file} with {len(df)} rows")
            
            # Update processed CIDs file
            if processed_now:
                with processed_file.open("a", encoding="utf-8") as f:
                    for cid in processed_now:
                        f.write(f"{cid}\n")
            
            # Clear accumulators
            rows.clear()
            processed_now.clear()
    
    print("[done] COMPLETE")
    print("outdir:", str(outdir))
    print("total_parts:", part_idx)


if __name__ == "__main__":
    main()
