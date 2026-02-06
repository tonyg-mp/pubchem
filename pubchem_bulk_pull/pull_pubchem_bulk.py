from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


HEADINGS = [
    "3D Conformer",
    "Depositor-Supplied Synonyms",
    "Depositor-Supplied Patent Identifiers",
    "MeSH Pharmacological Classification",
    "FDA Pharmacological Classification",
    "Related Records",
    "Chemical Vendors",
    "Patents",
]

RAW_HEADINGS = {
    "3D Conformer": "pugview_3d_conformer_json",
    "Related Records": "pugview_related_records_json",
    "Chemical Vendors": "pugview_chemical_vendors_json",
    "Patents": "pugview_patents_heading_json",
}


def _sha1(s: str) -> str:
    """
    Calculate SHA-1 hash of a string.
    
    Used to create cache filenames for API responses.
    Args:
        s: String to hash
    Returns:
        Hexadecimal SHA-1 hash of the input string
    """
    # Encode string to UTF-8 bytes and compute SHA-1 hash
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
    # Read all lines from the file
    raw = path.read_text(encoding="utf-8").splitlines()
    vals: list[int] = []
    for line in raw:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Handle both comma and semicolon separators by normalizing to semicolons
        for part in line.replace(",", ";").split(";"):
            p = part.strip()
            if not p:
                continue
            # Only process numeric values
            if not p.isdigit():
                continue
            vals.append(int(p))
    # Remove duplicates and sort
    vals = sorted(set(vals))
    # Optionally limit to max_ids
    if max_ids is not None:
        vals = vals[:max_ids]
    return vals


def _fetch_json(url: str, cache_path: Path | None, timeout: int = 30) -> dict[str, Any]:
    """
    Fetch JSON from a URL with optional caching.
    
    Attempts to read from cache first if cache_path is provided and exists.
    Falls back to HTTP request and caches the result. Handles HTTP errors and
    JSON parsing failures gracefully by returning error objects.
    
    Args:
        url: URL to fetch JSON from
        cache_path: Optional path to cache file; if None, no caching is used
        timeout: Request timeout in seconds (default: 30)
    Returns:
        Dictionary (parsed JSON) or error dict if fetch/parse fails
    """
    # Check cache first if cache_path is provided
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            # Cache file corrupted or unreadable, continue to fetch
            pass

    try:
        # Fetch from URL
        r = requests.get(url, timeout=timeout)
        try:
            # Try to parse as JSON
            j = r.json()
        except Exception:
            # If JSON parsing fails, create an error object
            j = {"error": {"message": f"HTTP {r.status_code} (non-json)"}}

        # Cache the result if cache path is provided
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")

        return j
    except Exception as e:
        # Network/request error - create error object
        j = {"error": {"message": str(e)}}
        # Cache error responses too
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
        return j


def _pugview_url(cid: int, heading: str) -> str:
    """
    Construct a PubChem PugView API URL for a specific compound heading.
    
    Args:
        cid: Compound ID (CID) to fetch
        heading: Section heading name to retrieve (e.g., 'MeSH Pharmacological Classification')
    Returns:
        Full URL string for the PugView API endpoint
    """
    # URL-encode the heading parameter (spaces become %20, etc.)
    h = requests.utils.quote(heading, safe="")
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={h}"


def _get_pugrest_properties_batch(cids: list[int]) -> dict[int, dict[str, Any]]:
    """
    Fetch core chemical properties for multiple CIDs in a single batch request.
    
    Retrieves: InChIKey, SMILES, ConnectivitySMILES, MolecularFormula, MolecularWeight.
    Uses the PUG REST API which is more efficient than individual PugView requests.
    
    Args:
        cids: List of compound IDs to fetch properties for
    Returns:
        Dictionary mapping CID -> properties dict; empty dict on HTTP error
    """
    # Construct comma-separated list of CIDs
    cid_list = ",".join(str(c) for c in cids)
    props = "InChIKey,SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}/property/{props}/JSON"
    # Fetch from PubChem REST API
    r = requests.get(url, timeout=60)
    # Return empty dict if request failed
    if r.status_code != 200:
        return {}
    j = r.json()
    # Extract Properties array from response
    rows = j.get("PropertyTable", {}).get("Properties", []) or []
    # Map each result by CID for quick lookup
    out: dict[int, dict[str, Any]] = {}
    for rec in rows:
        cid = rec.get("CID")
        if cid is None:
            continue
        out[int(cid)] = rec
    return out


def _extract_strings_with_markup(j: dict[str, Any]) -> list[str]:
    """
    Recursively extract all plain text strings from PugView JSON structure.
    
    PugView responses often contain text in {"StringWithMarkup": [{"String": "text"}]} format.
    This function walks the entire JSON tree and flattens all such strings into a single list.
    
    Args:
        j: PugView JSON response dictionary
    Returns:
        Flattened list of all extracted strings
    """
    out: list[str] = []

    # Nested function to recursively walk the JSON tree
    def walk(x: Any):
        if isinstance(x, dict):
            # Look for StringWithMarkup arrays in this dict
            if "StringWithMarkup" in x and isinstance(x["StringWithMarkup"], list):
                for itm in x["StringWithMarkup"]:
                    # Extract the String field from each item
                    if isinstance(itm, dict) and isinstance(itm.get("String"), str):
                        out.append(itm["String"])
            # Recursively walk all other values in the dict
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            # Recursively walk list items
            for v in x:
                walk(v)

    # Start traversal from root
    walk(j)
    return out


def _extract_mesh_pharm_rows(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract MeSH pharmacological classification data from PugView JSON.
    
    Navigates the nested Section/Information structure in PugView responses for
    the MeSH Pharmacological Classification heading. Extracts names, descriptions,
    and reference numbers for each classification.
    
    Args:
        cid: Compound ID being processed
        j: PugView JSON response for MeSH heading
    Returns:
        List of dictionaries with fields: cid, mesh_name, mesh_description, reference_number, raw_info_json
    """
    rows: list[dict[str, Any]] = []
    # Navigate to Record.Section (top-level sections)
    sections = (j.get("Record") or {}).get("Section") or []
    for sec in sections:
        # Navigate to subsections within each section
        for subsec in (sec.get("Section") or []):
            # Each subsection contains Information items
            infos = subsec.get("Information") or []
            for info in infos:
                # Extract name and reference from information item
                name = info.get("Name")
                ref = info.get("ReferenceNumber")
                desc = None
                # Extract description from StringWithMarkup value (use first item only)
                val = info.get("Value") or {}
                swm = val.get("StringWithMarkup")
                if isinstance(swm, list) and swm:
                    s0 = swm[0]
                    if isinstance(s0, dict):
                        desc = s0.get("String")
                # Create row with all extracted data
                rows.append(
                    {
                        "cid": cid,
                        "mesh_name": name,
                        "mesh_description": desc,
                        "reference_number": ref,
                        "raw_info_json": json.dumps(info, ensure_ascii=False),
                    }
                )
    return rows


_FDA_SUFFIX_RE = re.compile(r"^(?P<name>.+?)\s+\[(?P<typ>[A-Za-z]+)\]$")
_FDA_GROUP_RE = re.compile(r"^(?P<group>.+?)\s+\[(?P<typ>[A-Za-z]+)\]\s*-\s*(?P<name>.+)$")


def _extract_fda_pharm_rows(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract FDA pharmacological classification data from PugView JSON.
    
    Parses text using two regex patterns:
    - _FDA_GROUP_RE: "GroupName [Type] - ClassName" (e.g., "Kinase Inhibitor [EPC] - Tyrosine Kinase Inhibitor")
    - _FDA_SUFFIX_RE: "ClassName [Type]" (e.g., "Anti-inflammatory [EPC]")
    
    Args:
        cid: Compound ID being processed
        j: PugView JSON response for FDA heading
    Returns:
        List of dictionaries with fields: cid, class_type, class_group, class_name, raw_text
    """
    rows: list[dict[str, Any]] = []
    # Extract all text strings from the JSON
    strings = _extract_strings_with_markup(j)

    for s in strings:
        if not isinstance(s, str):
            continue
        # Split on semicolons to handle multiple classifications in one string
        parts = [p.strip() for p in s.split(";") if p.strip()]
        for p in parts:
            # Try matching the "Group [Type] - Name" pattern first
            m2 = _FDA_GROUP_RE.match(p)
            if m2:
                rows.append(
                    {
                        "cid": cid,
                        "class_type": m2.group("typ"),  # e.g., "EPC"
                        "class_group": m2.group("group"),  # e.g., "Kinase Inhibitor"
                        "class_name": m2.group("name"),  # e.g., "Tyrosine Kinase Inhibitor"
                        "raw_text": p,
                    }
                )
                continue

            # Try matching the "Name [Type]" pattern
            m1 = _FDA_SUFFIX_RE.match(p)
            if m1:
                rows.append(
                    {
                        "cid": cid,
                        "class_type": m1.group("typ"),  # e.g., "EPC"
                        "class_group": None,  # Not present in this pattern
                        "class_name": m1.group("name"),  # e.g., "Anti-inflammatory"
                        "raw_text": p,
                    }
                )
                continue
            # If neither pattern matches, skip (the unreachable code below is a leftover)
            continue

            rows.append(
                {
                    "cid": cid,
                    "class_type": None,
                    "class_group": None,
                    "class_name": None,
                    "raw_text": p,
                }
            )

    return rows


def _extract_depositor_patent_ids(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract US patent IDs from the Depositor-Supplied Patent Identifiers heading.
    
    Looks for strings matching the pattern "US" followed by digits (e.g., "US1234567").
    Deduplicates and sorts results.
    
    Args:
        cid: Compound ID being processed
        j: PugView JSON response for patent IDs heading
    Returns:
        List of dictionaries with fields: cid, patent_id
    """
    # Extract all text strings from the JSON
    strings = _extract_strings_with_markup(j)
    # Use a set to automatically deduplicate
    ids = set()
    for s in strings:
        # Match strings that are exactly "US" followed by digits
        if isinstance(s, str) and re.fullmatch(r"US\d+", s.strip()):
            ids.add(s.strip())
    # Return as list of dicts, sorted by patent_id
    return [{"cid": cid, "patent_id": pid} for pid in sorted(ids)]


def _extract_depositor_synonyms(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract depositor-supplied synonym names from PugView JSON.
    
    Extracts plain text strings and filters out navigation text like "Link to all..."
    which are not actual synonyms.
    
    Args:
        cid: Compound ID being processed
        j: PugView JSON response for synonyms heading
    Returns:
        List of dictionaries with fields: cid, synonym
    """
    # Extract all text strings from the JSON
    strings = _extract_strings_with_markup(j)
    syns = []
    for s in strings:
        if not isinstance(s, str):
            continue
        t = s.strip()
        # Skip empty strings
        if not t:
            continue
        # Filter out navigation links (these are not actual synonyms)
        if "Link to all" in t or "link to all" in t:
            continue
        syns.append(t)
    # Return as list of dicts
    return [{"cid": cid, "synonym": s} for s in syns]


def _write_parts(base: Path, folder: str, part_idx: int, df: pd.DataFrame) -> None:
    """
    Write a DataFrame to a parquet file in a partitioned directory structure.
    
    Creates numbered files (part-00001.parquet, part-00002.parquet, etc.) to allow
    output to be split across multiple files for easier processing.
    
    Args:
        base: Base output directory
        folder: Subdirectory name within base (e.g., 'core_properties')
        part_idx: Partition/part number (used in filename)
        df: DataFrame to write
    """
    # Skip empty DataFrames
    if df.empty:
        return
    # Create output directory
    outdir = base / folder
    outdir.mkdir(parents=True, exist_ok=True)
    # Write with zero-padded part number (5 digits)
    path = outdir / f"part-{part_idx:05d}.parquet"
    df.to_parquet(path, index=False)


def main():
    """
    Main orchestrator for pulling PubChem data. Coordinates the entire pipeline:
    1. Fetches core chemical properties (SMILES, InChIKey, etc.)
    2. Fetches PugView headings (synonyms, patents, classifications, etc.)
    3. Extracts and parses classification data from JSON responses
    4. Saves all data to parquet files organized by category
    
    Supports resumable operation via --resume flag with processed CID tracking.
    HTTP responses are optionally cached to avoid re-fetching.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)  # File with CID list to process
    ap.add_argument("--outdir", required=True)  # Directory for output parquet files
    ap.add_argument("--max-cids", type=int, default=None)  # Optional limit on number of CIDs
    ap.add_argument("--flush-every", type=int, default=250)  # Write parquet files every N CIDs
    ap.add_argument("--sleep", type=float, default=0.1)  # Sleep seconds between API requests
    ap.add_argument("--cache-dir", default="pubchem_bulk_pull/cache_json")  # Cache directory for API responses
    ap.add_argument("--properties-batch-size", type=int, default=100)  # Compounds per batch for properties API
    ap.add_argument("--resume", action="store_true")  # Resume from last checkpoint
    args = ap.parse_args()
    
    # Set up output directory structure
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Set up cache directory for API responses if specified
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    processed_file = outdir / "processed_cids.txt"
    # Track which CIDs have already been processed (for resume functionality)
    processed: set[int] = set()
    if args.resume and processed_file.exists():
        # Load previously processed CIDs
        for line in processed_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit():
                processed.add(int(line))

    print("[stage] setup")
    print("outdir:", str(outdir))
    print("cache_dir:", str(cache_dir) if cache_dir else "None")

    # Reads the list of CIDs to process
    cids_all = _read_id_list(Path(args.cid_list), args.max_cids)
    cids = [c for c in cids_all if c not in processed]
    print("cids_total_in_list:", len(cids_all))
    print("already_processed:", len(processed))
    print("cids_to_process:", len(cids))

    print("[stage] pulling properties in batches")
    core_rows: list[dict[str, Any]] = []
    part_idx = 0

    for start in range(0, len(cids), args.properties_batch_size):
        batch = cids[start : start + args.properties_batch_size]
        if not batch:
            continue
        props_map = _get_pugrest_properties_batch(batch)
        for cid in batch:
            rec = props_map.get(cid)
            if not rec:
                core_rows.append(
                    {
                        "cid": cid,
                        "inchikey": None,
                        "smiles": None,
                        "connectivity_smiles": None,
                        "molecular_formula": None,
                        "molecular_weight": None,
                    }
                )
            else:
                core_rows.append(
                    {
                        "cid": cid,
                        "inchikey": rec.get("InChIKey"),
                        "smiles": rec.get("SMILES"),
                        "connectivity_smiles": rec.get("ConnectivitySMILES"),
                        "molecular_formula": rec.get("MolecularFormula"),
                        "molecular_weight": rec.get("MolecularWeight"),
                    }
                )
        if (start // args.properties_batch_size + 1) % 10 == 0 or start + args.properties_batch_size >= len(cids):
            done = min(start + args.properties_batch_size, len(cids))
            print(f"properties_batch_done {done}/{len(cids)}")

    core_df = pd.DataFrame(core_rows)

    print("[stage] pulling PUG-View headings per CID")
    # Separate lists for different heading types that will be written to different parquet files
    heading_meta_rows: list[dict[str, Any]] = []  # HTTP metadata for each heading
    depositor_syn_rows: list[dict[str, Any]] = []  # Synonyms data
    depositor_pat_rows: list[dict[str, Any]] = []  # Patent ID data
    mesh_rows: list[dict[str, Any]] = []  # MeSH pharmacological classifications
    fda_rows: list[dict[str, Any]] = []  # FDA pharmacological classifications
    raw_heading_rows: list[dict[str, Any]] = []  # Raw JSON for non-parsed headings
    processed_now: list[int] = []  # CIDs processed since last flush

    def flush():
        """
        Write accumulated rows to parquet files and clear buffers.
        Called periodically to avoid holding too much data in memory.
        Also updates the processed CIDs tracking file for resume functionality.
        """
        nonlocal part_idx
        part_idx += 1
        # Write all accumulated data to numbered part files
        _write_parts(outdir, "core_properties", part_idx, core_df if part_idx == 1 else pd.DataFrame())
        _write_parts(outdir, "heading_meta", part_idx, pd.DataFrame(heading_meta_rows))
        _write_parts(outdir, "depositor_synonyms", part_idx, pd.DataFrame(depositor_syn_rows))
        _write_parts(outdir, "depositor_patent_ids", part_idx, pd.DataFrame(depositor_pat_rows))
        _write_parts(outdir, "mesh_pharm_class", part_idx, pd.DataFrame(mesh_rows))
        _write_parts(outdir, "fda_pharm_class", part_idx, pd.DataFrame(fda_rows))
        _write_parts(outdir, "pugview_raw_headings", part_idx, pd.DataFrame(raw_heading_rows))

        # Append newly processed CIDs to tracking file
        if processed_now:
            with processed_file.open("a", encoding="utf-8") as f:
                for cid in processed_now:
                    f.write(f"{cid}\n")

        # Clear all accumulators for next batch
        heading_meta_rows.clear()
        depositor_syn_rows.clear()
        depositor_pat_rows.clear()
        mesh_rows.clear()
        fda_rows.clear()
        raw_heading_rows.clear()
        processed_now.clear()

    t0 = time.time()
    for i, cid in enumerate(cids, start=1):
        for heading in HEADINGS:
            url = _pugview_url(cid, heading)
            cache_path = (cache_dir / f"{_sha1(url)}.json") if cache_dir else None

            r = requests.get(url, timeout=60)
            http_code = r.status_code
            body = r.content or b""
            nbytes = len(body)
            j: dict[str, Any] | None = None
            if http_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = None
            else:
                try:
                    j = r.json()
                except Exception:
                    j = {"error": {"message": f"HTTP {http_code}"}}

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")

            heading_meta_rows.append(
                {
                    "cid": cid,
                    "heading": heading,
                    "http_code": http_code,
                    "bytes": nbytes,
                }
            )

            if http_code != 200 or j is None:
                continue

            if heading == "Depositor-Supplied Synonyms":
                depositor_syn_rows.extend(_extract_depositor_synonyms(cid, j))
            elif heading == "Depositor-Supplied Patent Identifiers":
                depositor_pat_rows.extend(_extract_depositor_patent_ids(cid, j))
            elif heading == "MeSH Pharmacological Classification":
                mesh_rows.extend(_extract_mesh_pharm_rows(cid, j))
            elif heading == "FDA Pharmacological Classification":
                fda_rows.extend(_extract_fda_pharm_rows(cid, j))
            elif heading in RAW_HEADINGS:
                raw_heading_rows.append(
                    {
                        "cid": cid,
                        "heading": heading,
                        "column_name": RAW_HEADINGS[heading],
                        "json": json.dumps(j, ensure_ascii=False),
                    }
                )

            if args.sleep:
                time.sleep(args.sleep)

        processed_now.append(cid)

        if i % 25 == 0 or i == len(cids):
            rate = i / max(1e-9, (time.time() - t0))
            print(f"processed {i}/{len(cids)} | rate={rate:.2f} cid/s")

        if args.flush_every and (i % args.flush_every == 0):
            flush()

    flush()

    if not (outdir / "core_properties").exists():
        (outdir / "core_properties").mkdir(parents=True, exist_ok=True)
        core_df.to_parquet(outdir / "core_properties" / "part-00001.parquet", index=False)

    print("DONE")
    print("outdir:", str(outdir))
    print("processed_cids_file:", str(processed_file))


if __name__ == "__main__":
    main()
