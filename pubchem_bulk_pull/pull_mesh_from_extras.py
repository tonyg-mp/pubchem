from __future__ import annotations

import argparse
import json
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


def load_mesh_pharm_map(path: Path) -> dict[str, list[str]]:
    """
    Load MeSH to Pharmacological Class mapping.
    
    File format: Each line is MeSH_term followed by tab-separated pharmacological classes.
    Example: "Kinase Inhibitors\tProtein Kinase Inhibitors\tTyrosine Kinase Inhibitors"
    
    Args:
        path: Path to MeSH-Pharm mapping file
    Returns:
        Dictionary mapping MeSH term -> list of pharmacological classes
    """
    m: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Split on tabs and strip whitespace from each part
            parts = [p.strip() for p in line.split("\t")]
            # First part is the MeSH term, rest are pharmacological classes
            term = parts[0] if parts else ""
            classes = [c for c in parts[1:] if c]
            # Only add if both term and classes are present
            if term and classes:
                m[term] = classes
    return m


def main() -> None:
    """
    Extract MeSH pharmacological classifications for wanted CIDs.
    
    Process:
    1. Load CID-MeSH associations from bulk file
    2. Load MeSH to Pharmacological Class mappings
    3. Cross-reference: for each CID's MeSH terms, look up their pharmacological classes
    4. Output enriched data with CID -> pharmacological class mappings
    
    Output: Parquet file with columns [cid, mesh_name, mesh_description, reference_number, raw_info_json]
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)  # File with wanted CIDs
    ap.add_argument("--extras-dir", required=True)  # Directory containing CID-MeSH and MeSH-Pharm files
    ap.add_argument("--out-parquet", required=True)  # Output parquet file
    args = ap.parse_args()

    # Load wanted CIDs
    wanted = read_cids(Path(args.cid_list))
    extras = Path(args.extras_dir)
    cid_mesh_path = extras / "CID-MeSH"
    mesh_pharm_path = extras / "MeSH-Pharm"

    print("[stage] inputs")
    print("wanted_cids:", len(wanted))
    print("cid_mesh_path:", str(cid_mesh_path))
    print("mesh_pharm_path:", str(mesh_pharm_path))

    print("[stage] load MeSH-Pharm")
    # Create lookup table: MeSH term -> list of pharmacological classes
    mesh_to_classes = load_mesh_pharm_map(mesh_pharm_path)
    print("mesh_terms_with_classes:", len(mesh_to_classes))

    print("[stage] stream CID-MeSH")
    # Accumulate output rows
    rows = []
    # Track (CID, class, term) tuples to avoid duplicates
    seen = set()

    # Stream through CID-MeSH file
    with cid_mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            cid_str = parts[0].strip()
            # Validate CID is numeric
            if not cid_str.isdigit():
                continue
            cid = int(cid_str)
            # Skip if CID not in wanted list
            if cid not in wanted:
                continue

            # Extract all MeSH terms for this CID
            mesh_terms = [p.strip() for p in parts[1:] if p.strip()]
            # For each MeSH term, look up its pharmacological classes
            for mesh_term in mesh_terms:
                classes = mesh_to_classes.get(mesh_term)
                # Skip if MeSH term has no pharmacological class mapping
                if not classes:
                    continue
                # Create a row for each pharmacological class
                for cls in classes:
                    # Deduplicate
                    key = (cid, cls, mesh_term)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "cid": cid,
                            "mesh_name": cls,  # The pharmacological class name
                            "mesh_description": None,
                            "reference_number": None,
                            # Store raw info as JSON for reference
                            "raw_info_json": json.dumps(
                                {"mesh_term": mesh_term, "pharm_class": cls},
                                ensure_ascii=False,
                            ),
                        }
                    )

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
