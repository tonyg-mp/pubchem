from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_parts(folder: Path) -> pd.DataFrame:
    """
    Load and concatenate multiple parquet files from a directory.
    
    Reads all files matching "part-*.parquet" pattern, sorts them alphabetically,
    and concatenates into a single DataFrame.
    
    Args:
        folder: Directory containing parquet files
    Returns:
        DataFrame with concatenated data, or empty DataFrame if no files found
    """
    # Find all part files and sort them
    parts = sorted(folder.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    # Concatenate and reset index
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _json_dumps(x: Any) -> str:
    """
    Serialize object to JSON string, preserving Unicode characters.
    
    Args:
        x: Object to serialize
    Returns:
        JSON string with ensure_ascii=False to preserve Unicode
    """
    return json.dumps(x, ensure_ascii=False)


def _try_json_loads(x: Any) -> Any:
    """
    Safely parse JSON string, returning None if parsing fails.
    
    Args:
        x: Potential JSON string to parse
    Returns:
        Parsed object, or None if x is not a string or JSON parsing fails
    """
    # Only attempt to parse strings
    if not isinstance(x, str) or not x:
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _safe_preview_list(vals: list[str], k: int = 20) -> str:
    """
    Create a preview string from a list of values.
    
    Takes the first k items, filters to strings only, and joins with "; ".
    Used for Excel preview columns to show sample data without full content.
    
    Args:
        vals: List of values to preview
        k: Maximum number of items to include (default: 20)
    Returns:
        String preview joined by "; "
    """
    # Filter to only string values
    vals = [v for v in vals if isinstance(v, str)]
    # Limit to k items
    vals = vals[:k]
    # Join with semicolon separator
    return "; ".join(vals)


def _truncate_for_excel(s: Any, limit: int = 30000) -> Any:
    """
    Truncate string to fit Excel cell character limit.
    
    Excel has a ~32k character limit per cell. This function truncates with
    an ellipsis indicator if needed. Non-strings are returned unchanged.
    
    Args:
        s: Value to potentially truncate
        limit: Character limit (default: 30000 to leave margin)
    Returns:
        Original value if string is short, truncated with "…" if too long, or original non-string
    """
    # Return non-strings unchanged
    if not isinstance(s, str):
        return s
    # Return as-is if short enough
    if len(s) <= limit:
        return s
    # Truncate and add ellipsis
    return s[:limit] + "…"


def main():
    """
    Consolidate all parquet files into a single wide-format table.
    
    Process:
    1. Load all individual parquet files (core properties, synonyms, patents, classifications)
    2. Aggregate multi-value fields (synonyms, patents, classifications) into lists and JSON
    3. Join all data by CID into a wide format
    4. Output as parquet (full data) and Excel (preview, limited rows)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory with *.parquet OR outdir root containing part folders.")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--out-xlsx", required=True)
    ap.add_argument("--xlsx-rows", type=int, default=5000)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    # Load data: check if consolidated parquet files or part folder structure
    if (in_dir / "core_properties.parquet").exists():
        # Load consolidated parquet files (single file per data type)
        core = pd.read_parquet(in_dir / "core_properties.parquet")
        heading_meta = pd.read_parquet(in_dir / "heading_meta.parquet") if (in_dir / "heading_meta.parquet").exists() else pd.DataFrame()
        syn = pd.read_parquet(in_dir / "depositor_synonyms.parquet") if (in_dir / "depositor_synonyms.parquet").exists() else pd.DataFrame()
        pat = pd.read_parquet(in_dir / "depositor_patent_ids.parquet") if (in_dir / "depositor_patent_ids.parquet").exists() else pd.DataFrame()
        mesh = pd.read_parquet(in_dir / "mesh_pharm_class.parquet") if (in_dir / "mesh_pharm_class.parquet").exists() else pd.DataFrame()
        fda = pd.read_parquet(in_dir / "fda_pharm_class.parquet") if (in_dir / "fda_pharm_class.parquet").exists() else pd.DataFrame()
        rawh = pd.read_parquet(in_dir / "pugview_raw_headings.parquet") if (in_dir / "pugview_raw_headings.parquet").exists() else pd.DataFrame()
        titles = pd.read_parquet(in_dir / "cid_title.parquet") if (in_dir / "cid_title.parquet").exists() else pd.DataFrame()
    else:
        # Load partitioned data (multiple part-XXXXX.parquet files in folders)
        core = _read_parts(in_dir / "core_properties")
        heading_meta = _read_parts(in_dir / "heading_meta")
        syn = _read_parts(in_dir / "depositor_synonyms")
        pat = _read_parts(in_dir / "depositor_patent_ids")
        mesh = _read_parts(in_dir / "mesh_pharm_class")
        fda = _read_parts(in_dir / "fda_pharm_class")
        rawh = _read_parts(in_dir / "pugview_raw_headings")
        titles = _read_parts(in_dir / "cid_title")

    if core.empty or "cid" not in core.columns:
        raise SystemExit("core_properties missing or empty")

    # Deduplicate core properties (remove duplicate CID rows)
    core = core.drop_duplicates(subset=["cid"]).copy()

    # Aggregate titles (one per CID, use first if multiple exist)
    title_agg = pd.DataFrame({"cid": core["cid"]})
    if not titles.empty and "cid" in titles.columns:
        if "title" not in titles.columns:
            titles["title"] = None
        titles = titles.dropna(subset=["cid", "title"]).copy()
        titles["cid"] = titles["cid"].astype(int)
        titles["title"] = titles["title"].astype(str)
        titles = titles.drop_duplicates(subset=["cid"])
        # Merge to add title column
        title_agg = title_agg.merge(titles[["cid", "title"]], on="cid", how="left")
    else:
        title_agg["title"] = None

    # Aggregate synonyms (multiple per CID)
    syn_agg = pd.DataFrame({"cid": core["cid"]})
    if not syn.empty:
        # Clean and deduplicate synonym data
        syn = syn.dropna(subset=["cid", "synonym"]).copy()
        syn["synonym"] = syn["synonym"].astype(str)
        syn = syn.drop_duplicates(subset=["cid", "synonym"])
        # Group by CID to create lists of synonyms
        g = syn.groupby("cid")["synonym"].apply(list).reset_index(name="synonyms_list")
        syn_agg = syn_agg.merge(g, on="cid", how="left")
    else:
        syn_agg["synonyms_list"] = None

    # Helper function to get first synonym
    def first_syn(x: Any) -> Any:
        if isinstance(x, list) and x:
            return x[0]
        return None

    # Create synonym columns: JSON, count, preview, and first synonym as fallback name
    syn_agg["synonyms_json"] = syn_agg["synonyms_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    syn_agg["synonyms_n"] = syn_agg["synonyms_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    syn_agg["synonyms_preview"] = syn_agg["synonyms_list"].apply(lambda x: _safe_preview_list(x, 25) if isinstance(x, list) else "")
    syn_agg["primary_name_from_synonyms"] = syn_agg["synonyms_list"].apply(first_syn)

    # Aggregate patent IDs (multiple per CID)
    pat_agg = pd.DataFrame({"cid": core["cid"]})
    if not pat.empty:
        # Clean and deduplicate patent data
        pat = pat.dropna(subset=["cid", "patent_id"]).copy()
        pat["patent_id"] = pat["patent_id"].astype(str)
        pat = pat.drop_duplicates(subset=["cid", "patent_id"])
        # Group by CID to create lists of patent IDs
        g = pat.groupby("cid")["patent_id"].apply(list).reset_index(name="patent_ids_list")
        pat_agg = pat_agg.merge(g, on="cid", how="left")
    else:
        pat_agg["patent_ids_list"] = None

    # Create patent columns: JSON, count, and preview
    pat_agg["patent_ids_json"] = pat_agg["patent_ids_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    pat_agg["patent_ids_n"] = pat_agg["patent_ids_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    pat_agg["patent_ids_preview"] = pat_agg["patent_ids_list"].apply(lambda x: _safe_preview_list(x, 25) if isinstance(x, list) else "")

    # Aggregate MeSH pharmacological classifications (multiple per CID)
    mesh_agg = pd.DataFrame({"cid": core["cid"]})
    if not mesh.empty:
        # Clean MeSH data
        mesh = mesh.dropna(subset=["cid"]).copy()
        cols = ["mesh_name", "mesh_description", "reference_number", "raw_info_json"]
        # Ensure all expected columns exist
        for c in cols:
            if c not in mesh.columns:
                mesh[c] = None

        # Helper to convert rows to structured objects with parsed JSON
        def rows_to_objs(df: pd.DataFrame) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for _, r in df.iterrows():
                # Parse raw_info_json if present
                raw_obj = _try_json_loads(r.get("raw_info_json"))
                out.append(
                    {
                        "name": r.get("mesh_name"),
                        "description": r.get("mesh_description"),
                        "reference_number": r.get("reference_number"),
                        "raw_info": raw_obj,
                    }
                )
            return out

        # Group by CID to create lists of mesh classification objects
        g = mesh.groupby("cid").apply(rows_to_objs).reset_index(name="mesh_classes_list")
        mesh_agg = mesh_agg.merge(g, on="cid", how="left")
    else:
        mesh_agg["mesh_classes_list"] = None

    # Create mesh columns: JSON, count, and preview
    mesh_agg["mesh_classes_json"] = mesh_agg["mesh_classes_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    mesh_agg["mesh_classes_n"] = mesh_agg["mesh_classes_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    mesh_agg["mesh_classes_preview"] = mesh_agg["mesh_classes_list"].apply(
        lambda x: _safe_preview_list([o.get("name") for o in x if isinstance(o, dict)], 25) if isinstance(x, list) else ""
    )

    fda_agg = pd.DataFrame({"cid": core["cid"]})
    if not fda.empty:
        fda = fda.dropna(subset=["cid"]).copy()
        cols = ["class_type", "class_group", "class_name", "raw_text"]
        for c in cols:
            if c not in fda.columns:
                fda[c] = None
        fda = fda.drop_duplicates(subset=["cid", "class_type", "class_group", "class_name", "raw_text"])

        def rows_to_objs(df: pd.DataFrame) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for _, r in df.iterrows():
                out.append(
                    {
                        "class_type": r.get("class_type"),
                        "class_group": r.get("class_group"),
                        "class_name": r.get("class_name"),
                        "raw_text": r.get("raw_text"),
                    }
                )
            return out

        g = fda.groupby("cid").apply(rows_to_objs).reset_index(name="fda_classes_list")
        fda_agg = fda_agg.merge(g, on="cid", how="left")
    else:
        fda_agg["fda_classes_list"] = None

    fda_agg["fda_classes_json"] = fda_agg["fda_classes_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    fda_agg["fda_classes_n"] = fda_agg["fda_classes_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    fda_agg["fda_classes_preview"] = fda_agg["fda_classes_list"].apply(
        lambda x: _safe_preview_list([o.get("raw_text") for o in x if isinstance(o, dict) and o.get("raw_text")], 15) if isinstance(x, list) else ""
    )

    rawh_agg = pd.DataFrame({"cid": core["cid"]})
    if not rawh.empty:
        rawh = rawh.dropna(subset=["cid", "column_name", "json"]).copy()
        rawh = rawh.drop_duplicates(subset=["cid", "column_name"])
        piv = rawh.pivot(index="cid", columns="column_name", values="json").reset_index()
        rawh_agg = rawh_agg.merge(piv, on="cid", how="left")

    hm_agg = pd.DataFrame({"cid": core["cid"]})
    if not heading_meta.empty:
        heading_meta = heading_meta.dropna(subset=["cid", "heading", "http_code"]).copy()

        def to_map(df: pd.DataFrame) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for _, r in df.iterrows():
                out[str(r["heading"])] = int(r["http_code"])
            return out

        g = heading_meta.groupby("cid").apply(to_map).reset_index(name="heading_http_map")
        hm_agg = hm_agg.merge(g, on="cid", how="left")
    else:
        hm_agg["heading_http_map"] = None

    hm_agg["heading_http_codes_json"] = hm_agg["heading_http_map"].apply(lambda x: _json_dumps(x) if isinstance(x, dict) else _json_dumps({}))

    wide = core.merge(title_agg[["cid", "title"]], on="cid", how="left")
    wide = wide.merge(syn_agg[["cid", "synonyms_json", "synonyms_n", "synonyms_preview", "primary_name_from_synonyms"]], on="cid", how="left")
    wide["primary_name"] = wide["title"].where(wide["title"].notna() & (wide["title"].astype(str).str.len() > 0), wide["primary_name_from_synonyms"])
    wide = wide.drop(columns=["title", "primary_name_from_synonyms"], errors="ignore")

    wide = wide.merge(pat_agg[["cid", "patent_ids_json", "patent_ids_n", "patent_ids_preview"]], on="cid", how="left")
    wide = wide.merge(mesh_agg[["cid", "mesh_classes_json", "mesh_classes_n", "mesh_classes_preview"]], on="cid", how="left")
    wide = wide.merge(fda_agg[["cid", "fda_classes_json", "fda_classes_n", "fda_classes_preview"]], on="cid", how="left")
    wide = wide.merge(rawh_agg, on="cid", how="left")
    wide = wide.merge(hm_agg[["cid", "heading_http_codes_json"]], on="cid", how="left")

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_parquet, index=False)
    print("wrote", out_parquet, "rows=", len(wide))

    xlsx_path = Path(args.out_xlsx)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    excel_cols = [
        "cid",
        "primary_name",
        "inchikey",
        "smiles",
        "connectivity_smiles",
        "molecular_formula",
        "molecular_weight",
        "synonyms_n",
        "synonyms_preview",
        "patent_ids_n",
        "patent_ids_preview",
        "mesh_classes_n",
        "mesh_classes_preview",
        "fda_classes_n",
        "fda_classes_preview",
        "heading_http_codes_json",
    ]
    extra_cols = [c for c in wide.columns if c.endswith("_json") and c not in excel_cols]
    excel_cols = [c for c in excel_cols if c in wide.columns] + [c for c in extra_cols if c in wide.columns]

    excel_df = wide[excel_cols].head(args.xlsx_rows).copy()
    for c in excel_df.columns:
        if c.endswith("_json") or c == "heading_http_codes_json":
            excel_df[c] = excel_df[c].apply(_truncate_for_excel)

    with pd.ExcelWriter(xlsx_path) as w:
        excel_df.to_excel(w, sheet_name="pubchem_wide", index=False)

    print("wrote", xlsx_path, "rows=", len(excel_df))


if __name__ == "__main__":
    main()
