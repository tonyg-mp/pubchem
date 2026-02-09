from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_parts(folder: Path) -> pd.DataFrame:
    parts = sorted(folder.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def _try_json_loads(x: Any) -> Any:
    if not isinstance(x, str) or not x:
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _safe_preview_list(vals: list[str], k: int = 20) -> str:
    vals = [v for v in vals if isinstance(v, str)]
    vals = vals[:k]
    return "; ".join(vals)


def _truncate_for_excel(s: Any, limit: int = 30000) -> Any:
    if not isinstance(s, str):
        return s
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory with *.parquet OR outdir root containing part folders.")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--out-xlsx", required=True)
    ap.add_argument("--xlsx-rows", type=int, default=5000)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    print("[stage] loading data files")

    if (in_dir / "core_properties.parquet").exists():
        core = pd.read_parquet(in_dir / "core_properties.parquet")
        heading_meta = pd.read_parquet(in_dir / "heading_meta.parquet") if (in_dir / "heading_meta.parquet").exists() else pd.DataFrame()
        syn = pd.read_parquet(in_dir / "depositor_synonyms.parquet") if (in_dir / "depositor_synonyms.parquet").exists() else pd.DataFrame()
        pat = pd.read_parquet(in_dir / "depositor_patent_ids.parquet") if (in_dir / "depositor_patent_ids.parquet").exists() else pd.DataFrame()
        mesh = pd.read_parquet(in_dir / "mesh_pharm_class.parquet") if (in_dir / "mesh_pharm_class.parquet").exists() else pd.DataFrame()
        fda = pd.read_parquet(in_dir / "fda_pharm_class.parquet") if (in_dir / "fda_pharm_class.parquet").exists() else pd.DataFrame()
        rawh = pd.read_parquet(in_dir / "pugview_raw_headings.parquet") if (in_dir / "pugview_raw_headings.parquet").exists() else pd.DataFrame()
        titles = pd.read_parquet(in_dir / "cid_title.parquet") if (in_dir / "cid_title.parquet").exists() else pd.DataFrame()
        additional = pd.read_parquet(in_dir / "additional_headings.parquet") if (in_dir / "additional_headings.parquet").exists() else pd.DataFrame()
    else:
        print("  - reading core_properties...")
        core = _read_parts(in_dir / "core_properties")
        print("  - reading heading_meta...")
        heading_meta = _read_parts(in_dir / "heading_meta")
        print("  - reading depositor_synonyms...")
        syn = _read_parts(in_dir / "depositor_synonyms")
        print("  - reading depositor_patent_ids...")
        pat = _read_parts(in_dir / "depositor_patent_ids")
        print("  - reading mesh_pharm_class...")
        mesh = _read_parts(in_dir / "mesh_pharm_class")
        print("  - reading fda_pharm_class...")
        fda = _read_parts(in_dir / "fda_pharm_class")
        print("  - reading pugview_raw_headings...")
        rawh = _read_parts(in_dir / "pugview_raw_headings")
        print("  - reading cid_title...")
        titles = _read_parts(in_dir / "cid_title")
        print("  - reading additional_headings...")
        additional = _read_parts(in_dir / "additional_headings")

    if core.empty or "cid" not in core.columns:
        raise SystemExit("core_properties missing or empty")

    print(f"[stage] processing {len(core):,} compounds")

    core = core.drop_duplicates(subset=["cid"]).copy()

    print("[stage] processing titles")
    title_agg = pd.DataFrame({"cid": core["cid"]})
    if not titles.empty and "cid" in titles.columns:
        if "title" not in titles.columns:
            titles["title"] = None
        titles = titles.dropna(subset=["cid", "title"]).copy()
        titles["cid"] = titles["cid"].astype(int)
        titles["title"] = titles["title"].astype(str)
        titles = titles.drop_duplicates(subset=["cid"])
        title_agg = title_agg.merge(titles[["cid", "title"]], on="cid", how="left")
        print(f"  ✓ {len(titles):,} titles merged")
    else:
        title_agg["title"] = None

    syn_agg = pd.DataFrame({"cid": core["cid"]})
    if not syn.empty:
        print(f"[stage] aggregating {len(syn):,} synonym records")
        syn = syn.dropna(subset=["cid", "synonym"]).copy()
        syn["synonym"] = syn["synonym"].astype(str)
        syn = syn.drop_duplicates(subset=["cid", "synonym"])
        print("  - grouping synonyms by CID...")
        g = syn.groupby("cid")["synonym"].apply(list).reset_index(name="synonyms_list")
        print(f"  - creating synonym aggregations for {len(g):,} compounds...")
        syn_agg = syn_agg.merge(g, on="cid", how="left")
        print(f"  ✓ synonyms aggregated")
    else:
        syn_agg["synonyms_list"] = None

    def first_syn(x: Any) -> Any:
        if isinstance(x, list) and x:
            return x[0]
        return None

    syn_agg["synonyms_json"] = syn_agg["synonyms_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    syn_agg["synonyms_n"] = syn_agg["synonyms_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    syn_agg["synonyms_preview"] = syn_agg["synonyms_list"].apply(lambda x: _safe_preview_list(x, 25) if isinstance(x, list) else "")
    syn_agg["primary_name_from_synonyms"] = syn_agg["synonyms_list"].apply(first_syn)

    pat_agg = pd.DataFrame({"cid": core["cid"]})
    if not pat.empty:
        print(f"[stage] aggregating {len(pat):,} patent records")
        pat = pat.dropna(subset=["cid", "patent_id"]).copy()
        pat["patent_id"] = pat["patent_id"].astype(str)
        pat = pat.drop_duplicates(subset=["cid", "patent_id"])
        print("  - grouping patents by CID (this may take several minutes)...")
        g = pat.groupby("cid")["patent_id"].apply(list).reset_index(name="patent_ids_list")
        print(f"  - creating patent aggregations for {len(g):,} compounds...")
        pat_agg = pat_agg.merge(g, on="cid", how="left")
        print(f"  ✓ patents aggregated")
    else:
        pat_agg["patent_ids_list"] = None

    pat_agg["patent_ids_json"] = pat_agg["patent_ids_list"].apply(lambda x: _json_dumps(x) if isinstance(x, list) else _json_dumps([]))
    pat_agg["patent_ids_n"] = pat_agg["patent_ids_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    pat_agg["patent_ids_preview"] = pat_agg["patent_ids_list"].apply(lambda x: _safe_preview_list(x, 25) if isinstance(x, list) else "")

    mesh_agg = pd.DataFrame({"cid": core["cid"]})
    if not mesh.empty:
        print(f"[stage] aggregating {len(mesh):,} MeSH records")
        mesh = mesh.dropna(subset=["cid"]).copy()
        cols = ["mesh_name", "mesh_description", "reference_number", "raw_info_json"]
        for c in cols:
            if c not in mesh.columns:
                mesh[c] = None

        def rows_to_objs(df: pd.DataFrame) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for _, r in df.iterrows():
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

        print("  - grouping MeSH classifications by CID...")
        g = mesh.groupby("cid").apply(rows_to_objs).reset_index(name="mesh_classes_list")
        mesh_agg = mesh_agg.merge(g, on="cid", how="left")
        print(f"  ✓ MeSH classes aggregated")
    else:
        mesh_agg["mesh_classes_list"] = None

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

    # Process additional headings (ClinicalTrials.gov and IUPHAR)
    add_agg = pd.DataFrame({"cid": core["cid"]})
    if not additional.empty and "cid" in additional.columns:
        print("[stage] processing additional headings")
        additional = additional.drop_duplicates(subset=["cid"]).copy()
        add_agg = add_agg.merge(additional, on="cid", how="left")
        non_null_counts = {col: additional[col].notna().sum() for col in additional.columns if col != "cid"}
        print(f"  ✓ loaded {len(additional):,} compounds with additional data: {non_null_counts}")

    print("[stage] merging all data into wide format")
    print(f"  - starting with {len(core):,} core compounds")

    wide = core.merge(title_agg[["cid", "title"]], on="cid", how="left")
    print(f"  ✓ titles merged ({len(wide):,} rows)")
    wide = wide.merge(syn_agg[["cid", "synonyms_json", "synonyms_n", "synonyms_preview", "primary_name_from_synonyms"]], on="cid", how="left")
    print(f"  ✓ synonyms merged ({len(wide):,} rows)")
    wide["primary_name"] = wide["title"].where(wide["title"].notna() & (wide["title"].astype(str).str.len() > 0), wide["primary_name_from_synonyms"])
    wide = wide.drop(columns=["title", "primary_name_from_synonyms"], errors="ignore")

    wide = wide.merge(pat_agg[["cid", "patent_ids_json", "patent_ids_n", "patent_ids_preview"]], on="cid", how="left")
    print(f"  ✓ patents merged ({len(wide):,} rows)")
    wide = wide.merge(mesh_agg[["cid", "mesh_classes_json", "mesh_classes_n", "mesh_classes_preview"]], on="cid", how="left")
    print(f"  ✓ mesh classes merged ({len(wide):,} rows)")
    wide = wide.merge(fda_agg[["cid", "fda_classes_json", "fda_classes_n", "fda_classes_preview"]], on="cid", how="left")
    print(f"  ✓ fda classes merged ({len(wide):,} rows)")
    wide = wide.merge(rawh_agg, on="cid", how="left")
    print(f"  ✓ raw headings merged ({len(wide):,} rows)")
    wide = wide.merge(hm_agg[["cid", "heading_http_codes_json"]], on="cid", how="left")
    print(f"  ✓ heading http codes merged ({len(wide):,} rows)")
    
    # Merge additional headings if available
    if not add_agg.empty and len(add_agg.columns) > 1:
        add_cols = [c for c in add_agg.columns if c != "cid"]
        wide = wide.merge(add_agg[["cid"] + add_cols], on="cid", how="left")
        print(f"  ✓ additional headings merged ({len(wide):,} rows)")

    print("[stage] writing parquet file")

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_parquet, index=False)
    print("wrote", out_parquet, "rows=", len(wide))

    print("[stage] writing excel file")

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
        "clinicaltrials_count",
        "iuphar_hid",
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
