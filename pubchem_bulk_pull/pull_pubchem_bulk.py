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
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _read_id_list(path: Path, max_ids: int | None) -> list[int]:
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


def _fetch_json(url: str, cache_path: Path | None, timeout: int = 30) -> dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        r = requests.get(url, timeout=timeout)
        try:
            j = r.json()
        except Exception:
            j = {"error": {"message": f"HTTP {r.status_code} (non-json)"}}

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")

        return j
    except Exception as e:
        j = {"error": {"message": str(e)}}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
        return j


def _pugview_url(cid: int, heading: str) -> str:
    h = requests.utils.quote(heading, safe="")
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={h}"


def _get_pugrest_properties_batch(cids: list[int]) -> dict[int, dict[str, Any]]:
    cid_list = ",".join(str(c) for c in cids)
    props = "InChIKey,SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}/property/{props}/JSON"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return {}
    j = r.json()
    rows = j.get("PropertyTable", {}).get("Properties", []) or []
    out: dict[int, dict[str, Any]] = {}
    for rec in rows:
        cid = rec.get("CID")
        if cid is None:
            continue
        out[int(cid)] = rec
    return out


def _extract_strings_with_markup(j: dict[str, Any]) -> list[str]:
    out: list[str] = []

    def walk(x: Any):
        if isinstance(x, dict):
            if "StringWithMarkup" in x and isinstance(x["StringWithMarkup"], list):
                for itm in x["StringWithMarkup"]:
                    if isinstance(itm, dict) and isinstance(itm.get("String"), str):
                        out.append(itm["String"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(j)
    return out


def _extract_mesh_pharm_rows(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sections = (j.get("Record") or {}).get("Section") or []
    for sec in sections:
        for subsec in (sec.get("Section") or []):
            infos = subsec.get("Information") or []
            for info in infos:
                name = info.get("Name")
                ref = info.get("ReferenceNumber")
                desc = None
                val = info.get("Value") or {}
                swm = val.get("StringWithMarkup")
                if isinstance(swm, list) and swm:
                    s0 = swm[0]
                    if isinstance(s0, dict):
                        desc = s0.get("String")
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
    rows: list[dict[str, Any]] = []
    strings = _extract_strings_with_markup(j)

    for s in strings:
        if not isinstance(s, str):
            continue
        parts = [p.strip() for p in s.split(";") if p.strip()]
        for p in parts:
            m2 = _FDA_GROUP_RE.match(p)
            if m2:
                rows.append(
                    {
                        "cid": cid,
                        "class_type": m2.group("typ"),
                        "class_group": m2.group("group"),
                        "class_name": m2.group("name"),
                        "raw_text": p,
                    }
                )
                continue

            m1 = _FDA_SUFFIX_RE.match(p)
            if m1:
                rows.append(
                    {
                        "cid": cid,
                        "class_type": m1.group("typ"),
                        "class_group": None,
                        "class_name": m1.group("name"),
                        "raw_text": p,
                    }
                )
                continue
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
    strings = _extract_strings_with_markup(j)
    ids = set()
    for s in strings:
        if isinstance(s, str) and re.fullmatch(r"US\d+", s.strip()):
            ids.add(s.strip())
    return [{"cid": cid, "patent_id": pid} for pid in sorted(ids)]


def _extract_depositor_synonyms(cid: int, j: dict[str, Any]) -> list[dict[str, Any]]:
    strings = _extract_strings_with_markup(j)
    syns = []
    for s in strings:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t:
            continue
        if "Link to all" in t or "link to all" in t:
            continue
        syns.append(t)
    return [{"cid": cid, "synonym": s} for s in syns]


def _write_parts(base: Path, folder: str, part_idx: int, df: pd.DataFrame) -> None:
    if df.empty:
        return
    outdir = base / folder
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"part-{part_idx:05d}.parquet"
    df.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-cids", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=250)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--cache-dir", default="pubchem_bulk_pull/cache_json")
    ap.add_argument("--properties-batch-size", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
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
    heading_meta_rows: list[dict[str, Any]] = []
    depositor_syn_rows: list[dict[str, Any]] = []
    depositor_pat_rows: list[dict[str, Any]] = []
    mesh_rows: list[dict[str, Any]] = []
    fda_rows: list[dict[str, Any]] = []
    raw_heading_rows: list[dict[str, Any]] = []
    processed_now: list[int] = []

    def flush():
        nonlocal part_idx
        part_idx += 1
        _write_parts(outdir, "core_properties", part_idx, core_df if part_idx == 1 else pd.DataFrame())
        _write_parts(outdir, "heading_meta", part_idx, pd.DataFrame(heading_meta_rows))
        _write_parts(outdir, "depositor_synonyms", part_idx, pd.DataFrame(depositor_syn_rows))
        _write_parts(outdir, "depositor_patent_ids", part_idx, pd.DataFrame(depositor_pat_rows))
        _write_parts(outdir, "mesh_pharm_class", part_idx, pd.DataFrame(mesh_rows))
        _write_parts(outdir, "fda_pharm_class", part_idx, pd.DataFrame(fda_rows))
        _write_parts(outdir, "pugview_raw_headings", part_idx, pd.DataFrame(raw_heading_rows))

        if processed_now:
            with processed_file.open("a", encoding="utf-8") as f:
                for cid in processed_now:
                    f.write(f"{cid}\n")

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
