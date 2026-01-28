from __future__ import annotations

import argparse
import json
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


def load_mesh_pharm_map(path: Path) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            term = parts[0] if parts else ""
            classes = [c for c in parts[1:] if c]
            if term and classes:
                m[term] = classes
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-list", required=True)
    ap.add_argument("--extras-dir", required=True)
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    wanted = read_cids(Path(args.cid_list))
    extras = Path(args.extras_dir)
    cid_mesh_path = extras / "CID-MeSH"
    mesh_pharm_path = extras / "MeSH-Pharm"

    print("[stage] inputs")
    print("wanted_cids:", len(wanted))
    print("cid_mesh_path:", str(cid_mesh_path))
    print("mesh_pharm_path:", str(mesh_pharm_path))

    print("[stage] load MeSH-Pharm")
    mesh_to_classes = load_mesh_pharm_map(mesh_pharm_path)
    print("mesh_terms_with_classes:", len(mesh_to_classes))

    print("[stage] stream CID-MeSH")
    rows = []
    seen = set()

    with cid_mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            cid_str = parts[0].strip()
            if not cid_str.isdigit():
                continue
            cid = int(cid_str)
            if cid not in wanted:
                continue

            mesh_terms = [p.strip() for p in parts[1:] if p.strip()]
            for mesh_term in mesh_terms:
                classes = mesh_to_classes.get(mesh_term)
                if not classes:
                    continue
                for cls in classes:
                    key = (cid, cls, mesh_term)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "cid": cid,
                            "mesh_name": cls,
                            "mesh_description": None,
                            "reference_number": None,
                            "raw_info_json": json.dumps(
                                {"mesh_term": mesh_term, "pharm_class": cls},
                                ensure_ascii=False,
                            ),
                        }
                    )

    df = pd.DataFrame(rows)
    outp = Path(args.out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    print("[done] wrote", str(outp))
    print("rows:", len(df))
    print("unique_cids:", df["cid"].nunique() if not df.empty else 0)


if __name__ == "__main__":
    main()
