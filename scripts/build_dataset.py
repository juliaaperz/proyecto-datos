"""Semana 1: construye el dataset etiquetado y la partición fija.

Uso (desde la raíz del repo):
    python scripts/build_dataset.py                 # crea dataset + split (si no existe)
    python scripts/build_dataset.py --rule first_valid
    python scripts/build_dataset.py --force-split   # regenera splits/split.csv (¡avisad al equipo!)

Genera:
    data/processed/dataset.parquet   (NO se sube a git)
    splits/split.csv                 (SÍ se sube: partición común)
    splits/labels.csv                (códigos, nombres y tamaño de cada clase por partición)
    results/week1_checks.json        (comprobaciones frente al enunciado + baseline)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config  # noqa: E402
from cordis.data import (DEFAULT_RULE, build_labels, code_names, load_projects,  # noqa: E402
                         load_taxonomy, project_label)
from cordis.split import make_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default=DEFAULT_RULE, choices=["first", "first_valid"])
    ap.add_argument("--force-split", action="store_true")
    args = ap.parse_args()

    print("Leyendo datos (projects.xlsx tarda ~1 min)...")
    tax = load_taxonomy()
    raw = load_projects()
    n_raw = len(raw)
    df = build_labels(raw, tax, rule=args.rule)
    code2path = dict(zip(tax["code"], tax["path"]))
    names = code_names(tax)

    # ---------------- Comprobaciones frente al enunciado ----------------
    checks = {
        "rule": args.rule,
        "n_raw": n_raw,
        "n_without_codes": int((raw["codes"].str.len() == 0).sum()),
        "n_kept": len(df),
        "pct_kept": round(100 * len(df) / n_raw, 3),
        "pct_kept_expected": 90.31,
        "n_classes_l2": int(df["label_l2"].nunique()),
        "n_classes_l1": int(df["label_l1"].nunique()),
        "pct_unambiguous_l2": round(100 * (df["n_l2"] == 1).mean(), 1),                # ~32 %
        "pct_unambiguous_l1": round(100 * (df["all_l1"].str.len() == 1).mean(), 1),    # ~47 %
    }
    for lvl in (1, 2):
        maj = df["codes"].apply(lambda c: project_label(c, code2path, lvl, "majority"))
        checks[f"pct_change_majority_rule_l{lvl}"] = round(
            100 * (maj != df[f"label_l{lvl}"]).mean(), 1)                              # ~12 %

    # ---------------- Partición ----------------
    config.SPLITS_DIR.mkdir(exist_ok=True)
    if config.SPLIT_PATH.exists() and not args.force_split:
        split = pd.read_csv(config.SPLIT_PATH)
        print(f"Usando partición existente {config.SPLIT_PATH} (usa --force-split para regenerarla)")
        if set(split["projectID"]) != set(df["projectID"]):
            raise SystemExit("La partición guardada no coincide con el dataset (¿otra --rule?). "
                             "Revisad antes de usar --force-split.")
    else:
        split = make_split(df)
        split.to_csv(config.SPLIT_PATH, index=False)
        print(f"Partición nueva guardada en {config.SPLIT_PATH}")

    d = df.merge(split, on="projectID")
    checks["split_sizes"] = d["split"].value_counts().reindex(["train", "val", "test"]).to_dict()
    checks["min_class_count_per_split"] = (
        d.groupby(["split", "label_l2"]).size().groupby("split").min().to_dict())
    # Fuga: ningún resumen repetido puede estar en dos particiones
    checks["summaries_in_several_splits"] = int(
        (d.groupby("summary")["split"].nunique() > 1).sum())

    # ---------------- Baseline de clase mayoritaria ----------------
    for lvl in (2, 1):
        y = f"label_l{lvl}"
        maj = d.loc[d["split"] == "train", y].mode()[0]
        for s in ("val", "test"):
            yt = d.loc[d["split"] == s, y]
            yp = np.full(len(yt), maj)
            checks[f"baseline_l{lvl}_{s}"] = {
                "majority_class": f"{maj} - {names[maj]}",
                "accuracy": round(accuracy_score(yt, yp), 4),
                "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
            }

    # ---------------- Guardar ----------------
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATASET_PATH, index=False)

    labels = (d.groupby(["label_l1", "label_l1_name", "label_l2", "label_l2_name", "split"])
                .size().unstack("split", fill_value=0)[["train", "val", "test"]]
                .assign(total=lambda t: t.sum(axis=1)).sort_values("total", ascending=False)
                .reset_index())
    labels.to_csv(config.SPLITS_DIR / "labels.csv", index=False)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "week1_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False, default=int)

    print(json.dumps(checks, indent=2, ensure_ascii=False, default=int))
    if args.rule == "first":
        ok = abs(checks["pct_kept"] - 90.31) < 0.05
        print("\nPunto de control 90,31 %:", "OK" if ok else "NO COINCIDE")


if __name__ == "__main__":
    main()
