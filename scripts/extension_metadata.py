"""Compara metadatos, texto y texto + metadatos con la partición común."""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config
from cordis.data import load_dataset
from cordis.features import make_tfidf
from cordis.preprocess import load_tokens


def metadata_frame(df):
    out = df[["totalCost", "ecMaxContribution"]].apply(pd.to_numeric, errors="coerce")
    out["duration_months"] = (pd.to_datetime(df["endDate"]) - pd.to_datetime(df["startDate"])).dt.days / 30.4375
    out.loc[out["duration_months"] < 0, "duration_months"] = np.nan
    out["n_countries"] = df["countryContr"].map(
        lambda s: len(set(re.findall(r"(?:^|\s)([A-Z]{2})\|", str(s)))) if pd.notna(s) else np.nan)
    return out


def encode_metadata(tr, va, te, min_country_count=100):
    numeric = [metadata_frame(d) for d in (tr, va, te)]
    imp = SimpleImputer(strategy="median", add_indicator=True)
    for frame in numeric:
        for col in ("totalCost", "ecMaxContribution"):
            frame[col] = np.log1p(frame[col].where(frame[col] >= 0))
    arrays = [imp.fit_transform(numeric[0])]
    arrays += [imp.transform(x) for x in numeric[1:]]
    scaler = StandardScaler()
    arrays = [scaler.fit_transform(arrays[0])] + [scaler.transform(a) for a in arrays[1:]]
    counts = tr["coordinatorCountry"].value_counts()
    common = set(counts[counts >= min_country_count].index)
    countries = [d["coordinatorCountry"].fillna("Otros").map(lambda c: c if c in common else "Otros").to_numpy().reshape(-1, 1) for d in (tr, va, te)]
    categories = sorted(common | {"Otros"})
    enc = OneHotEncoder(categories=[categories], handle_unknown="ignore", dtype=np.float32)
    cats = [enc.fit_transform(countries[0])] + [enc.transform(c) for c in countries[1:]]
    matrices = [hstack([csr_matrix(a.astype(np.float32)), c], format="csr") for a, c in zip(arrays, cats)]
    return matrices, sorted(common)


def metrics(y, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()
    level = args.level
    df = load_dataset().merge(load_tokens().rename("tokens"), left_on="projectID", right_index=True, validate="one_to_one")
    tr, va, te = [df[df["split"] == s].copy() for s in ("train", "val", "test")]
    ycol = f"label_l{level}"
    ytr, yva = tr[ycol].to_numpy(), va[ycol].to_numpy()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pred_dir = config.RESULTS_DIR / "preds"
    pred_dir.mkdir(exist_ok=True)
    meta, countries = encode_metadata(tr, va, te)
    best_path = config.RESULTS_DIR / f"tfidf_svm_l{level}_best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Ejecuta primero scripts/tfidf_svm.py --level {level}")
    text_config = json.loads(best_path.read_text())["best"]
    vec = make_tfidf(**text_config["vec"])
    text = [vec.fit_transform(tr["tokens"]), vec.transform(va["tokens"]), vec.transform(te["tokens"])]
    meta_scale = float(np.sqrt(meta[0].multiply(meta[0]).sum() / len(tr)))
    combined = [hstack([t, m / meta_scale], format="csr") for t, m in zip(text, meta)]
    rows, summary, predictions = [], {}, {}

    def select(name, matrices, candidates):
        winner, best, best_score = None, None, -1
        for params in candidates:
            if name == "meta":
                model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2, **params)
                a, b = matrices[0].toarray(), matrices[1].toarray()
            else:
                model = LinearSVC(max_iter=5000, random_state=42, dual="auto", **params)
                a, b = matrices[0], matrices[1]
            model.fit(a, ytr)
            pred = model.predict(b)
            m = metrics(yva, pred)
            rows.append({"model": name, "params": json.dumps(params, sort_keys=True), **m})
            pd.DataFrame(rows).to_csv(config.RESULTS_DIR / f"extension_search_l{level}.csv", index=False)
            print(name, params, m, flush=True)
            if m["macro_f1"] > best_score:
                winner, best, best_score = model, params, m["macro_f1"]
        for split, d, a in zip(("val", "test"), (va, te), matrices[1:]):
            pred = winner.predict(a.toarray() if name == "meta" else a)
            pd.DataFrame({"projectID": d["projectID"].to_numpy(), "y_true": d[ycol].to_numpy(), "y_pred": pred}).to_csv(pred_dir / f"{name}_l{level}_{split}.csv", index=False)
            if split == "val":
                predictions[name] = pred
                score = metrics(yva, pred)
        summary[name] = {"best": best, "val": score}

    select("meta", meta, [{"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
        for depth in (12, None) for leaf in (2, 10) for weight in (None, "balanced")])
    select("text", text, [text_config["clf"]])
    select("text_meta", combined, [{"C": c, "class_weight": weight}
        for c in (0.03, 0.1, 0.3, 1.0) for weight in (None, "balanced")])
    majority = pd.Series(ytr).value_counts().index[0]
    summary["majority"] = {"val": metrics(yva, np.full(len(yva), majority))}
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / f"extension_search_l{level}.csv", index=False)
    table = pd.DataFrame([{ "model": k, **v["val"]} for k, v in summary.items()])
    table.to_csv(config.RESULTS_DIR / f"extension_comparison_l{level}.csv", index=False)
    classes = np.unique(ytr)
    names = tr.drop_duplicates(ycol).set_index(ycol)[f"label_l{level}_name"]
    per_class = pd.DataFrame({"class": classes, "name": [names[c] for c in classes]})
    for name, pred in predictions.items():
        _, _, f, support = precision_recall_fscore_support(yva, pred, labels=classes, zero_division=0)
        per_class[f"f1_{name}"] = f
        per_class["support_val"] = support
    per_class["delta_f1_text_meta_vs_text"] = per_class["f1_text_meta"] - per_class["f1_text"]
    per_class.to_csv(config.RESULTS_DIR / f"extension_per_class_l{level}.csv", index=False)
    output = {"level": level, "seed": 42, "selection": "macro_f1_val", "train": len(tr), "val": len(va), "test": len(te),
              "metadata": ["totalCost", "ecMaxContribution", "duration_months", "coordinatorCountry", "n_countries"],
              "combined_metadata_scale_rms_train": meta_scale, "country_threshold_train": 100, "common_countries": countries, "text_config": text_config, "models": summary}
    (config.RESULTS_DIR / f"extension_l{level}_best.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    config.FIGURES_DIR.mkdir(exist_ok=True)
    costs = metadata_frame(tr)["totalCost"]
    groups = sorted(tr["label_l1_name"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([np.log1p(costs[tr["label_l1_name"] == g].clip(lower=0).dropna()) for g in groups], orientation="horizontal")
    ax.set_yticks(range(1, len(groups) + 1), labels=groups)
    ax.set_xlabel("log(1 + coste total en euros)")
    ax.set_title("Coste por área de nivel 1 (solo entrenamiento)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "extension_cost_by_l1.png", dpi=150)
    plt.close(fig)
    print(table.to_string(index=False), flush=True)
    print("Guardadas predicciones de test; no se han calculado métricas de test.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
