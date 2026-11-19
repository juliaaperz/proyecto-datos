"""Repite en nivel 1 la configuración de TF-IDF + SVM elegida en nivel 2."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config
from cordis.data import load_dataset
from cordis.features import make_tfidf
from cordis.preprocess import load_tokens


def main():
    original = json.loads((config.RESULTS_DIR / "tfidf_svm_l2_best.json").read_text())
    best = original["best"]
    df = load_dataset().merge(load_tokens().rename("tokens"), left_on="projectID", right_index=True, validate="one_to_one")
    tr, va, te = [df[df["split"] == s] for s in ("train", "val", "test")]
    vec = make_tfidf(**best["vec"])
    train = vec.fit_transform(tr["tokens"])
    model = LinearSVC(random_state=42, max_iter=5000, **best["clf"])
    model.fit(train, tr["label_l1"])
    score = None
    pred_dir = config.RESULTS_DIR / "preds"
    pred_dir.mkdir(exist_ok=True)
    for split, d in (("val", va), ("test", te)):
        pred = model.predict(vec.transform(d["tokens"]))
        pd.DataFrame({"projectID": d["projectID"].to_numpy(), "y_true": d["label_l1"].to_numpy(), "y_pred": pred}).to_csv(pred_dir / f"tfidf_svm_l1_fixed_l2_{split}.csv", index=False)
        if split == "val":
            score = {"val_accuracy": float(accuracy_score(d["label_l1"], pred)), "val_macro_f1": float(f1_score(d["label_l1"], pred, average="macro"))}
    result = {"best": best, "level": 1, "configuration_selected_at_level": 2, **score}
    (config.RESULTS_DIR / "tfidf_svm_l1_fixed_l2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
