"""TF-IDF + SVM lineal: ajuste de hiperparámetros en validación.

Protocolo
---------
* Todo se ajusta con train; se elige con el **macro-F1 en val**; test no se mira para decidir.
* Búsqueda en dos fases, para que sea asumible en un portátil:
    1. Vectorizador (min_df, max_df, n-gramas) con un clasificador fijo (C=0.3, sin pesos de clase).
    2. Con el mejor vectorizador: C y class_weight (None / balanced).
* Clasificador: SVM lineal (LinearSVC, uno-contra-resto). Se probó antes regresión logística:
  con lbfgs tarda ~90 s por ajuste y con saga + class_weight='balanced' no converge en minutos;
  con pesos uniformes daba el mismo macro-F1 en val (0,36) que la SVM, que tarda ~5 s.
* Cada configuración se guarda en results/tfidf_logreg_search.csv nada más terminar; si el
  script se interrumpe, al relanzarlo se salta lo ya hecho.
* Al final se reentrena la mejor configuración con train y se guardan sus predicciones en
  val y test (results/preds/) para el módulo de evaluación común. Las métricas de test
  NO se imprimen aquí: se analizan una sola vez, junto al resto de modelos.

Uso:
    python scripts/tfidf_svm.py [--max-seconds 150] [--level 2]
"""
import argparse
import warnings
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config  # noqa: E402
from cordis.data import load_dataset  # noqa: E402
from cordis.features import make_tfidf  # noqa: E402
from cordis.preprocess import load_tokens  # noqa: E402

VEC_GRID = [dict(min_df=m, max_df=x, ngram_range=n)
            for n in [[1, 1], [1, 2]] for m in [2, 5] for x in [0.3, 0.5, 1.0]]
CLF_FIXED = dict(C=0.3, class_weight=None)
CLF_GRID = [dict(C=c, class_weight=w) for w in [None, "balanced"] for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]]


def key(vec, clf):
    return json.dumps({"vec": vec, "clf": clf}, sort_keys=True, default=list)


def fit_eval(vec_p, clf_p, Xtr, ytr, Xva, yva):
    t = time.time()
    vec = make_tfidf(**vec_p)
    A = vec.fit_transform(Xtr)
    clf = LinearSVC(max_iter=5000, random_state=config.SEED, **clf_p)
    clf.fit(A, ytr)
    yp = clf.predict(vec.transform(Xva))
    return vec, clf, {
        "vocab": len(vec.vocabulary_), "val_macro_f1": f1_score(yva, yp, average="macro"),
        "val_accuracy": accuracy_score(yva, yp), "seconds": round(time.time() - t, 1)}


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()

    df = load_dataset().merge(load_tokens().rename("tokens"), left_on="projectID", right_index=True)
    y = f"label_l{args.level}"
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr)} · val {len(va)} · test {len(te)} · nivel {args.level}")

    log_path = config.RESULTS_DIR / f"tfidf_svm_search_l{args.level}.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["key"])
    done = set(log["key"])

    def run(phase, vec_p, clf_p):
        nonlocal log
        k = key(vec_p, clf_p)
        if k in done:
            return True
        if time.time() - t0 > args.max_seconds:
            return False
        _, _, r = fit_eval(vec_p, clf_p, tr["tokens"], tr[y], va["tokens"], va[y])
        row = {"key": k, "phase": phase, **{f"vec_{a}": str(b) for a, b in vec_p.items()},
               **{f"clf_{a}": str(b) for a, b in clf_p.items()}, **r}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(log_path, index=False)
        done.add(k)
        print(f"[{phase}] {vec_p} {clf_p} -> macro-F1 val {r['val_macro_f1']:.4f} "
              f"acc {r['val_accuracy']:.4f} ({r['seconds']} s)", flush=True)
        return True

    # Fase 1: vectorizador
    for v in VEC_GRID:
        if not run("1-vectorizer", v, CLF_FIXED):
            return print("Tiempo agotado; relanza para continuar.")
    p1 = log[log["phase"] == "1-vectorizer"]
    best_vec = json.loads(p1.loc[p1["val_macro_f1"].astype(float).idxmax(), "key"])["vec"]

    # Fase 2: clasificador
    for c in CLF_GRID:
        if not run("2-classifier", best_vec, c):
            return print("Tiempo agotado; relanza para continuar.")
    p2 = log[log["key"].map(lambda k: json.loads(k)["vec"] == best_vec)]
    best = json.loads(p2.loc[p2["val_macro_f1"].astype(float).idxmax(), "key"])
    print("Mejor configuración (según val):", best)

    # Reentrenar la mejor y guardar predicciones para la evaluación común
    vec, clf, r = fit_eval(best["vec"], best["clf"], tr["tokens"], tr[y], va["tokens"], va[y])
    pred_dir = config.RESULTS_DIR / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    name = f"tfidf_svm_l{args.level}"
    for s, d in (("val", va), ("test", te)):
        P = clf.decision_function(vec.transform(d["tokens"]))   # puntuaciones, no probabilidades
        pd.DataFrame({"projectID": d["projectID"].values, "y_true": d[y].values,
                      "y_pred": clf.classes_[P.argmax(1)]}).to_csv(pred_dir / f"{name}_{s}.csv", index=False)
        np.save(pred_dir / f"{name}_{s}_scores.npy", P.astype("float32"))
    json.dump({"model": name, "best": best, "val": r, "classes": clf.classes_.tolist()},
              open(config.RESULTS_DIR / f"{name}_best.json", "w"), indent=2, default=list)
    print(f"Guardado {name}: macro-F1 val {r['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
