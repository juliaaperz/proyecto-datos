"""Figuras de la búsqueda de hiperparámetros TF-IDF + SVM (lee results/tfidf_svm_search_l2.csv)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config  # noqa: E402

plt.rcParams.update({
    "savefig.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b0b0a8", "axes.grid": True, "grid.color": "#e6e5df", "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
BLUE, ORANGE = "#2a78d6", "#eb6834"

log = pd.read_csv(config.RESULTS_DIR / "tfidf_svm_search_l2.csv")
log["k"] = log["key"].map(json.loads)

# --- Validación de C (fase 2) ---
best_vec = json.load(open(config.RESULTS_DIR / "tfidf_svm_l2_best.json"))["best"]["vec"]
p2 = log[log["k"].map(lambda k: k["vec"] == best_vec)].copy()   # incluye el punto de la fase 1
p2["C"] = p2["k"].map(lambda k: k["clf"]["C"])
p2["w"] = p2["k"].map(lambda k: k["clf"]["class_weight"] or "ninguno")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
for ax, metric, title in ((axes[0], "val_macro_f1", "Macro-F1 (val)"), (axes[1], "val_accuracy", "Accuracy (val)")):
    for (w, g), c in zip(p2.groupby("w"), (BLUE, ORANGE)):
        g = g.sort_values("C")
        ax.plot(g["C"], g[metric], marker="o", ms=6, lw=2, color=c, label=f"pesos de clase: {w}")
    ax.set_xscale("log"); ax.set_xlabel("C (regularización inversa)"); ax.set_title(title, loc="left")
best = p2.loc[p2["val_macro_f1"].idxmax()]
axes[0].annotate(f"elegido: C={best['C']}, {best['w']}\nmacro-F1={best['val_macro_f1']:.3f}",
                 (best["C"], best["val_macro_f1"]), xytext=(12, -38), textcoords="offset points",
                 fontsize=8, arrowprops=dict(arrowstyle="-", color="#5c5c56", lw=0.8))
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("TF-IDF + SVM lineal: curva de validación de C (vectorizador fijo)", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "tfidf_svm_validation_C.png")

# --- Tabla de la fase 1 ---
p1 = log[log["phase"] == "1-vectorizer"].copy()
tab = pd.DataFrame({
    "n-gramas": p1["k"].map(lambda k: "1" if k["vec"]["ngram_range"] == [1, 1] else "1–2"),
    "min_df": p1["k"].map(lambda k: k["vec"]["min_df"]),
    "max_df": p1["k"].map(lambda k: k["vec"]["max_df"]),
    "vocabulario": p1["vocab"].astype(int),
    "macro-F1 val": p1["val_macro_f1"].round(3),
    "accuracy val": p1["val_accuracy"].round(3),
}).sort_values("macro-F1 val", ascending=False)
tab.to_csv(config.RESULTS_DIR / "tfidf_svm_vectorizer_table.csv", index=False)
print(tab.to_markdown(index=False))
