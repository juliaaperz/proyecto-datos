"""Compara nivel 1 y 2 en validación con configuraciones elegidas en nivel 2."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cordis import config
from cordis.data import load_dataset


def validate_predictions(predictions, expected, label, allowed_classes=None):
    required = {"projectID", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError("Faltan columnas projectID, y_true o y_pred.")
    if predictions["projectID"].duplicated().any():
        raise ValueError("Hay projectID duplicados en las predicciones.")
    if set(predictions["projectID"]) != set(expected["projectID"]):
        raise ValueError("Los proyectos no coinciden exactamente con validación.")
    aligned = expected[["projectID", label]].merge(predictions, on="projectID", validate="one_to_one")
    if not np.array_equal(aligned[label].to_numpy(), aligned["y_true"].to_numpy()):
        raise ValueError("Las etiquetas reales no coinciden con el dataset y el nivel indicado.")
    allowed = set(expected[label]) if allowed_classes is None else set(allowed_classes)
    if aligned["y_pred"].isna().any() or not aligned["y_pred"].isin(allowed).all():
        raise ValueError("Hay predicciones ausentes o clases desconocidas.")
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contextual-l1-val", type=Path)
    parser.add_argument("--contextual-l2-val", type=Path)
    args = parser.parse_args()
    if bool(args.contextual_l1_val) != bool(args.contextual_l2_val):
        parser.error("Indica ambos CSV contextuales, de nivel 1 y 2, o ninguno.")
    dataset = load_dataset()
    val = dataset[dataset["split"] == "val"]
    pred_dir = config.RESULTS_DIR / "preds"
    files = {
        "TF-IDF": {1: pred_dir / "tfidf_svm_l1_fixed_l2_val.csv", 2: pred_dir / "tfidf_svm_l2_val.csv"},
        "Word2Vec": {1: pred_dir / "w2v_svm_l1_fixed_l2_val.csv", 2: pred_dir / "w2v_svm_l2_val.csv"},
    }
    pending = []
    if args.contextual_l1_val:
        files["Contextuales"] = {1: args.contextual_l1_val, 2: args.contextual_l2_val}
    else:
        pending.append("Embeddings contextuales: faltan los modelos y sus predicciones de validación en ambos niveles.")
    rows = []
    for representation, levels in files.items():
        for level, path in levels.items():
            if not path.exists():
                raise FileNotFoundError(f"Faltan predicciones: {path}")
            label = f"label_l{level}"
            a = validate_predictions(pd.read_csv(path), val, label, dataset[label].unique())
            rows.append({"representation": representation, "level": level, "split": "val", "n": len(a),
                         "accuracy": float(accuracy_score(a[label], a["y_pred"])),
                         "macro_f1": float(f1_score(a[label], a["y_pred"], labels=np.sort(dataset[label].unique()), average="macro", zero_division=0)),
                         "source": str(path.relative_to(config.REPO_ROOT)) if path.is_relative_to(config.REPO_ROOT) else str(path)})
    table = pd.DataFrame(rows)
    table["rank_among_available"] = table.groupby("level")["macro_f1"].rank(method="min", ascending=False).astype(int)
    table.to_csv(config.RESULTS_DIR / "representations_levels_val.csv", index=False)
    report = {"protocol": "validation_only; level1 uses configuration selected at level2", "pending": pending, "complete_three_representations": not pending, "rows": table.to_dict(orient="records")}
    (config.RESULTS_DIR / "representations_levels_val.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    display = table[["representation", "level", "accuracy", "macro_f1", "rank_among_available"]]
    markdown = "# Comparación de niveles (validación)\n\n" + display.to_markdown(index=False, floatfmt=".4f")
    markdown += "\n\nSe comparan solo las representaciones disponibles. Las tareas tienen distinto número de clases; el aumento de F1 entre niveles no mide una mejora en una misma tarea.\n"
    if pending:
        markdown += "\nPendiente: " + " ".join(pending) + "\n"
    markdown += "\nNo se han leído predicciones de test ni calculado métricas de test.\n"
    (config.RESULTS_DIR / "representations_levels_val.md").write_text(markdown, encoding="utf-8")
    print(display.to_string(index=False))
    for message in pending:
        print("PENDIENTE:", message)


if __name__ == "__main__":
    main()
