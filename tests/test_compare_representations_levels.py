import importlib.util
from pathlib import Path
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("compare_levels", Path(__file__).resolve().parents[1] / "scripts" / "compare_representations_levels.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    expected = pd.DataFrame({"projectID": [1, 2], "label_l1": [23, 25]})
    predictions = pd.DataFrame({"projectID": [2, 1], "y_true": [25, 23], "y_pred": [23, 23]})
    return expected, predictions


def test_predictions_are_aligned_by_id_not_row_order():
    expected, predictions = sample()
    aligned = module.validate_predictions(predictions, expected, "label_l1")
    assert aligned["projectID"].tolist() == [1, 2]
    assert aligned["y_true"].tolist() == [23, 25]


@pytest.mark.parametrize("error", ["duplicate", "missing", "wrong_label", "wrong_class"])
def test_rejects_invalid_prediction_files(error):
    expected, predictions = sample()
    if error == "duplicate":
        predictions.loc[1, "projectID"] = 2
    elif error == "missing":
        predictions = predictions.iloc[:1]
    elif error == "wrong_label":
        predictions.loc[0, "y_true"] = 23
    else:
        predictions.loc[0, "y_pred"] = 999
    with pytest.raises(ValueError):
        module.validate_predictions(predictions, expected, "label_l1")
