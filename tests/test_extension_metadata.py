import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("extension_metadata", Path(__file__).resolve().parents[1] / "scripts" / "extension_metadata.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def sample():
    return pd.DataFrame({"totalCost": [10., 20.], "ecMaxContribution": [5., 10.],
        "startDate": ["2020-01-01", "2020-01-01"], "endDate": ["2021-01-01", None],
        "countryContr": ["BE|1 FR|0 BE|2", "ES|1"], "coordinatorCountry": ["BE", "BE"]})


def test_unique_participant_countries_include_zero_contribution():
    assert ext.metadata_frame(sample())["n_countries"].tolist() == [2, 1]


def test_validation_country_and_extreme_value_do_not_change_train_transform():
    train = sample()
    val = train.iloc[[0]].copy()
    val["coordinatorCountry"] = "ZZ"
    val["totalCost"] = 1e9
    matrices, countries = ext.encode_metadata(train, val, val, min_country_count=2)
    reference, _ = ext.encode_metadata(train, train, train, min_country_count=2)
    assert countries == ["BE"]
    assert np.allclose(matrices[0].toarray(), reference[0].toarray())
    assert all(np.isfinite(m.data).all() for m in matrices)
    assert matrices[1][0, -1] == 1
    assert matrices[1][0, 0] > 10
