"""Small smoke test that does not require or download the intrusion dataset."""
from nids.config import load_config
from nids.policy import confidence_alert
from nids.features import split_features_and_labels
import pandas as pd

def test_core_imports_and_default_configuration():
    config = load_config("config.yaml")
    assert config["data"]["dataset_version"] == "NF-CSE-CIC-IDS2018-v2"
    assert config["data"]["label_column"] == "Attack"
    assert confidence_alert(0.9, 0.8)["status"] == "high_priority"
    X, y = split_features_and_labels(pd.DataFrame({"Flow ID": ["f1"], "bytes": [42], "Label": ["Benign"]}), "Label")
    assert X.columns.tolist() == ["bytes"]
    assert y.tolist() == ["Benign"]
