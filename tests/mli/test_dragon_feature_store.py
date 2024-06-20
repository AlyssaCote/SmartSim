import pickle
import pytest
from smartsim._core.mli.infrastructure import DragonFeatureStore, DragonDict

@pytest.mark.parametrize(
    "expected_keys, expected_values",
    [
        pytest.param(["key1", "key2", "key3"], ["val1", "val2", "val3"]),
        pytest.param(["key1", "key2", "key3"], [b"val1", b"val2", b"val3"]),
        pytest.param([], []),
    ]
)
def test_dragon_feature_store_pickle(expected_keys, expected_values):
    dragon_dict = DragonDict()
    key_value_pair_list = zip(expected_keys, expected_values)
    for key, value in key_value_pair_list:
        dragon_dict[key] = value
    dragon_fs = DragonFeatureStore(dragon_dict)
    serialized = pickle.dumps(dragon_fs)
    deserialized = pickle.loads(serialized)
    for key, value in key_value_pair_list:
        assert deserialized[key] == value
