import json

import pytest

from crisis_vision.config import CrisisConfig


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CrisisConfig().to_dict()), encoding="utf-8")

    assert CrisisConfig.from_json(path) == CrisisConfig()


def test_unknown_key_fails_fast():
    with pytest.raises(ValueError, match="Unknown configuration keys: typo"):
        CrisisConfig.from_mapping({"typo": 1})
