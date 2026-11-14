import json

import pytest

from src.experiments.final_holdout_evaluation import run_final_holdout_evaluation


def test_holdout_evaluation_refuses_existing_output(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "metrics.json").write_text(json.dumps({"selected_method": "sigmoid", "test_used": False}))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_final_holdout_evaluation(experiment, output)
