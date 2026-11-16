from src.models.model_artifact_contract import dump_versioned_artifact, load_versioned_artifact, runtime_versions


def test_versioned_artifact_round_trip_and_runtime_metadata(tmp_path):
    path = tmp_path / "artifact.joblib"
    dump_versioned_artifact({"value": 1}, path, artifact_kind="test")
    assert load_versioned_artifact(path, artifact_kind="test") == {"value": 1}
    assert set(runtime_versions()) >= {"python", "scikit-learn", "lightgbm"}


def test_versioned_artifact_rejects_recorded_runtime_mismatch(tmp_path):
    path = tmp_path / "artifact.joblib"
    dump_versioned_artifact({"value": 1}, path, artifact_kind="test")
    import joblib
    envelope = joblib.load(path)
    envelope["runtime_versions"]["scikit-learn"] = "0.0.0"
    joblib.dump(envelope, path)
    try:
        load_versioned_artifact(path, artifact_kind="test")
    except RuntimeError as exc:
        assert "runtime mismatch" in str(exc)
    else:
        raise AssertionError("incompatible artifact was accepted")
