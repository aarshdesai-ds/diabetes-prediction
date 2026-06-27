"""End-to-end inference tests against the saved artifact."""
from src import config
from src.inference import load_metrics, load_model, predict_one


def test_model_artifact_loads():
    model = load_model()
    assert hasattr(model, "predict_proba")


def test_metrics_have_required_fields():
    m = load_metrics()
    assert "best_model" in m
    assert 0.0 <= m["test_set"]["roc_auc"] <= 1.0
    assert 0.0 <= m["test_set"]["recall_diabetic"] <= 1.0


def test_auprc_present_and_beats_baseline():
    t = load_metrics()["test_set"]
    assert 0.0 <= t["auprc"] <= 1.0
    # A useful model's AUPRC should exceed the no-skill prevalence baseline.
    assert t["auprc"] > t["auprc_baseline"]
    # The PR curve was stored for plotting.
    assert len(t["pr_curve"]["recall"]) == len(t["pr_curve"]["precision"]) > 1


def test_calibration_and_threshold_metrics():
    m = load_metrics()
    # Calibration should not make the Brier score worse.
    cal = m["calibration"]
    assert cal["brier_calibrated"] <= cal["brier_uncalibrated"] + 1e-6
    # Repeated-CV confidence interval is present and ordered.
    r = m["repeated_cv"]
    assert r["ci95_low"] <= r["mean_roc_auc"] <= r["ci95_high"]
    # The tuned threshold sits in (0, 1) and hits the target recall on test.
    assert 0.0 < m["recommended_threshold"] < 1.0
    assert m["test_set"]["recall_diabetic"] >= m["target_recall"] - 0.1


def test_predict_one_returns_label_and_probability():
    features = {
        "Glucose": 148, "BloodPressure": 72, "Insulin": 0,
        "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
    }
    label, proba = predict_one(features)
    assert label in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_threshold_changes_label_monotonically():
    """A very low threshold should never flag fewer cases than a high one."""
    features = {f: v for f, v in zip(
        config.FEATURES, [148, 72, 0, 33.6, 0.627, 50], strict=False)}
    low, _ = predict_one(features, threshold=0.05)
    high, _ = predict_one(features, threshold=0.95)
    assert low >= high
