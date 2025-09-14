from src.models.abuse_detector import AbuseDetector
from src.models.crisis_detector import CrisisDetector

def test_abuse_detector_fit_predict():
    config = {
        "labels": ["toxic", "threat"],
        "sklearn": {"vectorizer_max_features": 1000, "c": 1.0}
    }
    model = AbuseDetector(config)
    texts = ["you are kind", "i will hurt you"]
    labels = [["toxic"], ["threat"]]
    model.fit(texts, labels)

    out = model.predict(["hurt"], thresholds={"toxic": 0.5, "threat": 0.5})
    assert isinstance(out, list)
    assert "scores" in out[0]
    assert isinstance(out[0]["scores"], dict)

def test_crisis_detector_fit_predict():
    config = {"sklearn": {"vectorizer_max_features": 1000, "c": 1.0}}
    model = CrisisDetector(config)
    texts = ["i need help", "i'm fine"]
    labels = [1, 0]
    model.fit(texts, labels)

    out = model.predict(["i need help"], threshold=0.5)
    assert isinstance(out, list)
    assert "score" in out[0]
    assert out[0]["label"] in ("crisis", "non-crisis")
