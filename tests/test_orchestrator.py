from src.orchestrator.inference_pipeline import InferenceOrchestrator

def test_orchestrator_infer_basic():
    # Minimal config for testing
    cfgs = {
        "preprocessing": {
            "language_detection": {"enabled": True},
            "normalization": {"lower": True, "strip_urls": True, "strip_punctuation": True, "collapse_whitespace": True, "unicode_nfkc": True},
            "pii_masking": {"mask_email": True, "mask_phone": True, "email_token": "<EMAIL>", "phone_token": "<PHONE>"}
        },
        "models": {
            "abuse": {"labels": ["toxic", "threat"], "sklearn": {"vectorizer_max_features": 1000, "c": 1.0}},
            "crisis": {"sklearn": {"vectorizer_max_features": 1000, "c": 1.0}},
            "escalation": {"ewma_alpha": 0.3, "slope_window": 5, "risk_floor": 0.05},
            "content_filter": {
                "rules": {
                    "sexual_keywords": ["sex"],
                    "violence_keywords": ["kill"],
                    "substances_keywords": ["alcohol"]
                },
                "classifier": {"vectorizer_max_features": 1000, "c": 1.0}
            }
        },
        "policy": {
            "thresholds": {
                "abuse": {"toxic": 0.5, "threat": 0.5},
                "crisis": 0.5,
                "escalation": {"ewma_threshold": 0.5}
            },
            "actions": {
                "allow_max_risk": 0.4,
                "warn_max_risk": 0.7,
                "redact_labels": ["sexual"],
                "block_labels": ["threat", "crisis"]
            },
            "routing": {
                "route_to_human_if_crisis": True,
                "route_if_blocked": True
            },
            "age_rules": {
                "13+": {"prohibit": ["sexual"], "caution": ["violence"]}
            },
            "fairness": {
                "identity_terms": ["muslim", "dalit"]
            }
        },
        "ui": {}
    }

    orch = InferenceOrchestrator(cfgs)
    result = orch.infer("I will kill you", age="13+")

    assert "decision" in result
    assert result["decision"]["action"] in ("allow", "warn", "block")
    assert isinstance(result["abuse"]["scores"], dict)
    assert isinstance(result["crisis"]["score"], float)
    assert isinstance(result["escalation"]["ewma"], float)
    assert isinstance(result["content"]["rule_flags"], dict)
