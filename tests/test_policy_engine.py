from src.policy_engine.policy_decision import PolicyEngine

def test_policy_allow_low_risk():
    config = {
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
    }

    engine = PolicyEngine(config)
    decision = engine.decide(
        age="13+",
        abuse={"toxic": 0.3, "threat": 0.2},
        crisis=0.1,
        escalation={"ewma": 0.2, "slope": 0.01},
        content_flags={"sexual": False, "violence": False}
    )

    assert decision["action"] == "allow"
    assert decision["route_to_human"] is False

def test_policy_block_crisis():
    config = {
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
    }

    engine = PolicyEngine(config)
    decision = engine.decide(
        age="13+",
        abuse={"toxic": 0.6, "threat": 0.3},
        crisis=0.8,
        escalation={"ewma": 0.6, "slope": 0.05},
        content_flags={"sexual": False, "violence": False}
    )

    assert decision["action"] == "block"
    assert decision["route_to_human"] is True
    assert "crisis" in decision["rationale"][0].lower() or "high risk" in decision["rationale"][0].lower()
