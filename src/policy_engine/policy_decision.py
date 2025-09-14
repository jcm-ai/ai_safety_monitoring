from __future__ import annotations
from typing import Dict, Any, List

def compute_max_risk(abuse_scores: Dict[str, float], crisis_score: float) -> float:
    return max([crisis_score] + list(abuse_scores.values()) if abuse_scores else [crisis_score])

class PolicyEngine:
    """
    Applies thresholds, age rules, and fairness guardrails to produce an action and rationale.
    Config keys: policy.yaml
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def decide(
        self,
        age: str,
        abuse: dict[str, float],
        crisis: float,
        escalation: dict[str, float],
        content_flags: dict[str, bool],
        crisis_labels: list[str] = None
    ) -> dict[str, any]:
        crisis_labels = crisis_labels or []

        max_risk = max([crisis, *abuse.values(), escalation["ewma"]])
        action = "allow"
        rationale = []
        route_to_human = False

        # Decision thresholds
        if max_risk >= self.cfg["actions"]["warn_max_risk"]:
            action = "block"
            rationale.append("High risk")
        elif max_risk >= self.cfg["actions"]["allow_max_risk"]:
            action = "warn"
            rationale.append("Moderate risk")

        # Fairness guardrails
        if self.cfg.get("fairness", {}).get("identity_terms"):
            rationale.append(f"Fairness guardrails active on: {self.cfg['fairness']['identity_terms']}")

        # Redaction logic
        redact = []
        for label in self.cfg["actions"]["redact_labels"]:
            if label in abuse or label in crisis_labels or content_flags.get(label, False):
                redact.append(label)

        # Routing logic
        if self.cfg["routing"].get("route_to_human_if_crisis") and "crisis" in crisis_labels:
            route_to_human = True
        elif self.cfg["routing"].get("route_if_blocked") and action == "block":
            route_to_human = True

        return {
            "action": action,
            "route_to_human": route_to_human,
            "max_risk": max_risk,
            "rationale": rationale,
            "redact": redact
        }
