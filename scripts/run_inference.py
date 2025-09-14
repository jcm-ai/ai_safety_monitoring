import argparse
from src.config_loader import load_config
from src.orchestrator.inference_pipeline import InferenceOrchestrator
from pprint import pprint

def build_configs():
    return {
        "preprocessing": load_config(["configs/preprocessing.yaml"]),
        "models": load_config(["configs/models.yaml"]),
        "policy": load_config(["configs/policy.yaml"]),
        "ui": load_config(["configs/ui.yaml"]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True, help="Input message to analyze")
    ap.add_argument("--age", type=str, default="13+", help="User age group")
    args = ap.parse_args()

    cfgs = build_configs()
    orch = InferenceOrchestrator(cfgs)
    result = orch.infer(args.text, args.age)

    print("\n🧠 Decision:")
    pprint(result["decision"])
    print("\n📊 Model Outputs:")
    pprint({
        "abuse": result["abuse"],
        "crisis": result["crisis"],
        "escalation": result["escalation"],
        "content": result["content"]
    })

if __name__ == "__main__":
    main()
