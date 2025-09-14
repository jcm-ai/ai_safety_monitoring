import streamlit as st
from src.config_loader import load_config
from src.orchestrator.inference_pipeline import InferenceOrchestrator

st.set_page_config(page_title="AI Safety Monitor", layout="centered")

@st.cache_resource
def get_orchestrator():
    cfg = {
        "preprocessing": load_config(["configs/preprocessing.yaml"]),
        "models": load_config(["configs/models.yaml"]),
        "policy": load_config(["configs/policy.yaml"]),
        "ui": load_config(["configs/ui.yaml"]),
    }
    return InferenceOrchestrator(cfg)

# UI setup
st.title("🛡️ AI Safety Monitor")
orch = get_orchestrator()

age = st.selectbox("Select user age", options=["7+", "13+", "16+", "18+"], index=1)
msg = st.text_area("Enter a message", height=120, placeholder="Type a chat message...")

if st.button("Analyze") and msg.strip():
    result = orch.infer(msg, age)

    st.subheader("🧠 Decision")
    st.json(result["decision"])

    st.subheader("📊 Model Outputs")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Abuse Detection**")
        st.json(result["abuse"]["scores"])
        st.markdown(f"**Labels:** {result['abuse']['labels']}")
        st.markdown(f"**Crisis Score:** {result['crisis']['score']:.3f} → {result['crisis']['label']}")
        st.markdown(f"**Crisis Labels:** {', '.join(result['crisis']['labels'])}")

    with col2:
        st.markdown("**Escalation Tracker**")
        st.json({
            "EWMA": result["escalation"]["ewma"],
            "Slope": result["escalation"]["slope"]
        })
        st.markdown("**Content Filter**")
        st.json(result["content"])

    st.subheader("🛑 Redaction Preview")
    if result["decision"]["redact"]:
        st.markdown(f"🔒 The following labels will be redacted: `{', '.join(result['decision']['redact'])}`")
    else:
        st.markdown("✅ No redaction applied for this message.")

    st.caption("Real-time AI safety system 💡 Made with ❤️ by [Jagadish](https://jcm-ai.github.io)")
