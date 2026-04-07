import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ToMCoT · Theory of Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CSS  (no div wrappers around widgets — only
#  cosmetic styling of native Streamlit elements)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0c0d10;
    color: #ddd8cc;
}
.block-container {
    padding: 1.1rem 2rem 1rem 2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
footer { display: none !important; }

/* ── Remove Streamlit's default column gap noise ── */
[data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* ── Hero ── */
.hero {
    margin-bottom: 0.85rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #1e1f28;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 400;
    color: #f2ece0;
    margin: 0 0 0.15rem 0;
    line-height: 1.2;
}
.hero h1 em { color: #e8a830; font-style: italic; }
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3e4155;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Column background cards (pure CSS, no widget wrapping) ── */
[data-testid="column"]:first-child > [data-testid="stVerticalBlock"] {
    background: #111218;
    border: 1px solid #1d1e28;
    border-radius: 13px;
    padding: 1.2rem 1.3rem 1.3rem 1.3rem;
}
[data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
    background: #111218;
    border: 1px solid #1d1e28;
    border-radius: 13px;
    padding: 1.3rem 1.7rem 1.5rem 1.7rem;
    min-height: calc(100vh - 130px);
}

/* ── Input labels ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 2px !important;
    color: #e8a830 !important;
    text-transform: uppercase !important;
}

/* ── Input widgets ── */
div[data-testid="stSelectbox"] > div > div {
    background: #0d0e14 !important;
    border: 1px solid #252635 !important;
    border-radius: 7px !important;
    color: #ddd8cc !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.87rem !important;
}
textarea {
    background: #0d0e14 !important;
    border: 1px solid #252635 !important;
    border-radius: 7px !important;
    color: #ddd8cc !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.87rem !important;
    padding: 0.6rem !important;
    resize: none !important;
}
textarea:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #e8a830 !important;
    box-shadow: 0 0 0 2px rgba(232,168,48,0.1) !important;
}

/* ── Submit button ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: #e8a830 !important;
    color: #0c0d10 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin-top: 0.3rem !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: #f5b93a !important;
    box-shadow: 0 4px 18px rgba(232,168,48,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Right panel response components ── */
.right-placeholder {
    padding: 4rem 0;
    text-align: center;
    color: #252735;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.mental-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1d1e28;
    margin-bottom: 1rem;
}
.mental-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e8a830;
}
.mental-pill {
    background: rgba(232,168,48,0.1);
    border: 1px solid rgba(232,168,48,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #e8a830;
}

.resp-card {
    background: #0d0e14;
    border: 1px solid #252635;
    border-radius: 7px;
    padding: 1.1rem 1.15rem;
    margin-bottom: 0.75rem;
}
.resp-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e8a830;
    display: block;
    margin-bottom: 0.5rem;
}
.resp-card-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.75;
    color: #ddd8cc;
    min-height: 7rem;
}

.answer-box {
    background: #0f1f0a;
    border: 1px solid #224016;
    border-radius: 7px;
    padding: 0.9rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin-top: 0.2rem;
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e8a830;
}
.answer-letter {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #78cc50;
    line-height: 1;
}

.empty-hint {
    color: #2e3045;
    font-style: italic;
    font-size: 0.82rem;
}

/* Spinner */
div[data-testid="stSpinner"] > div { border-top-color: #e8a830 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────
MODELS = {
    "Base":      "Qwen/Qwen2.5-7B-Instruct",
    "SFT only":  "/path/to/your/sft-checkpoint",
    "SFT + DPO": "/path/to/your/sft-dpo-checkpoint",
}

GENERATION_KWARGS = dict(
    max_new_tokens=768,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

# ─────────────────────────────────────────────
#  Prompt builder (your prompt_v2)
# ─────────────────────────────────────────────
def prompt_v2(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    system_prompt = f"""
    You are an AI agent trained in cognitive modeling and mental state attribution. Your task is to apply Theory of Mind principles to identify and reason about a user's beliefs, desires, intentions, emotions, and thoughts from conversational cues.
    Objective: Given the user's input, conversation context, and social memory, infer the most likely mental state. Focus on the five mental state categories.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state categories: {{{mental_states}}}
    Instructions conditioning on the conversational context:
    1. Examine the provided inputs within the conversational context C_t.
    2. Identify the **most fitting mental state category** for the given inputs.
    3. Construct a **single** hypothesis regarding the user's mental state.
    4. The hypothesis should be a brief and precise explanation.
    5. Produce the final reply to the user's input by incorporating the hypothesis and identified mental state category.

    Output Format (Strictly follow this format):
    {{
        "Mental State Type": "one of the {{{mental_states}}}",
        "Hypothesis": "Hypothesis about the user's mental state",
        "Response": "Final response",
        "Final answer": "The final answer to the user's question in a single letter format (e.g., A, B, C, D)"
    }}
"""
    prompt = f"""
    - User Input (u_t): {question}
      Options: {", ".join(q_options)}
    - Conversational Context (C_t): {context}
"""
    return system_prompt, prompt

# ─────────────────────────────────────────────
#  Model loading + inference
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model

def run_inference(model_id: str, context: str, question: str, q_options: list[str]) -> dict:
    tokenizer, model = load_model(model_id)
    system_prompt, user_prompt = prompt_v2(context, question, q_options)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_KWARGS)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "Mental State Type": "unknown",
            "Hypothesis": raw,
            "Response": "(Could not parse a structured response.)",
            "Final answer": "?",
        }

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = {
        "Mental State Type": "",
        "Hypothesis": "",
        "Response": "",
        "Final answer": "",
    }
if "ran" not in st.session_state:
    st.session_state.ran = False

# ─────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ToMCoT: Tuning Language Models with a <em>Piece of Mind</em></h1>
  <div class="hero-sub">Theory of Mind Model Comparison</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Two-column layout  1/3  |  2/3
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

# ── LEFT: inputs ─────────────────────────────
with left_col:
    selected_label = st.selectbox("Select Model", options=list(MODELS.keys()))

    context_in = st.text_area("Context",
        placeholder="E.g. Alice makes a big mistake at work and causes a huge loss to the company. Her boss criticizes her and her colleagues come to comfort her.",
        height=130)

    question_in = st.text_area("Question",
        placeholder="e.g. What kind of emotion may the colleagues have?",
        height=75)

    options_in = st.text_area("Options",
        placeholder="A. …\nB. …\nC. …\nD. …",
        height=120)

    if st.button("▶  Submit"):
        if not context_in.strip() or not question_in.strip():
            st.warning("Please fill in at least Context and Question.")
        else:
            # Parse options into a list ["A. …", "B. …", …]
            q_options = [ln.strip() for ln in options_in.strip().splitlines() if ln.strip()]
            model_id = MODELS[selected_label]
            with st.spinner(f"Running {selected_label}…"):
                st.session_state.result = run_inference(
                    model_id, context_in, question_in, q_options
                )
            st.session_state.ran = True

# ── RIGHT: response ───────────────────────────
with right_col:
    r = st.session_state.result
    mental   = r.get("Mental State Type", "").strip()
    hyp      = r.get("Hypothesis", "").strip()
    response = r.get("Response", "").strip()
    answer   = r.get("Final answer", "").strip().upper()

    # 1 · Mental state pill (always shown; empty until submitted)
    st.markdown(f"""
    <div class="mental-row">
        <span class="mental-label">Mental State</span>
        <span class="mental-pill">{mental if mental else "&nbsp;"}</span>
    </div>
    """, unsafe_allow_html=True)

    # 2 · Hypothesis card
    st.markdown(f"""
    <div class="resp-card">
        <span class="resp-card-label">Hypothesis</span>
        <div class="resp-card-body resp-card-body--empty">{hyp if hyp else "<span class='empty-hint'>Awaiting response…</span>"}</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 · Response card
    st.markdown(f"""
    <div class="resp-card">
        <span class="resp-card-label">Final Response</span>
        <div class="resp-card-body">{response if response else "<span class='empty-hint'>Awaiting response…</span>"}</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 · Answer box
    st.markdown(f"""
    <div class="answer-box">
        <span class="answer-label">Answer</span>
        <span class="answer-letter">{answer if answer else "—"}</span>
    </div>
    """, unsafe_allow_html=True)
