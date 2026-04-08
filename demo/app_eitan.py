import streamlit as st
import json
import requests
import re

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
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

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

[data-testid="stVerticalBlock"] { gap: 0.55rem; }

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

div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 2px !important;
    color: #e8a830 !important;
    text-transform: uppercase !important;
}

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

div[data-testid="stSpinner"] > div { border-top-color: #e8a830 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────
MODELS = {
    "Base":      "qwen_base:latest",   # pulls from Ollama library
    "SFT Model": "tomcot:latest",       # our fine-tuned model
}

OLLAMA_URL = "http://localhost:11434/api/chat"
# OLLAMA_URL = "http://localhost:11434/api/chat"

# ─────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────
def prompt_v2(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    letters = ", ".join([o[0] for o in q_options])
    system_prompt = f"""
    You are an AI agent trained in cognitive modeling and mental state attribution. Your task is to apply Theory of Mind principles to identify and reason about a user's beliefs, desires, intentions, emotions, and thoughts from conversational cues.
    Objective: Given the user's input, conversation context, and social memory, infer the most likely mental state. Focus on the five mental state categories.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state categories: {{{mental_states}}}
    Instructions:
    1. Examine the provided inputs within the conversational context C_t.
    2. Identify the most fitting mental state category.
    3. Construct a single hypothesis regarding the user's mental state.
    4. Produce a final response incorporating the hypothesis.

    Strictly follow this output format:
    - Hypothesis: <your hypothesis>
    - Response: <your response>
    - ANSWER: <one of {letters}>
"""
    user_prompt = f"""
    - User Input (u_t): {question}
      Options: {", ".join(q_options)}
    - Conversational Context (C_t): {context}
"""
    return system_prompt, user_prompt

# ─────────────────────────────────────────────
#  Inference via Ollama
# ─────────────────────────────────────────────
# 
def prompt_base(context, question, q_options):
    letters = ", ".join([o[0] for o in q_options])
    system_prompt = "You are a helpful assistant that reasons about mental states."
    user_prompt = f"""Context: {context}

    Question: {question}
    {chr(10).join(q_options)}

    Think through your reasoning, then end your response with ANSWER: X where X is one of {letters}."""
    return system_prompt, user_prompt

def run_inference(model_id: str, context: str, question: str, q_options: list) -> dict:
    if model_id == "qwen_base:latest":
        system_prompt, user_prompt = prompt_base(context, question, q_options)
    else:
        system_prompt, user_prompt = prompt_v2(context, question, q_options)
    
    # system_prompt, user_prompt = prompt_v2(context, question, q_options)

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 300,
            "stop": ["\nYou are", "- Correctness", "- Refutation"]
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw = response.json()["message"]["content"].strip()
    print(raw)

    if model_id == "qwen_base:latest":
        ans_match = re.search(r'\b([A-D])\b', raw)
        return {
            "Mental State Type": "N/A",
            "Hypothesis": "N/A",
            "Response": raw,
            "Final answer": ans_match.group(1) if ans_match else "?",
        }
    clean = raw.replace("```json", "").replace("```", "").strip()


    # Try JSON first
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: parse plain text format
    print("CLEAN:", repr(clean))
    hyp_match = re.search(r'-\s*Hypothesis:\s*(.+?)(?=\n\nAssistant:|$)', clean, re.DOTALL)
    sol_match = re.search(r'-\s*Response:\s*(.+?)(?=\\n\\nANSWER:|ANSWER:|$)', clean, re.DOTALL)
    ans_match  = re.search(r'ANSWER:\s*([A-D])', clean)

    if hyp_match or sol_match or ans_match:
        return {
            "Mental State Type": "Emotion",
            "Hypothesis": hyp_match.group(1).strip() if hyp_match else "",
            "Response":   sol_match.group(1).strip()  if sol_match  else "",
            "Final answer": ans_match.group(1).strip() if ans_match else "?",
        }

    return {
        "Mental State Type": "unknown",
        "Hypothesis": raw,
        "Response": "(Could not parse a structured response.)",
        "Final answer": "?",
    }
    


# ─────────────────────────────────────────────
#  Session state
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
#  Layout
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    selected_label = st.selectbox("Select Model", options=list(MODELS.keys()))

    context_in = st.text_area("Context",
    value="Alice makes a big mistake at work and causes a huge loss to the company. Her boss criticizes her and her colleagues come to comfort her.",
    height=130)

    question_in = st.text_area("Question",
        value="What kind of emotion may the colleagues have?",
        height=75)

    options_in = st.text_area("Options",
        value="A. Compassion\nB. Anger\nC. Indifference\nD. Jealousy",
        height=120)

    if st.button("▶  Submit"):
        if not context_in.strip() or not question_in.strip():
            st.warning("Please fill in at least Context and Question.")
        else:
            q_options = [ln.strip() for ln in options_in.strip().splitlines() if ln.strip()]
            model_id = MODELS[selected_label]
            with st.spinner(f"Running {selected_label}..."):
                st.session_state.result = run_inference(
                    model_id, context_in, question_in, q_options
                )
            st.session_state.ran = True

with right_col:
    r = st.session_state.result
    mental   = r.get("Mental State Type", "").strip()
    hyp      = r.get("Hypothesis", "").strip()
    response = r.get("Response", "").strip()
    answer   = r.get("Final answer", "").strip().upper()

    st.markdown(f"""
    <div class="mental-row">
        <span class="mental-label">Mental State</span>
        <span class="mental-pill">{mental if mental else "&nbsp;"}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="resp-card">
        <span class="resp-card-label">Hypothesis</span>
        <div class="resp-card-body">{hyp if hyp else "<span class='empty-hint'>Awaiting response...</span>"}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="resp-card">
        <span class="resp-card-label">Final Response</span>
        <div class="resp-card-body">{response if response else "<span class='empty-hint'>Awaiting response...</span>"}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-box">
        <span class="answer-label">Answer</span>
        <span class="answer-letter">{answer if answer else "—"}</span>
    </div>
    """, unsafe_allow_html=True)