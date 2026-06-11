import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.55
MODEL_PATH = "model.pkl"
DATA_PATH = "bangladesh_triage_dataset.csv"

SYMPTOM_COLUMNS = [
    "fever", "cough", "chest_pain", "shortness_of_breath", "headache",
    "dizziness", "vomiting", "abdominal_pain", "diarrhea", "constipation",
    "rash", "itching", "joint_pain", "back_pain", "swelling",
    "urine_problem", "fatigue", "anxiety", "depression", "seizure",
    "loss_of_consciousness", "sore_throat", "ear_pain"
]

DEPT_ICONS = {
    "Cardiology": "❤️", "Dermatology": "🩺", "ENT": "👂",
    "Emergency": "🚨", "Gastroenterology": "🫁", "General Medicine": "🏥",
    "Gynecology": "🌸", "Nephrology": "💧", "Neurology": "🧠",
    "Orthopedics": "🦴", "Pediatrics": "👶", "Psychiatry": "🧘",
    "Pulmonology": "🌬️"
}

label = lambda s: s.replace("_", " ").title()


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, criterion="entropy",
        oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def predict(model, symptom_values: dict):
    vector = np.zeros(len(SYMPTOM_COLUMNS))
    for s, v in symptom_values.items():
        vector[SYMPTOM_COLUMNS.index(s)] = v

    probs = model.predict_proba(vector.reshape(1, -1))[0]
    sorted_idx = np.argsort(probs)[::-1]

    max_idx = sorted_idx[0]
    max_prob = float(probs[max_idx])
    predicted = model.classes_[max_idx]

    # Warn but never override — let the top prediction show
    low_confidence = max_prob < CONFIDENCE_THRESHOLD

    top_3 = [
        {"department": model.classes_[i], "probability": round(float(probs[i]), 4)}
        for i in sorted_idx[:3]
    ]

    return predicted, max_prob, low_confidence, top_3


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Triage Assistant", page_icon="🏥", layout="centered")

st.markdown("""
<style>
  .block-container { max-width: 760px; padding-top: 2rem; }
  .badge {
    display: inline-block;
    background: #eff6ff; color: #2563eb;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 0.5rem;
  }
  .result-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem; margin-top: 1rem;
  }
  .dept-name { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
  .dept-sub { color: #64748b; font-size: 14px; margin-top: 2px; }
  .low-conf-box {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #92400e; margin-top: 1rem; line-height: 1.6;
  }
  .top3-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f1f5f9; font-size: 14px;
  }
  .top3-row:last-child { border-bottom: none; }
  .top3-pct { color: #2563eb; font-weight: 600; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown('<div class="badge">🔵 AI Triage System</div>', unsafe_allow_html=True)
st.title("Where should this patient go?")
st.caption("Select symptoms present, rate their severity, then get a department recommendation.")
st.divider()

model = load_model()

# =========================
# STEP 1: SELECT SYMPTOMS
# =========================
st.subheader("Step 1 — Select symptoms")
search = st.text_input("", placeholder="🔍  Search symptoms...", label_visibility="collapsed")
filtered = [s for s in SYMPTOM_COLUMNS if search.lower().replace(" ", "_") in s] if search else SYMPTOM_COLUMNS

cols = st.columns(3)
selected = []
for i, s in enumerate(filtered):
    with cols[i % 3]:
        if st.checkbox(label(s), key=f"chk_{s}"):
            selected.append(s)

# =========================
# STEP 2: RATE SEVERITY
# =========================
symptom_values = {}

if selected:
    st.divider()
    st.subheader("Step 2 — Rate severity")
    st.caption("0.0 = mild  ·  1.0 = severe")
    for s in selected:
        symptom_values[s] = st.slider(
            label(s), min_value=0.0, max_value=1.0,
            value=0.5, step=0.01, key=f"slider_{s}"
        )
    st.divider()

    if st.button("🔍  Get recommendation", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            dept, confidence, low_confidence, top_3 = predict(model, symptom_values)

        icon = DEPT_ICONS.get(dept, "🏥")
        pct = round(confidence * 100)
        bar_color = "#059669" if pct >= 75 else "#2563eb" if pct >= 55 else "#d97706"
        sub_text = "Low confidence — verify with a clinician" if low_confidence else "Recommended department"

        st.markdown(f"""
        <div class="result-box">
          <div style="display:flex; align-items:center; gap:14px; margin-bottom:1.2rem;">
            <div style="font-size:36px;">{icon}</div>
            <div>
              <div class="dept-name">{dept}</div>
              <div class="dept-sub">{sub_text}</div>
            </div>
          </div>
          <div style="font-size:12px; color:#64748b; margin-bottom:4px;">Model confidence</div>
          <div style="background:#e2e8f0; border-radius:4px; height:8px; overflow:hidden; margin-bottom:4px;">
            <div style="width:{pct}%; background:{bar_color}; height:100%; border-radius:4px;"></div>
          </div>
          <div style="font-size:13px; color:{bar_color}; font-weight:600; text-align:right;">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

        if low_confidence:
            st.markdown(f"""
            <div class="low-conf-box">
              ⚠️ <strong>Low confidence ({pct}%)</strong> — this symptom combination is uncommon in the training data.
              <strong>{dept}</strong> is the best guess, but please verify with a clinician before routing.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Top alternatives")
        rows = ""
        for item in top_3:
            p = round(item["probability"] * 100)
            d_icon = DEPT_ICONS.get(item["department"], "🏥")
            rows += f'<div class="top3-row"><span>{d_icon} <strong>{item["department"]}</strong></span><span class="top3-pct">{p}%</span></div>'
        st.markdown(f'<div style="border:1px solid #e2e8f0; border-radius:10px; padding:4px 12px;">{rows}</div>', unsafe_allow_html=True)

else:
    st.info("☝️ Select at least one symptom above to continue.")
