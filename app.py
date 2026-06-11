"""
SciBlitz AI Challenge 2026: Live Triage Interface
Author: Senior Machine Learning Engineer & Medical Informatics Specialist
Description: Streamlit UI layer mapping 23 continuous symptom sliders to the
             localized Bangladesh Random Forest diagnostic core.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set page configurations to clean, professional layout
st.set_page_config(
    page_title="Intelligent Medical Triage Engine",
    page_icon="🏥",
    layout="wide"
)

# =====================================================================
# 1. CACHED MODEL TRAINING CORE (Ensures instant UI hot-reloads)
# =====================================================================
@st.cache_resource
def load_and_train_pipeline():
    """Reads dataset and fits the continuous ensemble model once."""
    if not os.path.exists("bangladesh_triage_dataset.csv"):
        return None, None
    
    df = pd.read_csv("bangladesh_triage_dataset.csv")
    df.columns = df.columns.str.strip()
    
    X = df.drop(columns=['target_department'])
    y = df['target_department'].str.strip()
    feature_cols = X.columns.tolist()
    
    # Train the exact optimized continuous architecture
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, 
        criterion='entropy', random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model, feature_cols

import os
model, feature_columns = load_and_train_pipeline()

# Handle missing file state gracefully
if model is None:
    st.error("❌ Critical System Error: 'bangladesh_triage_dataset.csv' not found in current execution directory.")
    st.stop()

# =====================================================================
# 2. UI HEADER & HOSPITAL BRANDING
# =====================================================================
st.title("🏥 Intelligent Medical Triage System")
st.subheader("SciBlitz AI Challenge 2026 — Advanced Continuous Intensity Protocol")
st.markdown("""
*Instead of restrictive checkboxes, patients adjust sliders from **0.0 (Absent)** to **1.0 (Severe/Extreme)**. 
The continuous decision criteria engine optimizes triage classification based on precise physical intensity layers.*
""")
st.write("---")

# Organize screen space: Left for inputs, Right for real-time model outputs
left_column, right_column = st.columns([3, 2], gap="large")

# =====================================================================
# 3. CONVERTING FEATURE SPECTRUMS TO INTERACTIVE SLIDERS (Left Column)
# =====================================================================
with left_column:
    st.markdown("### 📋 Patient Symptom Intake Form")
    st.caption("Drag the sliders to indicate the true physiological intensity of your current symptoms:")
    
    # Clean categorization of the 23 symptoms to make the UI look organized
    symptom_groups = {
        "🔴 Systemic & Vital Indicators": ["fever", "fatigue", "dizziness", "loss_of_consciousness"],
        "🫁 Cardio-Respiratory": ["cough", "chest_pain", "shortness_of_breath", "sore_throat"],
        "🧠 Neurological & Mental Health": ["headache", "seizure", "anxiety", "depression"],
        "🤢 Gastrointestinal": ["vomiting", "abdominal_pain", "diarrhea", "constipation"],
        "🦵 Musculoskeletal & Cutaneous": ["rash", "itching", "joint_pain", "back_pain", "swelling"],
        "👂 Otolaryngology & Renal": ["urine_problem", "ear_pain"]
    }
    
    # Initialize dictionary to capture runtime user inputs
    user_inputs = {}
    
    # Render sliders dynamically inside collapsible sections for scannability
    for group_name, symptoms in symptom_groups.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(2)  # Multi-column grid for space optimization
            for idx, sym in enumerate(symptoms):
                with cols[idx % 2]:
                    # Create clean display text from column names (e.g., chest_pain -> Chest Pain)
                    display_label = sym.replace("_", " ").title()
                    user_inputs[sym] = st.slider(
                        label=display_label,
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        key=f"slider_{sym}"
                    )

# =====================================================================
# 4. REAL-TIME AI INFERENCE ROUTING & SAFETY WRAPPING (Right Column)
# =====================================================================
with right_column:
    st.markdown("### ⚡ Live AI Engine Assessment")
    
    # Reconstruct the user input list into a verified array matches model features order
    input_vector = [user_inputs[col] for col in feature_columns]
    vector_parsed = np.array(input_vector).reshape(1, -1)
    
    # Run continuous model prediction distributions
    probabilities = model.predict_proba(vector_parsed)[0]
    class_labels = model.classes_
    
    probability_map = sorted(
        zip(class_labels, probabilities), key=lambda x: x[1], reverse=True
    )
    
    top_dept, top_conf = probability_map[0]
    
    # Clinical Safety Catch Threshold (55%)
    SAFETY_THRESHOLD = 0.55
    final_routing = top_dept
    is_overridden = False
    
    if top_conf < SAFETY_THRESHOLD:
        final_routing = "General Medicine"
        is_overridden = True
        
    # --- RENDER RESULTS PANEL ---
    st.markdown("#### Primary Routed Department Assignment:")
    
    if is_overridden:
        # Warning notification panel for ambiguous cases dropped to General Medicine clearinghouse
        st.warning(f"⚠️ **{final_routing}** (Safety Override Activated)")
        st.info(
            f"**Reason:** The raw engine intent favored *{top_dept}* with a confidence score of "
            f"**{top_conf*100:.1f}%**, which sits below our clinical safety clearance margin of **55%**. "
            f"Patient is securely rerouted to General Medicine for direct physical intake verification."
        )
    else:
        # High confidence medical match panel
        st.success(f"✅ **{final_routing}**")
        st.metric(label="System Matching Certainty Score", value=f"{top_conf * 100:.2f}%")
        
    st.write("---")
    
    # --- RENDER DIFFERENTIAL DIAGNOSTICS TREE ---
    st.markdown("#### 🔬 Automated Differential Diagnostics")
    st.caption("Secondary alternative specialties calculated across the continuous index spectrum:")
    
    # Create simple dataframe layout for analytics scannability
    diff_data = {
        "Specialty Specialty": [item[0] for item in probability_map[1:4]],
        "Confidence Probability": [f"{item[1]*100:.2f}%" for item in probability_map[1:4]]
    }
    st.table(pd.DataFrame(diff_data))