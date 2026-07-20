from pdf_report import generate_report
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="DiaSense AI",
    page_icon="🏥",
    layout="wide"
)


# css

css_path = Path(__file__).parent / "style.css"

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)





st.info(
    """
    👋 Welcome to **DiaSense AI**.

    Enter your health information to estimate the risk of diabetes using a trained Machine Learning model.

    **Note:** This tool is for educational purposes and should not replace medical advice.
    """
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------------------------------------
# ENCODING MAPS
# --------------------------------------------------
gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_map = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🏥 DiaSense AI")

st.sidebar.markdown("---")

st.sidebar.success("AI-Based Early Diabetes Risk Assessment")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Vipul Varshney")

st.sidebar.markdown("### 🎓 Department")
st.sidebar.write("CSE (AI & ML)")

st.sidebar.markdown("### 🤖 Model")
st.sidebar.write("Random Forest Classifier")

st.sidebar.markdown("### 📊 Dataset")
st.sidebar.write("100,000 Patient Records")

st.sidebar.markdown("---")
# --------------------------------------------------
# HEADER
# --------------------------------------------------
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🏥 DiaSense AI")

st.subheader("Intelligent Early Diabetes Risk Assessment Platform")

st.write("Fill in the details below to estimate diabetes risk.")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Prediction",
    "📊 Model Information",
    "📈 Performance",
    "📉 Dataset Analysis",
    "ℹ️ About"
])

with tab1:

    # --------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------
    # --------------------------------------------------
# Dashboard Cards
# --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("🤖 AI Model")
        st.metric("Model", "Random Forest")

    with col2:
        st.info("📊 Dataset")
        st.metric("Records", "100K")

    with col3:
        st.info("🎯 Accuracy")
        st.metric("Accuracy", "96.9%")

    with col4:
        st.info("⚡ Status")
        st.metric("System", "Online")

    st.divider()

    # --------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------

    st.header("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input(
    "Patient Name",
    placeholder="Enter full name"
)
        
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        age = st.slider(
            "Age",
            1,
            100,
            25
        )

    with col2:

        height = st.number_input(
            "Height (cm)",
            100,
            250,
            170
        )

        weight = st.number_input(
            "Weight (kg)",
            20,
            200,
            70
        )

    # --------------------------------------------------
    # BMI
    # --------------------------------------------------
    bmi = weight / ((height / 100) ** 2)

    st.success(f"Calculated BMI : **{bmi:.2f}**")

    st.divider()

    # --------------------------------------------------
    # HEALTH PARAMETERS
    # --------------------------------------------------
    st.header("❤️ Health Parameters")

    col3, col4 = st.columns(2)

    with col3:

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

        heart = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

    with col4:

        hba1c = st.number_input(
            "HbA1c Level",
            3.0,
            10.0,
            5.5
        )

        glucose = st.number_input(
            "Blood Glucose Level",
            50,
            300,
            100
        )

    st.divider()

    # --------------------------------------------------
    # LIFESTYLE
    # --------------------------------------------------
    st.header("🚬 Lifestyle")

    smoking = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "ever",
            "not current",
            "No Info"
        ]
    )

    st.divider()

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    if st.button(
    "🩺 Analyze Diabetes Risk",
    use_container_width=True,
    type="primary"
):
        
        if patient_name.strip() == "":
            st.warning("⚠ Please enter the patient's name.")
            st.stop()

        gender_value = gender_map[gender]

        hypertension_value = 1 if hypertension == "Yes" else 0

        heart_value = 1 if heart == "Yes" else 0

        smoking_value = smoking_map[smoking]

        input_df = pd.DataFrame([[
            gender_value,
            age,
            hypertension_value,
            heart_value,
            smoking_value,
            bmi,
            hba1c,
            glucose
        ]],
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ])

        import time

        input_scaled = scaler.transform(input_df)

        with st.spinner("🧠 AI is analyzing the patient's health data..."):
            time.sleep(2)   # 2-second loading animation

            prediction = model.predict(input_scaled)[0]

            probability = model.predict_proba(input_scaled)[0][1]

        

        risk_percent = probability * 100

        health_score = max(0, 100 - risk_percent)

        # -------------------------
        # Risk Level
        # -------------------------
        if risk_percent < 20:
            risk_level = "🟢 Low"

        elif risk_percent < 50:
            risk_level = "🟡 Moderate"

        elif risk_percent < 75:
            risk_level = "🟠 High"

        else:
            risk_level = "🔴 Very High"

        # -------------------------
        # Top Cards
        # -------------------------
        
        st.header("📊 Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Risk Score",
                f"{risk_percent:.1f}%"
            )

        with col2:
            st.metric(
                "Health Score",
                f"{health_score:.1f}/100"
            )

        with col3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.divider()

        # -------------------------
        # Gauge Color
        # -------------------------
        if risk_percent < 20:
            bar_color = "green"
        elif risk_percent < 50:
            bar_color = "gold"
        elif risk_percent < 75:
            bar_color = "orange"
        else:
            bar_color = "red"

        
        
        st.subheader("📝 Patient Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Age",
                "BMI",
                "HbA1c",
                "Blood Glucose",
                "Smoking"
            ],
            "Value": [
                age,
                round(bmi, 2),
                hba1c,
                glucose,
                smoking
            ]
        })

        st.dataframe(summary, use_container_width=True)
        fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    title={"text": "Diabetes Risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": bar_color},
                        "steps": [
                            {"range": [0, 20], "color": "#90EE90"},
                            {"range": [20, 50], "color": "#FFF176"},
                            {"range": [50, 75], "color": "#FFB74D"},
                            {"range": [75, 100], "color": "#EF5350"},
                        ]
                    }
                ))

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Prediction Result
        # -------------------------
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes Detected")
        else:
            st.success("✅ Low Risk of Diabetes")

        # -------------------------
        # Recommendations
        # -------------------------
        st.subheader("🧠 AI Prediction Explanation")

        reasons = []

        if glucose >= 180:
            reasons.append("🩸 Blood glucose is significantly higher than the normal range.")

        elif glucose >= 140:
            reasons.append("🩸 Blood glucose is slightly elevated.")

        if hba1c >= 6.5:
            reasons.append("🧪 HbA1c indicates possible diabetes.")

        elif hba1c >= 5.7:
            reasons.append("🧪 HbA1c is in the pre-diabetic range.")

        if bmi >= 30:
            reasons.append("⚖️ BMI indicates obesity, which increases diabetes risk.")

        elif bmi >= 25:
            reasons.append("⚖️ BMI is above the healthy range.")

        if hypertension == "Yes":
            reasons.append("❤️ Hypertension is a known diabetes risk factor.")

        if heart == "Yes":
            reasons.append("💓 Heart disease increases overall health risk.")

        if smoking == "current":
            reasons.append("🚬 Current smoking increases long-term health risks.")

        if age >= 45:
            reasons.append("👤 Age above 45 is associated with increased diabetes risk.")

        if reasons:
            for reason in reasons:
                st.warning(reason)
        else:
            st.success("✅ All entered health parameters are within a healthy range.")
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        st.subheader("💡 Personalized Recommendations")

        recommendations = []

        if bmi >= 25:
            recommendations.append(
                "🏃 Reduce BMI through regular exercise and a balanced diet."
            )

        if glucose > 140:
            recommendations.append(
                "🩸 Blood glucose is elevated. Consult a healthcare professional."
            )

        if hba1c > 6.5:
            recommendations.append(
                "🧪 HbA1c level is high. Regular monitoring is recommended."
            )

        if hypertension == "Yes":
            recommendations.append(
                "❤️ Monitor blood pressure regularly."
            )

        if smoking == "current":
            recommendations.append(
                "🚭 Quitting smoking can improve your overall health."
            )

        if not recommendations:
            recommendations.append(
                "🎉 Great! Keep maintaining your healthy lifestyle."
            )

        for item in recommendations:
            st.info(item)
            
            
            
            generate_report(
    "report.pdf",
    patient_name,
    risk_percent,
    health_score,
    risk_level
)

with open("report.pdf", "rb") as pdf:

    st.download_button(
        "📄 Download Report",
        pdf,
        file_name="DiaSense_AI_Report.pdf",
        mime="application/pdf"
    )
    
    
    

    st.divider()

    st.markdown("---")

    st.markdown(
    """
    <div style='text-align:center'>

    ### 🏥 DiaSense AI

    Developed by **Vipul Varshney**

    B.Tech CSE (AI & ML)

    Random Forest Machine Learning Model

    © 2026 All Rights Reserved

    </div>
    """,
    unsafe_allow_html=True
    )
    
with tab2:

    st.header("📊 Model Information")

    st.success("Random Forest Classifier")

    st.markdown("### Dataset")

    st.write("• 100,000 Patient Records")

    st.write("• 8 Input Features")

    st.write("• Binary Classification")

    st.markdown("### Features Used")

    st.write("✅ Age")

    st.write("✅ Gender")

    st.write("✅ BMI")

    st.write("✅ Hypertension")

    st.write("✅ Heart Disease")

    st.write("✅ Smoking History")

    st.write("✅ HbA1c Level")

    st.write("✅ Blood Glucose Level")
    
    st.subheader("📈 Model Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Samples", "80,000")
        st.metric("Testing Samples", "20,000")

    with col2:
        st.metric("Features Used", "8")
        st.metric("Algorithm", "Random Forest")


with tab3:

    st.header("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "96.94%")

    with col2:
        st.metric("Precision", "87%")

    with col3:
        st.metric("Recall", "64%")

    with col4:
        st.metric("F1 Score", "79%")

    st.divider()

    st.subheader("📊 Confusion Matrix")

    st.image(
        "images/confusion_matrix.png",
        use_container_width=True
    )

    st.divider()

    st.subheader("⭐ Feature Importance")

    st.image(
        "images/feature_importance.png",
        use_container_width=True
    )
    
    
with tab4:

    st.header("📉 Exploratory Data Analysis")

    st.image("images/age_distribution.png", use_container_width=True)

    st.image("images/bmi_distribution.png", use_container_width=True)

    st.image("images/hba1c_distribution.png", use_container_width=True)

    st.image("images/blood_glucose_distribution.png", use_container_width=True)

    st.image("images/smoking_distribution.png", use_container_width=True)    

with tab5:

    st.header("ℹ️ About DiaSense AI")

    st.markdown("""
### Objective

DiaSense AI predicts the early risk of diabetes using Machine Learning.

### Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly

### Developer

Vipul Varshney

### Department

CSE (AI & ML)

### Disclaimer

This application is for educational purposes only.
""")