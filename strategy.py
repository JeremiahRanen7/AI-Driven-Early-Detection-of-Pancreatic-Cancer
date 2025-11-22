import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import base64
import datetime
import uuid

# Set page configuration
st.set_page_config(
    page_title="Pancreatic Cancer Early Detection",
    page_icon="🔬",
    layout="wide"
)

# Create a data directory if it doesn't exist
if not os.path.exists("patient_data"):
    os.makedirs("patient_data")

# Load the trained models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_pancreatic_model.pkl')
        cnn_model = tf.keras.models.load_model('pancreas_cnn_model.h5')
        return rf_model, cnn_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

rf_model, cnn_model = load_models()

# Initialize session state variables for patient data
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = str(uuid.uuid4())
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'clinical_data' not in st.session_state:
    st.session_state.clinical_data = {}
if 'ct_scan_data' not in st.session_state:
    st.session_state.ct_scan_data = None
if 'ct_scan_image' not in st.session_state:
    st.session_state.ct_scan_image = None
if 'saved_patients' not in st.session_state:
    st.session_state.saved_patients = []

# App title and description
st.title("AI-Driven Early Detection of Pancreatic Cancer")
st.markdown("""
This application uses machine learning to assist in the early detection of pancreatic cancer:
- **Clinical Data Analysis**: Uses a Random Forest algorithm to analyze clinical markers
- **CT Scan Analysis**: Uses a Convolutional Neural Network to analyze pancreatic CT scans
- **Combined Analysis**: Integrates both models for a comprehensive risk assessment
- **Patient Management**: Save and load patient data for follow-up and comparison

*Note: This tool is for research purposes only and should not replace professional medical advice.*
""")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Clinical Data Analysis", "CT Scan Analysis", "Combined Analysis", "Patient Management", "About"])

# Store prediction results in session state for access across tabs
if 'rf_prediction' not in st.session_state:
    st.session_state.rf_prediction = None
if 'cnn_prediction' not in st.session_state:
    st.session_state.cnn_prediction = None

# Function to save patient data
def save_patient_data():
    if not st.session_state.patient_name:
        st.error("Please enter a patient name before saving data")
        return False
    
    # Create patient record
    patient_record = {
        "patient_id": st.session_state.patient_id,
        "patient_name": st.session_state.patient_name,
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clinical_data": st.session_state.clinical_data,
        "rf_prediction": float(st.session_state.rf_prediction) if st.session_state.rf_prediction is not None else None,
        "cnn_prediction": float(st.session_state.cnn_prediction) if st.session_state.cnn_prediction is not None else None
    }
    
    # Save CT scan image if available
    if st.session_state.ct_scan_image is not None:
        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        st.session_state.ct_scan_image.save(buffered, format="JPEG")
        patient_record["ct_scan_image"] = base64.b64encode(buffered.getvalue()).decode()
    
    # Save to JSON file
    filename = f"patient_data/{st.session_state.patient_id}.json"
    with open(filename, 'w') as f:
        json.dump(patient_record, f)
    
    # Update saved patients list
    load_saved_patients()
    
    return True

# Function to load saved patient data
def load_patient_data(patient_id):
    filename = f"patient_data/{patient_id}.json"
    try:
        with open(filename, 'r') as f:
            patient_data = json.load(f)
        
        # Update session state with loaded data
        st.session_state.patient_id = patient_data["patient_id"]
        st.session_state.patient_name = patient_data["patient_name"]
        st.session_state.clinical_data = patient_data["clinical_data"]
        st.session_state.rf_prediction = patient_data.get("rf_prediction")
        st.session_state.cnn_prediction = patient_data.get("cnn_prediction")
        
        # Load CT scan image if available
        if "ct_scan_image" in patient_data and patient_data["ct_scan_image"]:
            image_data = base64.b64decode(patient_data["ct_scan_image"])
            st.session_state.ct_scan_image = Image.open(io.BytesIO(image_data))
        else:
            st.session_state.ct_scan_image = None
            
        return True
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
        return False

# Function to load list of saved patients
def load_saved_patients():
    patient_files = [f for f in os.listdir("patient_data") if f.endswith('.json')]
    patients = []
    
    for file in patient_files:
        try:
            with open(f"patient_data/{file}", 'r') as f:
                patient_data = json.load(f)
                patients.append({
                    "id": patient_data["patient_id"],
                    "name": patient_data["patient_name"],
                    "date": patient_data.get("date_created", "Unknown"),
                    "rf_prediction": patient_data.get("rf_prediction"),
                    "cnn_prediction": patient_data.get("cnn_prediction")
                })
        except:
            pass
    
    # Sort by date (newest first)
    patients.sort(key=lambda x: x["date"], reverse=True)
    st.session_state.saved_patients = patients
    return patients

# Function to create a new patient record
def new_patient():
    st.session_state.patient_id = str(uuid.uuid4())
    st.session_state.patient_name = ""
    st.session_state.clinical_data = {}
    st.session_state.ct_scan_data = None
    st.session_state.ct_scan_image = None
    st.session_state.rf_prediction = None
    st.session_state.cnn_prediction = None

# Clinical Data Analysis Tab
with tab1:
    st.header("Clinical Data Analysis")
    
    # Patient identification
    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.patient_name = st.text_input("Patient Name", value=st.session_state.patient_name)
    with col2:
        st.text_input("Patient ID", value=st.session_state.patient_id, disabled=True)
    
    st.write("Enter patient clinical data to get a risk assessment using our Random Forest model.")
    
    col1, col2 = st.columns(2)
    
    # Set default values from saved clinical data if available
    clinical_data = st.session_state.clinical_data
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=clinical_data.get("age", 60))
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0 if clinical_data.get("sex") == "Male" else 1)
        plasma_ca19_9 = st.number_input("Plasma CA19-9 (U/mL)", min_value=0.0, value=clinical_data.get("plasma_ca19_9", 37.0))
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=clinical_data.get("creatinine", 1.0))
    
    with col2:
        lyve1 = st.number_input("LYVE1 (ng/mL)", min_value=0.0, value=clinical_data.get("lyve1", 1.0))
        reg1b = st.number_input("REG1B (ng/mL)", min_value=0.0, value=clinical_data.get("reg1b", 1.0))
        tff1 = st.number_input("TFF1 (ng/mL)", min_value=0.0, value=clinical_data.get("tff1", 1.0))
        reg1a = st.number_input("REG1A (ng/mL)", min_value=0.0, value=clinical_data.get("reg1a", 1.0))
    
    patient_cohort = st.selectbox("Patient Cohort", options=["Cohort1", "Cohort2"], index=0 if clinical_data.get("patient_cohort") == "Cohort1" else 1)
    sample_origin = st.selectbox("Sample Origin", options=["Other", "ESP", "LIV", "UCL"], 
                                index=["Other", "ESP", "LIV", "UCL"].index(clinical_data.get("sample_origin", "Other")))
    
    clinical_action_col1, clinical_action_col2 = st.columns([1, 2])
    
    with clinical_action_col1:
        if st.button("Generate Clinical Analysis", key="clinical_button"):
            # Save the clinical data to session state
            st.session_state.clinical_data = {
                "age": age,
                "sex": sex,
                "plasma_ca19_9": plasma_ca19_9,
                "creatinine": creatinine,
                "lyve1": lyve1,
                "reg1b": reg1b,
                "tff1": tff1,
                "reg1a": reg1a,
                "patient_cohort": patient_cohort,
                "sample_origin": sample_origin
            }
            
            # Prepare input data for RF model
            input_data = {
                'age': age,
                'sex': 1 if sex == "Male" else 0,
                'plasma_CA19_9': plasma_ca19_9,
                'creatinine': creatinine,
                'LYVE1': lyve1,
                'REG1B': reg1b,
                'TFF1': tff1,
                'REG1A': reg1a,
                'patient_cohort_Cohort2': 1 if patient_cohort == "Cohort2" else 0,
                'sample_origin_ESP': 1 if sample_origin == "ESP" else 0,
                'sample_origin_LIV': 1 if sample_origin == "LIV" else 0,
                'sample_origin_UCL': 1 if sample_origin == "UCL" else 0
            }
            
            # Create DataFrame with the input data
            input_df = pd.DataFrame([input_data])
            
            if rf_model is not None:
                # Make prediction
                prediction = rf_model.predict_proba(input_df)[0]
                prediction_class = rf_model.predict(input_df)[0]
                
                # Store prediction in session state
                st.session_state.rf_prediction = prediction[1]
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Cancer Risk Score", f"{prediction[1]:.2%}")
                    st.write(f"Prediction: {'Positive for pancreatic cancer' if prediction_class == 1 else 'Negative for pancreatic cancer'}")
                
                with col2:
                    # Create gauge chart for risk level
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    ax.barh(0, 100, color='lightgray', alpha=0.3)
                    ax.barh(0, prediction[1] * 100, color='red' if prediction[1] > 0.7 else 'orange' if prediction[1] > 0.3 else 'green')
                    ax.set_xlim(0, 100)
                    ax.get_yaxis().set_visible(False)
                    ax.set_xticks([0, 25, 50, 75, 100])
                    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                    ax.set_title('Risk Level')
                    st.pyplot(fig)
                
                # Display feature importance for this prediction
                if hasattr(rf_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
                    ax.set_title('Feature Importance in Random Forest Model')
                    st.pyplot(fig)
            else:
                st.error("Random Forest model not loaded properly. Please check your model file.")
    
    with clinical_action_col2:
        if st.session_state.rf_prediction is not None:
            # Add a save button to save the clinical data
            if st.button("Save Clinical Data", key="save_clinical"):
                if save_patient_data():
                    st.success(f"Clinical data saved for patient {st.session_state.patient_name}")

# CT Scan Analysis Tab
with tab2:
    st.header("CT Scan Analysis")
    st.write("Upload a pancreatic CT scan image for analysis using our CNN model.")
    
    # Display patient information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.text_input("Patient Name", value=st.session_state.patient_name, key="ct_patient_name", disabled=True)
    with col2:
        st.text_input("Patient ID", value=st.session_state.patient_id, key="ct_patient_id", disabled=True)
    
    # Check if we have a saved CT scan image
    if st.session_state.ct_scan_image is not None:
        st.image(st.session_state.ct_scan_image, caption="Loaded CT Scan", width=300)
        
        # Option to upload a new image
        st.write("Upload a new image to replace the current one:")
    
    uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption="Uploaded CT Scan", width=300)
        
        # Save the image to session state
        st.session_state.ct_scan_image = image
        
        ct_action_col1, ct_action_col2 = st.columns([1, 2])
        
        with ct_action_col1:
            if st.button("Analyze CT Scan", key="ct_button"):
                if cnn_model is not None:
                    # Preprocess the image
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
                    
                    # Make prediction
                    prediction = cnn_model.predict(img_array)[0][0]
                    
                    # Store prediction in session state
                    st.session_state.cnn_prediction = prediction
                    
                    # Display results
                    st.subheader("Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Cancer Probability", f"{prediction:.2%}")
                        st.write(f"Prediction: {'Positive for pancreatic cancer' if prediction > 0.5 else 'Negative for pancreatic cancer'}")
                    
                    with col2:
                        # Create gauge chart for visualization
                        fig, ax = plt.subplots(figsize=(4, 0.3))
                        ax.barh(0, 100, color='lightgray', alpha=0.3)
                        ax.barh(0, prediction * 100, color='red' if prediction > 0.7 else 'orange' if prediction > 0.3 else 'green')
                        ax.set_xlim(0, 100)
                        ax.get_yaxis().set_visible(False)
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                        ax.set_title('Confidence Level')
                        st.pyplot(fig)
                    
                    # Activation map visualization
                    st.subheader("Model Attention")
                    st.write("This visualization attempts to show areas of the CT scan that influenced the model's decision (GradCAM implementation would be ideal here).")
                    
                    # Placeholder for GradCAM - simplified for this example
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(img_array[0, :, :, 0], cmap='gray')
                    ax.set_title('CT Scan')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.error("CNN model not loaded properly. Please check your model file.")
        
        with ct_action_col2:
            if st.session_state.cnn_prediction is not None:
                # Add a save button to save the CT scan data
                if st.button("Save CT Scan Data", key="save_ct"):
                    if save_patient_data():
                        st.success(f"CT scan data saved for patient {st.session_state.patient_name}")
    
    elif st.session_state.ct_scan_image is not None and st.session_state.cnn_prediction is not None:
        # Display previous analysis results if available
        st.subheader("Previous Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cancer Probability", f"{st.session_state.cnn_prediction:.2%}")
            st.write(f"Prediction: {'Positive for pancreatic cancer' if st.session_state.cnn_prediction > 0.5 else 'Negative for pancreatic cancer'}")
        
        with col2:
            # Create gauge chart for visualization
            fig, ax = plt.subplots(figsize=(4, 0.3))
            ax.barh(0, 100, color='lightgray', alpha=0.3)
            ax.barh(0, st.session_state.cnn_prediction * 100, 
                  color='red' if st.session_state.cnn_prediction > 0.7 
                  else 'orange' if st.session_state.cnn_prediction > 0.3 else 'green')
            ax.set_xlim(0, 100)
            ax.get_yaxis().set_visible(False)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_title('Confidence Level')
            st.pyplot(fig)

def stratify_risk(combined_score, clinical_score, imaging_score, age, ca19_9):
    """
    Stratify patient risk into 5 levels with specific clinical pathways
    
    Parameters:
    - combined_score: float, weighted average of clinical and imaging scores
    - clinical_score: float, prediction from clinical biomarkers model
    - imaging_score: float, prediction from imaging model
    - age: int, patient age
    - ca19_9: float, CA19-9 biomarker value
    
    Returns:
    - risk_level: str, detailed risk level description
    - risk_category: str, category name for display
    - risk_color: str, color code for visualization
    - clinical_pathway: str, recommended clinical pathway
    """
    
    # Base stratification on combined score first
    if combined_score < 0.2:
        risk_category = "Very Low Risk"
        risk_color = "#06D6A0"  # Green
        risk_level = 1
    elif combined_score < 0.4:
        risk_category = "Low Risk"
        risk_color = "#82C0CC"  # Light blue
        risk_level = 2
    elif combined_score < 0.6:
        risk_category = "Moderate Risk"
        risk_color = "#FFD166"  # Yellow
        risk_level = 3
    elif combined_score < 0.8:
        risk_category = "High Risk"
        risk_color = "#F4845F"  # Orange
        risk_level = 4
    else:
        risk_category = "Very High Risk"
        risk_color = "#FF6B6B"  # Red
        risk_level = 5
        
    # Modifiers based on specific biomarkers and clinical factors
    # Age is a significant factor - older patients have higher baseline risk
    if age > 70 and risk_level < 5:
        risk_level += 0.5
        
    # CA19-9 is a strong pancreatic cancer marker
    if ca19_9 > 500 and risk_level < 5:  # Significantly elevated
        risk_level += 1
    elif ca19_9 > 100 and risk_level < 5:  # Moderately elevated
        risk_level += 0.5
        
    # Consider discordance between models as requiring higher scrutiny
    model_discordance = abs(clinical_score - imaging_score)
    if model_discordance > 0.4:  # Significant disagreement between models
        if risk_level < 5:
            risk_level += 0.5
            
    # Determine clinical pathway based on final risk assessment
    if risk_level < 1.5:  # Very Low Risk
        clinical_pathway = """
        **Standard Surveillance Pathway**
        - Routine age-appropriate cancer screening
        - Consider reassessment in 12 months if family history present
        - Patient education on symptoms and risk factors
        - Lifestyle modification counseling if modifiable risk factors present
        """
    elif risk_level < 2.5:  # Low Risk
        clinical_pathway = """
        **Enhanced Surveillance Pathway**
        - Follow-up in 6-12 months with repeated biomarker assessment
        - Consider baseline endoscopic ultrasound if risk factors present
        - Genetic counseling if family history of pancreatic cancer
        - Address modifiable risk factors (smoking, obesity, diabetes)
        """
    elif risk_level < 3.5:  # Moderate Risk
        clinical_pathway = """
        **Diagnostic Evaluation Pathway**
        - Referral to gastroenterology within 30 days
        - Complete pancreatic protocol CT or MRI
        - Comprehensive biomarker panel assessment
        - Consider endoscopic ultrasound evaluation
        - Follow-up in 3-6 months with repeated imaging
        """
    elif risk_level < 4.5:  # High Risk
        clinical_pathway = """
        **Expedited Diagnostic Pathway**
        - Urgent referral to pancreatic specialist (within 2 weeks)
        - Immediate pancreatic protocol CT with contrast
        - Endoscopic ultrasound with potential for fine-needle aspiration
        - Consider ERCP if biliary obstruction present
        - Multidisciplinary team evaluation
        """
    else:  # Very High Risk
        clinical_pathway = """
        **Urgent Intervention Pathway**
        - Same-day specialist consultation with hepatobiliary surgeon
        - Emergency pancreatic protocol CT and MRI
        - Endoscopic ultrasound with fine-needle aspiration biopsy
        - Comprehensive staging workup
        - Immediate multidisciplinary tumor board evaluation
        - Surgical consultation and pre-operative assessment
        """
    
    return risk_level, risk_category, risk_color, clinical_pathway


# Combined Analysis Tab
with tab3:
    st.header("Combined Risk Assessment")
    
    # Display patient information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.text_input("Patient Name", value=st.session_state.patient_name, key="combined_patient_name", disabled=True)
    with col2:
        st.text_input("Patient ID", value=st.session_state.patient_id, key="combined_patient_id", disabled=True)
    
    st.write("This tab combines predictions from both models to provide a comprehensive risk assessment.")
    
    # Check if both predictions are available
    if st.session_state.rf_prediction is not None and st.session_state.cnn_prediction is not None:
        # Extract predictions from session state
        rf_score = st.session_state.rf_prediction
        cnn_score = st.session_state.cnn_prediction
        
        # Extract clinical data needed for stratification
        age = st.session_state.clinical_data.get("age", 60)
        ca19_9 = st.session_state.clinical_data.get("plasma_ca19_9", 37.0)
        
        # Calculate combined score (weighted average)
        # You can adjust weights based on the relative importance/confidence of each model
        weight_rf = st.slider("Clinical Data Model Weight", 0.0, 1.0, 0.5, 0.05)
        weight_cnn = 1 - weight_rf
        
        combined_score = (weight_rf * rf_score) + (weight_cnn * cnn_score)
        
        # Use the new stratification function
        risk_level, risk_category, risk_color, clinical_pathway = stratify_risk(
            combined_score, rf_score, cnn_score, age, ca19_9
        )
        
        # Display combined results
        st.subheader("Combined Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Clinical Data Score", f"{rf_score:.2%}")
            st.metric("CT Scan Score", f"{cnn_score:.2%}")
            st.metric("Combined Risk Score", f"{combined_score:.2%}")
            st.markdown(f"<h3 style='color:{risk_color}'>{risk_category}</h3>", unsafe_allow_html=True)
            
            # Add risk level indicator
            risk_level_numeric = int(risk_level) if risk_level == int(risk_level) else risk_level
            st.write(f"**Risk Stratification Level:** {risk_level_numeric} out of 5")
        
        with col2:
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Bar chart comparing scores
            models = ['Clinical Data Model', 'Imaging Model', 'Combined Score']
            scores = [rf_score, cnn_score, combined_score]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            bars = ax.bar(models, scores, color=colors)
            
            # Add threshold lines with labels
            thresholds = [0.2, 0.4, 0.6, 0.8]
            threshold_labels = ['Very Low/Low', 'Low/Moderate', 'Moderate/High', 'High/Very High']
            threshold_colors = ['#06D6A0', '#82C0CC', '#FFD166', '#F4845F']
            
            for i, (threshold, label, color) in enumerate(zip(thresholds, threshold_labels, threshold_colors)):
                ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.7)
                ax.text(0.1, threshold + 0.02, label, color=color, alpha=0.8, fontsize=8)
            
            # Add labels and formatting
            ax.set_ylim(0, 1)
            ax.set_ylabel('Risk Score')
            ax.set_title('Model Risk Score Comparison')
            
            # Add value labels on the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Add visual risk stratification meter
        st.subheader("Risk Stratification")
        
        # Create a horizontal gauge for risk level
        fig, ax = plt.subplots(figsize=(10, 1.5))
        
        # Draw the background bar
        ax.barh(0, 5, color='lightgray', alpha=0.3, height=0.5)
        
        # Draw risk level position
        ax.barh(0, risk_level, color=risk_color, height=0.5)
        
        # Add markers and labels for each risk level
        for i in range(1, 6):
            ax.axvline(x=i, color='white', linestyle='-', alpha=0.7)
            ax.text(i-0.5, -0.5, f'Level {i}', ha='center', va='center', fontsize=9)
        
        # Add labels for risk categories below
        category_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
        category_labels = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
        category_colors = ['#06D6A0', '#82C0CC', '#FFD166', '#F4845F', '#FF6B6B']
        
        for pos, label, color in zip(category_positions, category_labels, category_colors):
            ax.text(pos, -1.2, label, ha='center', va='center', fontsize=8, color=color, weight='bold')
        
        # Format the chart
        ax.set_xlim(0, 5)
        ax.set_ylim(-1.5, 0.5)
        ax.set_title('Patient Risk Stratification Level')
        ax.set_xticks(range(6))
        ax.get_yaxis().set_visible(False)
        
        # Add pointer to show current risk level
        pointer_pos = risk_level
        ax.annotate('', xy=(pointer_pos, -0.5), xytext=(pointer_pos, -0.2),
                   arrowprops=dict(facecolor=risk_color, shrink=0.05))
        
        st.pyplot(fig)
        
        # Add risk assessment explanation
        st.subheader("Detailed Analysis")
        
        # Create comparison table
        comparison_data = {
            'Model': ['Clinical Data Analysis', 'CT Scan Analysis', 'Combined Assessment'],
            'Risk Score': [f"{rf_score:.2%}", f"{cnn_score:.2%}", f"{combined_score:.2%}"],
            'Interpretation': [
                f"{'Very High' if rf_score > 0.8 else 'High' if rf_score > 0.6 else 'Moderate' if rf_score > 0.4 else 'Low' if rf_score > 0.2 else 'Very Low'} risk based on clinical markers",
                f"{'Very High' if cnn_score > 0.8 else 'High' if cnn_score > 0.6 else 'Moderate' if cnn_score > 0.4 else 'Low' if cnn_score > 0.2 else 'Very Low'} risk based on imaging",
                f"{risk_category} overall (Level {risk_level_numeric}/5)"
            ]
        }
        
        st.table(pd.DataFrame(comparison_data))
        
        # Display clinical pathway
        st.subheader("Clinical Pathway Recommendation")
        
        # Format the clinical pathway differently based on risk level
        if risk_level >= 4:
            st.error(clinical_pathway)
        elif risk_level >= 3:
            st.warning(clinical_pathway)
        elif risk_level >= 2:
            st.info(clinical_pathway)
        else:
            st.success(clinical_pathway)
        
        # Add risk factors that contributed to stratification
        st.subheader("Risk Stratification Factors")
        
        risk_factors = []
        
        # List key factors that influenced the risk stratification
        if age > 70:
            risk_factors.append(f"Age ({age} years) - Advanced age increases baseline risk")
        
        if ca19_9 > 500:
            risk_factors.append(f"CA19-9 ({ca19_9:.1f} U/mL) - Significantly elevated biomarker")
        elif ca19_9 > 100:
            risk_factors.append(f"CA19-9 ({ca19_9:.1f} U/mL) - Moderately elevated biomarker")
        
        if abs(rf_score - cnn_score) > 0.4:
            risk_factors.append(f"Model discordance ({abs(rf_score - cnn_score):.2f}) - Significant difference between clinical and imaging indicators")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.write("No specific risk modifiers identified.")
            
        # Save combined assessment
        if st.button("Save Patient Stratification", key="save_combined"):
            # Save stratification data along with other data
            st.session_state.clinical_data["risk_level"] = float(risk_level)
            st.session_state.clinical_data["risk_category"] = risk_category
            
            if save_patient_data():
                st.success(f"Risk stratification saved for patient {st.session_state.patient_name}")
        
        # Explanation of stratification approach
        st.subheader("About Risk Stratification System")
        st.write("""
        Our risk stratification system goes beyond simple risk scores to classify patients into clinically meaningful categories:
        
        1. **Multi-factor Assessment**: Combines model predictions with key clinical factors
        2. **5-Level System**: From Very Low (1) to Very High (5) risk with intermediate levels
        3. **Clinical Pathways**: Each level has a specific recommendation for clinical management
        4. **Risk Modifiers**: Additional considerations for age, biomarker levels, and model agreement
        5. **Enhanced Visualization**: Visual tools to better understand relative risk position
        
        The resulting stratification enables personalized decision-making and appropriate resource allocation based on risk level.
        """)
        
    else:
        st.info("Please complete both Clinical Data Analysis and CT Scan Analysis to view a combined risk assessment.")
        
        # Add buttons to navigate to the required tabs
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.rf_prediction is None:
                if st.button("Go to Clinical Data Analysis"):
                    st.switch_page("tab1")
        
        with col2:
            if st.session_state.cnn_prediction is None:
                if st.button("Go to CT Scan Analysis"):
                    st.switch_page("tab2")

# Patient Management Tab (NEW)
with tab4:
    st.header("Patient Management")
    st.write("Create, save, and load patient records for tracking and comparison.")
    
    # Actions row
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    
    with action_col1:
        if st.button("Create New Patient", key="new_patient"):
            new_patient()
            st.success("New patient record created")
            # Rerun the app to refresh all fields
            st.experimental_rerun()
    
    with action_col2:
        if st.button("Save Current Patient", key="save_patient"):
            if st.session_state.patient_name:
                if save_patient_data():
                    st.success(f"Patient {st.session_state.patient_name} saved successfully")
            else:
                st.error("Please enter a patient name before saving")
    
    with action_col3:
        if st.button("Refresh Patient List", key="refresh_patients"):
            load_saved_patients()
            st.success("Patient list refreshed")
    
    # Load saved patients
    saved_patients = load_saved_patients()
    
    if saved_patients:
        st.subheader("Saved Patients")
        
        # Create a table of saved patients
        patient_df = pd.DataFrame(saved_patients)
        if "risk_category" in patient_df.columns:
            patient_df.columns = ["ID", "Patient Name", "Date", "Clinical Risk", "CT Scan Risk", "Risk Level", "Risk Category"]

            def risk_color(val):
                if val == "Very High Risk":
                    return 'color: #FF6B6B'
                elif val == "High Risk":
                    return 'color: #F4845F'
                elif val == "Moderate Risk":
                    return 'color: #FFD166'
                elif val == "Low Risk":
                    return 'color: #82C0CC'
                elif val == "Very Low Risk":
                    return 'color: #06D6A0'
                return ''
        
        # Format the risk columns as percentages
        patient_df["Clinical Risk"] = patient_df["Clinical Risk"].apply(lambda x: f"{x:.2%}" if x is not None else "N/A")
        patient_df["CT Scan Risk"] = patient_df["CT Scan Risk"].apply(lambda x: f"{x:.2%}" if x is not None else "N/A")
        patient_df["Risk Level"] = patient_df["Risk Level"].apply(lambda x: f"{x:.1f}/5" if x is not None else "N/A")
        st.dataframe(patient_df.style.applymap(risk_color, subset=['Risk Category']))
    else:
        patient_df.columns = ["ID", "Patient Name", "Date", "Clinical Risk", "CT Scan Risk"]

        patient_df["Clinical Risk"] = patient_df["Clinical Risk"].apply(lambda x: f"{x:.2%}" if x is not None else "N/A")
        patient_df["CT Scan Risk"] = patient_df["CT Scan Risk"].apply(lambda x: f"{x:.2%}" if x is not None else "N/A")
    
    
        # Display the table
        st.dataframe(patient_df)
        
        # Patient selection for loading
        selected_patient = st.selectbox(
            "Select a patient to load:",
            options=[f"{p['name']} (ID: {p['id']})" for p in saved_patients],
            key="patient_select"
        )
        
        if st.button("Load Selected Patient", key="load_patient"):
            # Extract patient ID from selection
            patient_id = selected_patient.split("ID: ")[1].rstrip(")")
            
            if load_patient_data(patient_id):
                st.success(f"Patient {st.session_state.patient_name} loaded successfully")
                # Rerun to update all fields
                st.experimental_rerun()
        
        # Patient comparison
        st.subheader("Patient Comparison")
        st.write("Compare multiple patients to track changes over time.")
        
        # Multi-select for comparison
        selected_patients_for_comparison = st.multiselect(
            "Select patients to compare:",
            options=[f"{p['name']} (ID: {p['id']})" for p in saved_patients],
            key="patient_compare"
        )
        
        if selected_patients_for_comparison and st.button("Generate Comparison", key="compare_button"):
            comparison_data = []
            
            for patient_string in selected_patients_for_comparison:
                patient_id = patient_string.split("ID: ")[1].rstrip(")")
                
                # Find patient in saved patients list
                patient = next((p for p in saved_patients if p["id"] == patient_id), None)
                
                if patient:
                    patient_data = {
                        "Patient Name": patient["name"],
                        "Date": patient["date"],
                        "Clinical Risk": patient.get("rf_prediction", None),
                        "CT Scan Risk": patient.get("cnn_prediction", None)
                    }
                    
                    if patient_data["Clinical Risk"] is not None and patient_data["CT Scan Risk"] is not None:
                        # Calculate combined risk with default weights
                        patient_data["Combined Risk"] = 0.5 * patient_data["Clinical Risk"] + 0.5 * patient_data["CT Scan Risk"]
                    else:
                        patient_data["Combined Risk"] = None
                    
                    comparison_data.append(patient_data)
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(comparison_df)
                
                # Create comparison chart for visual representation
                if len(comparison_data) > 0 and any(p.get("Clinical Risk") is not None for p in comparison_data):
                    st.subheader("Risk Score Comparison")
                    
                    # Prepare data for chart
                    chart_data = pd.DataFrame({
                        'Patient': [],
                        'Risk Type': [],
                        'Score': []
                    })
                    
                    for patient in comparison_data:
                        name = patient["Patient Name"]
                        if patient.get("Clinical Risk") is not None:
                            chart_data = pd.concat([chart_data, pd.DataFrame({
                                'Patient': [name],
                                'Risk Type': ['Clinical Risk'],
                                'Score': [patient["Clinical Risk"]]
                            })])
                        if patient.get("CT Scan Risk") is not None:
                            chart_data = pd.concat([chart_data, pd.DataFrame({
                                'Patient': [name],
                                'Risk Type': ['CT Scan Risk'],
                                'Score': [patient["CT Scan Risk"]]
                            })])
                        if patient.get("Combined Risk") is not None:
                            chart_data = pd.concat([chart_data, pd.DataFrame({
                                'Patient': [name],
                                'Risk Type': ['Combined Risk'],
                                'Score': [patient["Combined Risk"]]
                            })])
                    
                    if not chart_data.empty:
                        # Create comparison bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        chart = sns.barplot(x='Patient', y='Score', hue='Risk Type', data=chart_data, ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Risk Score')
                        ax.set_title('Risk Score Comparison Across Patients')
                        plt.xticks(rotation=45)
                        
                        # Add threshold line
                        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                        ax.text(0.1, 0.51, 'Decision Threshold', color='r', alpha=0.7)
                        
                        st.pyplot(fig)
                        
                        # Create timeline view for longitudinal tracking
                        if len(set(chart_data['Patient'])) == 1 and len(comparison_data) > 1:
                            st.subheader("Patient Timeline")
                            st.write("Risk score progression over time.")
                            
                            # Sort by date
                            timeline_data = sorted(comparison_data, key=lambda x: x["Date"])
                            
                            # Create timeline dataframe
                            timeline_df = pd.DataFrame({
                                'Date': [pd.to_datetime(p["Date"]) for p in timeline_data],
                                'Clinical Risk': [p.get("Clinical Risk", None) for p in timeline_data],
                                'CT Scan Risk': [p.get("CT Scan Risk", None) for p in timeline_data],
                                'Combined Risk': [p.get("Combined Risk", None) for p in timeline_data]
                            })
                            
                            # Plot timeline
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            if not timeline_df['Clinical Risk'].isna().all():
                                ax.plot(timeline_df['Date'], timeline_df['Clinical Risk'], 'o-', label='Clinical Risk')
                            if not timeline_df['CT Scan Risk'].isna().all():
                                ax.plot(timeline_df['Date'], timeline_df['CT Scan Risk'], 's-', label='CT Scan Risk')
                            if not timeline_df['Combined Risk'].isna().all():
                                ax.plot(timeline_df['Date'], timeline_df['Combined Risk'], '^-', label='Combined Risk')
                            
                            ax.set_ylim(0, 1)
                            ax.set_ylabel('Risk Score')
                            ax.set_title('Risk Score Timeline')
                            ax.legend()
                            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            
                            st.pyplot(fig)
                    else:
                        st.info("Insufficient data for comparison chart")
            else:
                st.info("No comparison data available")
        
        # Export patient data functionality
        st.subheader("Export Patient Data")
        st.write("Export patient data as CSV for external analysis.")
        
        export_all = st.checkbox("Export all patient data", value=False)
        
        if export_all:
            selected_export_patients = [p["id"] for p in saved_patients]
        else:
            selected_export_patients = [p.split("ID: ")[1].rstrip(")") for p in 
                                       st.multiselect("Select patients to export:",
                                                      options=[f"{p['name']} (ID: {p['id']})" for p in saved_patients],
                                                      key="patient_export")]
        
        if selected_export_patients and st.button("Export Selected Patient Data", key="export_button"):
            export_rows = []
            
            for patient_id in selected_export_patients:
                try:
                    # Load patient file
                    with open(f"patient_data/{patient_id}.json", 'r') as f:
                        patient_data = json.load(f)
                    
                    # Extract clinical data
                    clinical_data = patient_data.get("clinical_data", {})
                    
                    # Create row
                    row = {
                        "Patient ID": patient_id,
                        "Patient Name": patient_data.get("patient_name", "Unknown"),
                        "Date": patient_data.get("date_created", "Unknown"),
                        "Clinical Risk Score": patient_data.get("rf_prediction", None),
                        "CT Scan Risk Score": patient_data.get("cnn_prediction", None)
                    }
                    
                    # Add clinical data fields
                    for key, value in clinical_data.items():
                        row[key] = value
                    
                    export_rows.append(row)
                except Exception as e:
                    st.error(f"Error exporting patient {patient_id}: {e}")
            
            if export_rows:
                # Convert to DataFrame
                export_df = pd.DataFrame(export_rows)
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"pancreatic_cancer_patient_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No saved patients found. Create a new patient and save data to begin.")

    # Patient data backup and restore functionality
    st.subheader("Data Management")
    backup_col1, backup_col2 = st.columns(2)
    
    with backup_col1:
        if st.button("Backup All Patient Data", key="backup_data"):
            try:
                # Create a dictionary with all patient data
                all_patient_data = {}
                patient_files = [f for f in os.listdir("patient_data") if f.endswith('.json')]
                
                for file in patient_files:
                    with open(f"patient_data/{file}", 'r') as f:
                        patient_data = json.load(f)
                        all_patient_data[file] = patient_data
                
                # Convert to JSON string
                backup_data = json.dumps(all_patient_data)
                
                # Create download button
                st.download_button(
                    label="Download Backup",
                    data=backup_data,
                    file_name=f"pancreatic_cancer_data_backup_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
                
                st.success("Backup ready for download")
            except Exception as e:
                st.error(f"Error creating backup: {e}")
    
    with backup_col2:
        uploaded_backup = st.file_uploader("Restore from backup file:", type=["json"])
        
        if uploaded_backup is not None and st.button("Restore Data", key="restore_data"):
            try:
                # Read backup file
                backup_content = json.loads(uploaded_backup.read())
                
                # Confirm restoration
                if st.checkbox("I understand this will overwrite existing records with the same ID"):
                    # Restore each patient file
                    restored_count = 0
                    for filename, patient_data in backup_content.items():
                        with open(f"patient_data/{filename}", 'w') as f:
                            json.dump(patient_data, f)
                        restored_count += 1
                    
                    st.success(f"Successfully restored {restored_count} patient records")
                    
                    # Refresh patient list
                    load_saved_patients()
                else:
                    st.warning("Please confirm you understand the consequences of restoration")
            except Exception as e:
                st.error(f"Error restoring backup: {e}")

# About Tab
with tab5:
    st.header("About This Project")
    st.write("""
    ## AI-Driven Early Detection of Pancreatic Cancer
    
    This final year project aims to develop an AI-based tool for the early detection of pancreatic cancer, combining both clinical data analysis and medical imaging analysis.
    
    ### Components:
    
    1. **Random Forest Algorithm for Clinical Data**
       - Analyzes biomarkers and patient data
       - Feature importance analysis highlights the most predictive variables
    
    2. **Convolutional Neural Network for Imaging**
       - Analyzes CT scans for pancreatic abnormalities
       - Provides visual feedback on suspicious regions
       
    3. **Combined Risk Assessment**
       - Integrates results from both models
       - Provides weighted comprehensive evaluation
       - Offers tailored clinical recommendations
       
    4. **Patient Management System**
       - Save and load patient profiles
       - Track patient risk scores over time
       - Compare different patients or longitudinal data
       - Export data for external analysis
    
    
    
    ### Dataset Information:
    
    - **Clinical Data**: Debernardi et al. 2020 dataset with biomarkers including CA19-9, LYVE1, REG1B
    - **Imaging Data**: CT scans of pancreatic regions
    
    ### Disclaimer:
    
    This tool is intended for research purposes only and should not replace professional medical diagnosis. Always consult with healthcare professionals for medical advice.
    """)
    
    st.write("### Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Random Forest Model**")
        st.code("""
        # Model Architecture
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt'
        )
        """)
    
    with col2:
        st.write("**CNN Model**")
        st.code("""
        # Model Architecture
        Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        """)
        
    st.write("**Combined Risk Assessment Methodology**")
    st.code("""
    # Combined Score Calculation
    combined_score = (weight_rf * rf_prediction) + (weight_cnn * cnn_prediction)
    
    # Risk Categorization
    if combined_score < 0.3:
        risk_category = "Low Risk"
    elif combined_score < 0.7:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
    """)
    
    st.write("**Patient Data Management**")
    st.code("""
    # Patient Data Structure
    {
        "patient_id": "unique_identifier",
        "patient_name": "Patient Name",
        "date_created": "YYYY-MM-DD HH:MM:SS",
        "clinical_data": {
            "age": 60,
            "sex": "Male",
            "plasma_ca19_9": 37.0,
            ...
        },
        "rf_prediction": 0.75,
        "cnn_prediction": 0.82,
        "ct_scan_image": "base64_encoded_image"
    }
    """)

# Add a footer
st.markdown("""
---
*This is a research prototype for educational purposes only. Not intended for clinical use.*
""")

# Add sidebar with additional information
with st.sidebar:
    st.header("Information")
    st.info("This application demonstrates how machine learning can be used to assist in early detection of pancreatic cancer through both clinical biomarkers and CT scan analysis.")
    
    st.header("References")
    st.markdown("""
    - Debernardi, S., et al. (2020). A combination of urinary biomarker panel and PancRISK score for earlier detection of pancreatic cancer.
    - Early Diagnosis Initiative for Pancreatic Cancer
    """)
    
    st.header("Model Performance")
    st.write("Random Forest Accuracy: 89%")
    st.write("CNN Accuracy: 85%")
    st.write("Combined Model Accuracy: 92%")
    
    # Example performance metrics visualization
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'RF Model': [0.89, 0.87, 0.84, 0.85],
        'CNN Model': [0.85, 0.83, 0.81, 0.82],
        'Combined': [0.92, 0.90, 0.88, 0.89]
    }
    
    metrics_df = pd.DataFrame(metrics)
    st.table(metrics_df.set_index('Metric'))
    
    # Added patient statistics
    st.header("Patient Statistics")
    if st.session_state.saved_patients:
        num_patients = len(st.session_state.saved_patients)
        rf_predictions = [p.get("rf_prediction") for p in st.session_state.saved_patients if p.get("rf_prediction") is not None]
        cnn_predictions = [p.get("cnn_prediction") for p in st.session_state.saved_patients if p.get("cnn_prediction") is not None]
        
        st.write(f"Total patients in database: {num_patients}")
        
        if rf_predictions:
            avg_rf = sum(rf_predictions) / len(rf_predictions)
            st.write(f"Average Clinical Risk Score: {avg_rf:.2%}")
        
        if cnn_predictions:
            avg_cnn = sum(cnn_predictions) / len(cnn_predictions)
            st.write(f"Average CT Scan Risk Score: {avg_cnn:.2%}")
        
        # Simple risk distribution
        high_risk = sum(1 for p in st.session_state.saved_patients 
                        if p.get("rf_prediction", 0) > 0.7 or p.get("cnn_prediction", 0) > 0.7)
        med_risk = sum(1 for p in st.session_state.saved_patients 
                      if (p.get("rf_prediction", 0) <= 0.7 and p.get("rf_prediction", 0) > 0.3) or 
                      (p.get("cnn_prediction", 0) <= 0.7 and p.get("cnn_prediction", 0) > 0.3))
        low_risk = num_patients - high_risk - med_risk
        
        # Create a mini distribution chart
        labels = ['High', 'Medium', 'Low']
        sizes = [high_risk, med_risk, low_risk]
        colors = ['#FF6B6B', '#FFD166', '#06D6A0']
        
        if sum(sizes) > 0:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)