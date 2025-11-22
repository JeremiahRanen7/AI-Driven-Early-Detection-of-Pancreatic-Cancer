import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_image
from lime import lime_tabular
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import datetime
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Pancreatic Cancer Early Detection",
    page_icon="🔬",
    layout="wide"
)

# Create data directory if it doesn't exist
os.makedirs("patient_data", exist_ok=True)

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

# Function to save patient data
def save_patient_data(patient_id, patient_name, clinical_data=None, rf_prediction=None, 
                      ct_image=None, processed_image=None, cnn_prediction=None, 
                      combined_score=None, timestamp=None):
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data to be saved
    patient_data = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "timestamp": timestamp,
        "rf_prediction": float(rf_prediction) if rf_prediction is not None else None,
        "cnn_prediction": float(cnn_prediction) if cnn_prediction is not None else None,
        "combined_score": float(combined_score) if combined_score is not None else None
    }
    
    # Save clinical data if available
    if clinical_data is not None:
        patient_data["clinical_data"] = clinical_data.to_dict('records')[0]
    
    # Save CT image if available
    if ct_image is not None:
        buffered = BytesIO()
        ct_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        patient_data["ct_image"] = img_str
    
    # Save processed image data if available (numpy array)
    if processed_image is not None:
        patient_data["processed_image"] = processed_image.tolist()
    
    # Create the file path
    file_path = f"patient_data/{patient_id}.json"
    
    # Save the data
    with open(file_path, 'w') as f:
        json.dump(patient_data, f)
    
    return file_path

# Function to load patient data
def load_patient_data(patient_id):
    try:
        # Load data from file
        file_path = f"patient_data/{patient_id}.json"
        with open(file_path, 'r') as f:
            patient_data = json.load(f)
        
        # Convert clinical data back to DataFrame if it exists
        if "clinical_data" in patient_data:
            clinical_data = pd.DataFrame([patient_data["clinical_data"]])
        else:
            clinical_data = None
        
        # Convert CT image back to PIL Image if it exists
        if "ct_image" in patient_data:
            img_data = base64.b64decode(patient_data["ct_image"])
            ct_image = Image.open(BytesIO(img_data))
        else:
            ct_image = None
        
        # Convert processed image back to numpy array if it exists
        if "processed_image" in patient_data:
            processed_image = np.array(patient_data["processed_image"])
        else:
            processed_image = None
            
        return {
            "patient_id": patient_data["patient_id"],
            "patient_name": patient_data["patient_name"],
            "timestamp": patient_data["timestamp"],
            "clinical_data": clinical_data,
            "rf_prediction": patient_data["rf_prediction"],
            "ct_image": ct_image,
            "processed_image": processed_image,
            "cnn_prediction": patient_data["cnn_prediction"],
            "combined_score": patient_data["combined_score"]
        }
    
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
        return None

# Function to get list of saved patients
def get_saved_patients():
    patients = []
    try:
        for file in os.listdir("patient_data"):
            if file.endswith(".json"):
                with open(f"patient_data/{file}", 'r') as f:
                    patient_data = json.load(f)
                    patients.append({
                        "patient_id": patient_data["patient_id"],
                        "patient_name": patient_data["patient_name"],
                        "timestamp": patient_data["timestamp"]
                    })
    except Exception as e:
        st.error(f"Error listing patients: {e}")
    
    return patients

# App title and description
st.title("AI-Driven Early Detection of Pancreatic Cancer")
st.markdown("""
This application uses machine learning to assist in the early detection of pancreatic cancer:
- **Clinical Data Analysis**: Uses a Random Forest algorithm to analyze clinical markers
- **CT Scan Analysis**: Uses a Convolutional Neural Network to analyze pancreatic CT scans
- **Combined Analysis**: Integrates both models for a comprehensive risk assessment
- **Explainable AI**: Visualize and understand model decisions with advanced explainability tools
- **Patient Management**: Save and load patient data for follow-up and monitoring

*Note: This tool is for research purposes only and should not replace professional medical advice.*
""")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Patient Management", "Clinical Data Analysis", "CT Scan Analysis", 
                                           "Combined Analysis", "Explainable AI Dashboard", "About"])

# Store prediction results in session state for access across tabs
if 'rf_prediction' not in st.session_state:
    st.session_state.rf_prediction = None
if 'cnn_prediction' not in st.session_state:
    st.session_state.cnn_prediction = None
if 'clinical_data' not in st.session_state:
    st.session_state.clinical_data = None
if 'ct_image' not in st.session_state:
    st.session_state.ct_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = None
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = None
if 'combined_score' not in st.session_state:
    st.session_state.combined_score = None

# NEW: Patient Management Tab
with tab1:
    st.header("Patient Management")
    
    # Create two columns for new patient and loading existing patient
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("New Patient")
        
        # New patient form
        with st.form("new_patient_form"):
            new_patient_id = st.text_input("Patient ID", placeholder="Enter patient ID (e.g., PID12345)")
            new_patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
            
            # Form submit button
            submit_new = st.form_submit_button("Create New Patient Record")
            
            if submit_new:
                if not new_patient_id or not new_patient_name:
                    st.error("Please enter both Patient ID and Name.")
                elif os.path.exists(f"patient_data/{new_patient_id}.json"):
                    st.error(f"Patient ID {new_patient_id} already exists. Please use a different ID.")
                else:
                    # Store new patient info in session state
                    st.session_state.patient_id = new_patient_id
                    st.session_state.patient_name = new_patient_name
                    
                    # Reset other session state variables for the new patient
                    st.session_state.rf_prediction = None
                    st.session_state.cnn_prediction = None
                    st.session_state.clinical_data = None
                    st.session_state.ct_image = None
                    st.session_state.processed_image = None
                    st.session_state.feature_values = None
                    st.session_state.combined_score = None
                    
                    # Create initial patient record
                    save_patient_data(
                        patient_id=new_patient_id,
                        patient_name=new_patient_name
                    )
                    
                    st.success(f"New patient record created for {new_patient_name} (ID: {new_patient_id})")
                    st.info("Proceed to Clinical Data Analysis or CT Scan Analysis tabs to add patient data.")
    
    with col2:
        st.subheader("Load Existing Patient")
        
        # Get list of saved patients
        saved_patients = get_saved_patients()
        
        if not saved_patients:
            st.info("No saved patients found.")
        else:
            # Create a DataFrame for better display
            patients_df = pd.DataFrame(saved_patients)
            
            # Add a selectbox to choose patient
            patient_options = [f"{p['patient_name']} (ID: {p['patient_id']}, Date: {p['timestamp']})" for p in saved_patients]
            selected_patient_idx = st.selectbox("Select Patient", range(len(patient_options)), format_func=lambda x: patient_options[x])
            
            selected_patient = saved_patients[selected_patient_idx]
            
            # Show load button
            if st.button("Load Patient Data"):
                # Load the selected patient data
                patient_data = load_patient_data(selected_patient["patient_id"])
                
                if patient_data:
                    # Update session state with loaded data
                    st.session_state.patient_id = patient_data["patient_id"]
                    st.session_state.patient_name = patient_data["patient_name"]
                    st.session_state.clinical_data = patient_data["clinical_data"]
                    st.session_state.rf_prediction = patient_data["rf_prediction"]
                    st.session_state.ct_image = patient_data["ct_image"]
                    st.session_state.processed_image = patient_data["processed_image"]
                    st.session_state.cnn_prediction = patient_data["cnn_prediction"]
                    st.session_state.combined_score = patient_data["combined_score"]
                    
                    # If clinical data exists, store feature values as well
                    if patient_data["clinical_data"] is not None:
                        st.session_state.feature_values = patient_data["clinical_data"].iloc[0].to_dict()
                    
                    st.success(f"Loaded patient data for {patient_data['patient_name']} (ID: {patient_data['patient_id']})")
                    
                    # Display summary of loaded data
                    st.subheader("Patient Summary")
                    
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.write(f"**Patient ID:** {patient_data['patient_id']}")
                        st.write(f"**Patient Name:** {patient_data['patient_name']}")
                        st.write(f"**Last Updated:** {patient_data['timestamp']}")
                    
                    with summary_col2:
                        if patient_data['rf_prediction'] is not None:
                            st.write(f"**Clinical Analysis:** {'Complete' if patient_data['rf_prediction'] is not None else 'Not performed'}")
                            st.write(f"**CT Scan Analysis:** {'Complete' if patient_data['cnn_prediction'] is not None else 'Not performed'}")
                            st.write(f"**Combined Score:** {patient_data['combined_score']:.2%}" if patient_data['combined_score'] else "Not calculated")
    
    # Display current patient information
    st.divider()
    
    if st.session_state.patient_id and st.session_state.patient_name:
        st.subheader("Current Patient")
        
        current_col1, current_col2, current_col3 = st.columns(3)
        
        with current_col1:
            st.info(f"**Patient ID:** {st.session_state.patient_id}")
            st.info(f"**Patient Name:** {st.session_state.patient_name}")
        
        with current_col2:
            # Show analysis status
            clinical_status = "✅ Complete" if st.session_state.rf_prediction is not None else "❌ Not performed"
            ct_status = "✅ Complete" if st.session_state.cnn_prediction is not None else "❌ Not performed"
            
            st.info(f"**Clinical Analysis:** {clinical_status}")
            st.info(f"**CT Scan Analysis:** {ct_status}")
        
        with current_col3:
            # Add button to save current state
            if st.button("Save Current Patient Data"):
                # Check if there's any data to save
                if (st.session_state.rf_prediction is not None or 
                    st.session_state.cnn_prediction is not None):
                    
                    # Save the current state
                    save_patient_data(
                        patient_id=st.session_state.patient_id,
                        patient_name=st.session_state.patient_name,
                        clinical_data=st.session_state.clinical_data,
                        rf_prediction=st.session_state.rf_prediction,
                        ct_image=st.session_state.ct_image,
                        processed_image=st.session_state.processed_image,
                        cnn_prediction=st.session_state.cnn_prediction,
                        combined_score=st.session_state.combined_score
                    )
                    
                    st.success(f"Data saved for patient {st.session_state.patient_name} (ID: {st.session_state.patient_id})")
                else:
                    st.warning("No analysis data to save. Please complete at least one analysis first.")
            
            # Add button to generate report
            if st.button("Generate Patient Report"):
                if st.session_state.rf_prediction is not None or st.session_state.cnn_prediction is not None:
                    st.info("Generating patient report... Feature will be available in the next update.")
                else:
                    st.warning("No analysis data available. Please complete at least one analysis first.")
    else:
        st.info("No patient selected. Please create a new patient record or load an existing one.")

# Clinical Data Analysis Tab
with tab2:
    st.header("Clinical Data Analysis")
    
    # Check if a patient is selected
    if not st.session_state.patient_id or not st.session_state.patient_name:
        st.warning("Please select or create a patient record in the Patient Management tab first.")
    else:
        st.write(f"Enter clinical data for patient: **{st.session_state.patient_name}** (ID: **{st.session_state.patient_id}**)")
        
        # Pre-fill form with existing data if available
        pre_filled = st.session_state.feature_values if st.session_state.feature_values else {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, 
                                 value=int(pre_filled.get('age', 60)))
            
            sex = st.selectbox("Sex", options=["Male", "Female"],
                              index=0 if pre_filled.get('sex', 1) == 1 else 1)
            
            plasma_ca19_9 = st.number_input("Plasma CA19-9 (U/mL)", min_value=0.0, 
                                          value=float(pre_filled.get('plasma_CA19_9', 37.0)))
            
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, 
                                        value=float(pre_filled.get('creatinine', 1.0)))
        
        with col2:
            lyve1 = st.number_input("LYVE1 (ng/mL)", min_value=0.0, 
                                   value=float(pre_filled.get('LYVE1', 1.0)))
            
            reg1b = st.number_input("REG1B (ng/mL)", min_value=0.0, 
                                   value=float(pre_filled.get('REG1B', 1.0)))
            
            tff1 = st.number_input("TFF1 (ng/mL)", min_value=0.0, 
                                  value=float(pre_filled.get('TFF1', 1.0)))
            
            reg1a = st.number_input("REG1A (ng/mL)", min_value=0.0, 
                                   value=float(pre_filled.get('REG1A', 1.0)))
        
        # Default values for categorical variables
        default_cohort = "Cohort2" if pre_filled.get('patient_cohort_Cohort2', 0) == 1 else "Cohort1"
        
        # Determine default sample origin
        default_origin = "Other"
        for origin in ["ESP", "LIV", "UCL"]:
            if pre_filled.get(f'sample_origin_{origin}', 0) == 1:
                default_origin = origin
                break
        
        patient_cohort = st.selectbox("Patient Cohort", options=["Cohort1", "Cohort2"], 
                                     index=["Cohort1", "Cohort2"].index(default_cohort))
        
        sample_origin = st.selectbox("Sample Origin", options=["Other", "ESP", "LIV", "UCL"],
                                    index=["Other", "ESP", "LIV", "UCL"].index(default_origin))
        
        if st.button("Generate Clinical Analysis", key="clinical_button"):
            # Prepare input data for RF model
            clinical_data = {
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
            input_df = pd.DataFrame([clinical_data])
            
            # Store feature values in session state
            st.session_state.feature_values = clinical_data
            st.session_state.clinical_data = input_df
            
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
                
                # Save patient data automatically
                save_patient_data(
                    patient_id=st.session_state.patient_id,
                    patient_name=st.session_state.patient_name,
                    clinical_data=st.session_state.clinical_data,
                    rf_prediction=st.session_state.rf_prediction,
                    ct_image=st.session_state.ct_image,
                    processed_image=st.session_state.processed_image,
                    cnn_prediction=st.session_state.cnn_prediction,
                    combined_score=st.session_state.combined_score
                )
                
                st.success("✓ Analysis complete! Results have been saved to patient record.")
                st.info("Visit the 'Explainable AI Dashboard' tab for in-depth insights or 'Combined Analysis' if you've also performed CT Scan Analysis.")
            else:
                st.error("Random Forest model not loaded properly. Please check your model file.")

# CT Scan Analysis Tab
with tab3:
    st.header("CT Scan Analysis")
    
    # Check if a patient is selected
    if not st.session_state.patient_id or not st.session_state.patient_name:
        st.warning("Please select or create a patient record in the Patient Management tab first.")
    else:
        st.write(f"Upload CT scan for patient: **{st.session_state.patient_name}** (ID: **{st.session_state.patient_id}**)")
        
        # Display existing CT scan if available
        if st.session_state.ct_image is not None:
            st.write("Current CT scan on record:")
            st.image(st.session_state.ct_image, caption="Current CT Scan", width=300)
            
            if st.button("Clear existing CT scan", key="clear_ct"):
                st.session_state.ct_image = None
                st.session_state.processed_image = None
                st.session_state.cnn_prediction = None
                st.experimental_rerun()
        
        uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            st.image(image, caption="Uploaded CT Scan", width=300)
            
            # Store original image in session state
            st.session_state.ct_image = image
            
            if st.button("Analyze CT Scan", key="ct_button"):
                if cnn_model is not None:
                    # Preprocess the image
                    img = image.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
                    
                    # Store processed image in session state
                    st.session_state.processed_image = img_array
                    
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
                    st.write("This visualization attempts to show areas of the CT scan that influenced the model's decision.")
                    
                    # Basic implementation of attention visualization
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(img_array[0, :, :, 0], cmap='gray')
                    ax.set_title('CT Scan')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Save patient data automatically
                    save_patient_data(
                        patient_id=st.session_state.patient_id,
                        patient_name=st.session_state.patient_name,
                        clinical_data=st.session_state.clinical_data,
                        rf_prediction=st.session_state.rf_prediction,
                        ct_image=st.session_state.ct_image,
                        processed_image=st.session_state.processed_image,
                        cnn_prediction=st.session_state.cnn_prediction,
                        combined_score=st.session_state.combined_score
                    )
                    
                    st.success("✓ Analysis complete! Results have been saved to patient record.")
                    st.info("Visit the 'Explainable AI Dashboard' tab for advanced visualization of CNN attention maps or 'Combined Analysis' if you've also performed Clinical Data Analysis.")
                else:
                    st.error("CNN model not loaded properly. Please check your model file.")

# Combined Analysis Tab
with tab4:
    st.header("Combined Risk Assessment")
    
    # Check if a patient is selected
    if not st.session_state.patient_id or not st.session_state.patient_name:
        st.warning("Please select or create a patient record in the Patient Management tab first.")
    else:
        st.write(f"Combined analysis for patient: **{st.session_state.patient_name}** (ID: **{st.session_state.patient_id}**)")
        
        # Check if both predictions are available
        if st.session_state.rf_prediction is not None and st.session_state.cnn_prediction is not None:
            # Extract predictions from session state
            rf_score = st.session_state.rf_prediction
            cnn_score = st.session_state.cnn_prediction
            
            # Calculate combined score (weighted average)
            # You can adjust weights based on the relative importance/confidence of each model
            weight_rf = st.slider("Clinical Data Model Weight", 0.0, 1.0, 0.5, 0.05)
            weight_cnn = 1 - weight_rf
            
            combined_score = (weight_rf * rf_score) + (weight_cnn * cnn_score)
            
            # Store combined score in session state
            st.session_state.combined_score = combined_score
            
            # Define risk categories based on combined score
            if combined_score < 0.3:
                risk_category = "Low Risk"
                risk_color = "green"
            elif combined_score < 0.7:
                risk_category = "Moderate Risk"
                risk_color = "orange"
            else:
                risk_category = "High Risk"
                risk_color = "red"
            
            # Display combined results
            st.subheader("Combined Risk Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Clinical Data Score", f"{rf_score:.2%}")
                st.metric("CT Scan Score", f"{cnn_score:.2%}")
                st.metric("Combined Risk Score", f"{combined_score:.2%}")
                st.markdown(f"<h3 style='color:{risk_color}'>{risk_category}</h3>", unsafe_allow_html=True)
            
            with col2:
                # Create comparison chart
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Bar chart comparing scores
                models = ['Clinical Data Model', 'Imaging Model', 'Combined Score']
                scores = [rf_score, cnn_score, combined_score]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                bars = ax.bar(models, scores, color=colors)
                
                # Add threshold line for positive classification
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                ax.text(0.1, 0.51, 'Decision Threshold', color='r', alpha=0.7)
                
                # Add labels and formatting
                ax.set_ylim(0, 1)
                ax.set_ylabel('Risk Score')
                ax.set_title('Model Risk Score Comparison')
                
                # Continuing from where the code was cut off

                # Add value labels on the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2%}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            # Risk assessment details
            st.subheader("Risk Assessment Details")
            
            # Create detailed results table
            details_data = {
                "Analysis Type": ["Clinical Data Analysis", "CT Scan Analysis", "Combined Assessment"],
                "Risk Score": [f"{rf_score:.2%}", f"{cnn_score:.2%}", f"{combined_score:.2%}"],
                "Interpretation": [
                    "Low Risk" if rf_score < 0.3 else "Moderate Risk" if rf_score < 0.7 else "High Risk",
                    "Low Risk" if cnn_score < 0.3 else "Moderate Risk" if cnn_score < 0.7 else "High Risk",
                    risk_category
                ]
            }
            
            details_df = pd.DataFrame(details_data)
            st.table(details_df)
            
            # Clinical recommendations based on risk level
            st.subheader("Clinical Recommendations")
            
            if combined_score < 0.3:
                st.write("""
                **Low Risk Assessment**
                - Consider routine follow-up according to standard guidelines
                - Monitor for any changes in symptoms or clinical markers
                - Review clinical risk factors and provide lifestyle guidance
                """)
            elif combined_score < 0.7:
                st.write("""
                **Moderate Risk Assessment**
                - Consider additional diagnostic tests for further evaluation
                - Shorter interval follow-up may be appropriate
                - Further investigation of specific biomarkers with abnormal values
                - Consider endoscopic ultrasound (EUS) for more detailed pancreatic imaging
                """)
            else:
                st.write("""
                **High Risk Assessment**
                - Urgent referral to specialist is recommended
                - Consider comprehensive diagnostic workup including:
                  * Endoscopic ultrasound (EUS)
                  * Magnetic resonance imaging (MRI)
                  * Endoscopic retrograde cholangiopancreatography (ERCP)
                - Consider biopsy for definitive diagnosis
                - Close monitoring and follow-up required
                """)
            
            # Save the combined assessment
            if st.button("Save Combined Assessment"):
                save_patient_data(
                    patient_id=st.session_state.patient_id,
                    patient_name=st.session_state.patient_name,
                    clinical_data=st.session_state.clinical_data,
                    rf_prediction=st.session_state.rf_prediction,
                    ct_image=st.session_state.ct_image,
                    processed_image=st.session_state.processed_image,
                    cnn_prediction=st.session_state.cnn_prediction,
                    combined_score=combined_score
                )
                
                st.success("Combined assessment saved successfully!")
        
        elif st.session_state.rf_prediction is not None or st.session_state.cnn_prediction is not None:
            st.warning("Only one analysis method has been completed. Please complete both Clinical Data Analysis and CT Scan Analysis to generate a combined risk assessment.")
            
            # Show which analysis is missing
            if st.session_state.rf_prediction is None:
                st.info("Missing: Clinical Data Analysis - Please complete this analysis in the 'Clinical Data Analysis' tab.")
            
            if st.session_state.cnn_prediction is None:
                st.info("Missing: CT Scan Analysis - Please complete this analysis in the 'CT Scan Analysis' tab.")
        
        else:
            st.info("No analysis data available. Please complete both Clinical Data Analysis and CT Scan Analysis first.")

# Explainable AI Dashboard Tab
with tab5:
    st.header("Explainable AI Dashboard")
    
    # Check if a patient is selected
    if not st.session_state.patient_id or not st.session_state.patient_name:
        st.warning("Please select or create a patient record in the Patient Management tab first.")
    else:
        st.write(f"Explainable AI for patient: **{st.session_state.patient_name}** (ID: **{st.session_state.patient_id}**)")
        
        # Create tabs for different explainability methods
        xai_tab1, xai_tab2, xai_tab3 = st.tabs(["Clinical Data XAI", "CT Scan XAI", "XAI Settings"])
        
        # Clinical Data XAI Tab
        with xai_tab1:
            if st.session_state.rf_prediction is not None and st.session_state.clinical_data is not None:
                st.subheader("Clinical Data Model Explanation")
                
                # Choose XAI method
                xai_method = st.selectbox(
                    "Select Explainability Method",
                    ["SHAP Values", "Feature Importance", "Permutation Importance", "Decision Path"],
                    index=0,
                    key="rf_xai_method"
                )
                
                if xai_method == "SHAP Values" and rf_model is not None:
                    st.write("SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the prediction.")
                    
                    # Create SHAP explainer and values
                    try:
                        # Use a sample of the training data as background
                        # In production, you would have this background dataset ready
                        background_data = st.session_state.clinical_data.sample(min(10, len(st.session_state.clinical_data)))
                        
                        explainer = shap.TreeExplainer(rf_model)
                        shap_values = explainer.shap_values(st.session_state.clinical_data)
                        
                        # SHAP force plot
                        st.write("### SHAP Force Plot")
                        st.write("This plot shows how each feature contributes to pushing the model output from the base value to the final prediction.")
                        
                        # Convert SHAP values to plots
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.summary_plot(shap_values[1], st.session_state.clinical_data, plot_type="bar", show=False)
                        st.pyplot(fig)
                        
                        # SHAP waterfall plot
                        st.write("### SHAP Waterfall Plot")
                        st.write("This plot shows the path from the base value to the model's output for this specific patient.")
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.waterfall_plot(explainer.expected_value[1], shap_values[1][0], feature_names=st.session_state.clinical_data.columns, show=False)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {e}")
                        st.info("SHAP analysis may require additional data that is not available in this demo version.")
                
                elif xai_method == "Feature Importance" and rf_model is not None:
                    st.write("Feature importance shows which features were most influential in the overall model.")
                    
                    if hasattr(rf_model, 'feature_importances_'):
                        # Get feature importances
                        feature_importance = pd.DataFrame({
                            'Feature': st.session_state.clinical_data.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importances
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
                        ax.set_title('Feature Importance in Random Forest Model')
                        st.pyplot(fig)
                        
                        # Feature importance table
                        st.write("### Feature Importance Values")
                        st.table(feature_importance)
                    else:
                        st.warning("This model doesn't provide built-in feature importance.")
                
                elif xai_method == "Permutation Importance" and rf_model is not None:
                    st.write("Permutation importance measures the decrease in model performance when a feature is randomly shuffled.")
                    st.info("This would require a reference dataset with known outcomes, which is not available in this demo.")
                    
                    # In a real implementation, you would do something like:
                    # perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10)
                    
                    # Generate some placeholder permutation importance for demonstration
                    features = st.session_state.clinical_data.columns
                    perm_scores = np.random.uniform(0, 0.3, size=len(features))
                    perm_std = np.random.uniform(0, 0.05, size=len(features))
                    
                    perm_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': perm_scores,
                        'Std Dev': perm_std
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot permutation importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=perm_df, palette='viridis', ax=ax)
                    ax.set_title('Permutation Feature Importance (Demonstration)')
                    st.pyplot(fig)
                    
                    st.write("### Permutation Importance Values")
                    st.write("Note: These values are simulated for demonstration purposes.")
                    st.table(perm_df)
                
                elif xai_method == "Decision Path" and rf_model is not None:
                    st.write("Decision path visualizes the path through the decision trees for this specific prediction.")
                    
                    # For demonstration, show a simplified decision path
                    st.info("A full decision path visualization requires complex tree extraction, which is simplified in this demo.")
                    
                    # Create a simplified decision path visualization
                    features = st.session_state.clinical_data.columns
                    feature_values = st.session_state.feature_values
                    
                    # Create a dummy decision path
                    decision_steps = [
                        {"feature": "plasma_CA19_9", "threshold": 40.0, "decision": ">" if feature_values["plasma_CA19_9"] > 40.0 else "≤"},
                        {"feature": "age", "threshold": 65, "decision": ">" if feature_values["age"] > 65 else "≤"},
                        {"feature": "REG1B", "threshold": 1.2, "decision": ">" if feature_values["REG1B"] > 1.2 else "≤"}
                    ]
                    
                    # Display decision path
                    st.write("### Simplified Decision Path")
                    st.write("This is a simplified representation of how the decision was made:")
                    
                    for i, step in enumerate(decision_steps):
                        feature = step["feature"]
                        threshold = step["threshold"]
                        decision = step["decision"]
                        value = feature_values.get(feature, "N/A")
                        
                        st.write(f"**Step {i+1}:** Is {feature} ({value}) {decision} {threshold}? → {'Yes' if decision == '>' and value > threshold or decision == '≤' and value <= threshold else 'No'}")
                    
                    st.write(f"**Final prediction:** {st.session_state.rf_prediction:.2%} probability of pancreatic cancer")
                    
                    # Add a note about the simplification
                    st.info("Note: In a full implementation, this would show the actual decision paths through the random forest trees.")
            else:
                st.info("Clinical data analysis has not been performed yet. Please complete the Clinical Data Analysis first.")
        
        # CT Scan XAI Tab
        with xai_tab2:
            if st.session_state.cnn_prediction is not None and st.session_state.processed_image is not None:
                st.subheader("CT Scan Model Explanation")
                
                # Choose XAI method
                xai_method = st.selectbox(
                    "Select Explainability Method",
                    ["Grad-CAM", "LIME", "Integrated Gradients", "Occlusion Sensitivity"],
                    index=0,
                    key="cnn_xai_method"
                )
                
                # Display original image
                st.write("### Original CT Scan")
                if st.session_state.ct_image is not None:
                    st.image(st.session_state.ct_image, width=300)
                
                if xai_method == "Grad-CAM" and cnn_model is not None:
                    st.write("Grad-CAM visualizes which regions of the image influenced the model's decision.")
                    
                    # In a real implementation, you would compute Grad-CAM
                    # For demonstration, we'll just show a simulated heat map
                    st.info("In this demo, we're showing a simulated Grad-CAM visualization.")
                    
                    # Create a simulated Grad-CAM overlay
                    img_array = st.session_state.processed_image[0, :, :, 0]
                    
                    # Generate a fake heatmap centered near the pancreas region
                    h, w = img_array.shape
                    y, x = np.ogrid[:h, :w]
                    center_y, center_x = h * 0.6, w * 0.4  # Approximate pancreas location
                    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2. * 30**2))
                    
                    # Plot the heatmap overlay
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_array, cmap='gray')
                    ax.imshow(heatmap, cmap='jet', alpha=0.5)
                    ax.set_title('Grad-CAM: Regions of Interest (Simulated)')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.write("""
                    #### Interpretation
                    The heat map highlights regions that most influenced the model's prediction:
                    - Red areas indicate regions that strongly contributed to the cancer prediction
                    - Blue areas had less influence on the prediction
                    """)
                
                elif xai_method == "LIME" and cnn_model is not None:
                    st.write("LIME (Local Interpretable Model-agnostic Explanations) explains the prediction by perturbing the input.")
                    
                    # In a real implementation, you would use LIME
                    # For demonstration, show a simulated LIME result
                    st.info("In this demo, we're showing a simulated LIME visualization.")
                    
                    # Create a simulated LIME segmentation
                    img_array = st.session_state.processed_image[0, :, :, 0]
                    
                    # Generate random segments
                    segments = np.zeros_like(img_array, dtype=int)
                    h, w = img_array.shape
                    num_segments = 10
                    
                    for i in range(num_segments):
                        center_y = np.random.randint(0, h)
                        center_x = np.random.randint(0, w)
                        radius = np.random.randint(10, 30)
                        y, x = np.ogrid[:h, :w]
                        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                        segments[mask] = i + 1
                    
                    # Assign importance scores to segments
                    segment_importance = np.random.uniform(-1, 1, num_segments + 1)
                    segment_importance[0] = 0  # Background
                    
                    # Create LIME visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original image
                    ax1.imshow(img_array, cmap='gray')
                    ax1.set_title('Original CT Scan')
                    ax1.axis('off')
                    
                    # LIME explanation
                    colored_segments = np.zeros((h, w, 3))
                    for i in range(num_segments + 1):
                        if segment_importance[i] > 0:
                            # Positive contribution - red
                            colored_segments[segments == i] = [segment_importance[i], 0, 0]
                        else:
                            # Negative contribution - blue
                            colored_segments[segments == i] = [0, 0, -segment_importance[i]]
                    
                    ax2.imshow(img_array, cmap='gray')
                    ax2.imshow(colored_segments, alpha=0.7)
                    ax2.set_title('LIME Explanation (Simulated)')
                    ax2.axis('off')
                    
                    st.pyplot(fig)
                    
                    st.write("""
                    #### Interpretation
                    - Red segments positively contribute to cancer prediction
                    - Blue segments negatively contribute to cancer prediction
                    - The intensity of the color indicates the strength of the contribution
                    """)
                
                elif xai_method == "Integrated Gradients" and cnn_model is not None:
                    st.write("Integrated Gradients attributes the prediction to input features by calculating gradients along a path.")
                    
                    # Simulated integrated gradients visualization
                    st.info("In this demo, we're showing a simulated Integrated Gradients visualization.")
                    
                    img_array = st.session_state.processed_image[0, :, :, 0]
                    
                    # Generate a fake attribution map
                    h, w = img_array.shape
                    y, x = np.ogrid[:h, :w]
                    center_y, center_x = h * 0.6, w * 0.4  # Approximate pancreas location
                    attribution = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2. * 20**2))
                    # Add some noise to make it look more realistic
                    attribution += np.random.normal(0, 0.1, size=attribution.shape)
                    attribution = np.clip(attribution, 0, 1)
                    
                    # Plot the attribution map
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_array, cmap='gray')
                    ax.imshow(attribution, cmap='hot', alpha=0.5)
                    ax.set_title('Integrated Gradients: Attribution Map (Simulated)')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.write("""
                    #### Interpretation
                    The attribution map shows how each pixel contributes to the prediction:
                    - Brighter areas have more impact on the model's decision
                    - This helps identify which specific regions of the pancreas the model is focusing on
                    """)
                
                elif xai_method == "Occlusion Sensitivity" and cnn_model is not None:
                    st.write("Occlusion Sensitivity measures how the prediction changes when parts of the image are masked.")
                    
                    # Simulated occlusion sensitivity map
                    st.info("In this demo, we're showing a simulated Occlusion Sensitivity visualization.")
                    
                    img_array = st.session_state.processed_image[0, :, :, 0]
                    
                    # Create a grid to represent sensitivity
                    grid_size = 16
                    h, w = img_array.shape
                    cell_h, cell_w = h // grid_size, w // grid_size
                    
                    # Generate random sensitivity values
                    sensitivity = np.random.uniform(0, 1, size=(grid_size, grid_size))
                    # Make it more realistic with some structure
                    y, x = np.ogrid[:grid_size, :grid_size]
                    center_y, center_x = grid_size * 0.6, grid_size * 0.4
                    structure = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2. * (grid_size/5)**2))
                    sensitivity = sensitivity * 0.2 + structure * 0.8
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_array, cmap='gray')
                    
                    # Draw grid with colors based on sensitivity
                    for i in range(grid_size):
                        for j in range(grid_size):
                            rect = plt.Rectangle((j*cell_w, i*cell_h), cell_w, cell_h, 
                                                fill=True, alpha=sensitivity[i, j] * 0.7,
                                                facecolor='red')
                            ax.add_patch(rect)
                    
                    ax.set_title('Occlusion Sensitivity Map (Simulated)')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.write("""
                    #### Interpretation
                    The occlusion sensitivity map shows how occluding different parts of the image affects the prediction:
                    - Brighter red regions indicate areas where occlusion causes a larger drop in the cancer prediction score
                    - These are the regions the model considers most important for its decision
                    """)
            else:
                st.info("CT scan analysis has not been performed yet. Please complete the CT Scan Analysis first.")
        
        # XAI Settings Tab
        with xai_tab3:
            st.subheader("Explainable AI Settings")
            
            st.write("""
            This tab allows you to configure how the explainable AI components work. In a production system, 
            you would be able to adjust parameters such as:
            
            1. **Visualization parameters**
                - Color maps for heatmaps
                - Threshold levels for significance
                - Segmentation granularity for LIME
            
            2. **Computation settings**
                - Number of samples for SHAP
                - Baseline reference points
                - Integration steps for integrated gradients
            
            3. **Export options**
                - Export explanations as images
                - Include explanations in patient reports
                - Save raw explanation data for further analysis
            """)
            
            st.info("These settings are not functional in this demo version. In a production system, they would allow fine-tuning of the explainability methods.")

# About Tab
with tab6:
    st.header("About This Application")
    
    st.markdown("""
    ### AI-Driven Early Detection of Pancreatic Cancer
    
    This application demonstrates how artificial intelligence can assist in the early detection of pancreatic cancer by analyzing multiple data sources and providing explainable results.
    
    #### Key Features:
    
    - **Multi-modal Analysis**: Combines clinical biomarkers with imaging data for more comprehensive assessment
    - **Advanced AI Models**: Uses Random Forest for clinical data and Convolutional Neural Networks for CT scan analysis
    - **Explainable AI**: Provides transparency into model decisions through various visualization techniques
    - **Patient Management**: Allows tracking of patients over time for longitudinal assessment
    - **User-friendly Interface**: Designed for clinical use with intuitive workflows
    
    #### Technical Details:
    
    - **Clinical Data Model**: Random Forest algorithm trained on clinical biomarkers including CA19-9, LYVE1, REG1B, TFF1, and REG1A
    - **Imaging Model**: Convolutional Neural Network trained on pancreatic CT scans
    - **Explainability Methods**: SHAP, LIME, Grad-CAM, Integrated Gradients, and more
    - **Built with**: Python, TensorFlow, scikit-learn, Streamlit
    
    #### Medical Disclaimer:
    
    This application is designed for research and educational purposes only. It should not be used as the sole basis for clinical decision-making without appropriate validation and medical expertise. Always consult with qualified healthcare professionals for diagnosis and treatment decisions.
    
    #### Acknowledgements:
    
    - Based on research in AI applications for pancreatic cancer detection
    - Uses open-source libraries for machine learning and visualization
    - Interface designed with clinical workflows in mind
    
    #### Privacy Note:
    
    All patient data used in this application is stored locally and not transmitted to external servers. We recommend using anonymized patient identifiers in accordance with healthcare privacy regulations.
    """)
    
    st.write("© 2025 AI-Driven Pancreatic Cancer Detection - Version 1.0")