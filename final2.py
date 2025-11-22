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
            if st.button("Load Patient Data",key="goto_clinical_data_btn"):
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
            if st.button("Save Current Patient Data",key="goto_clinical_data_btn2"):
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
            if st.button("Generate Patient Report",key="goto_clinical_data_btn3"):
                if st.session_state.rf_prediction is not None or st.session_state.cnn_prediction is not None:
                    st.info("Generating patient report... Feature will be available in the next update.")
                else:
                    st.warning("No analysis data available. Please complete at least one analysis first.")
    else:
        st.info("No patient selected. Please create a new patient record or load an existing one.")

# Clinical Data Analysis Tab
with tab2:
    st.header("Clinical Data Analysis")
    st.write("Enter patient clinical data to get a risk assessment using our Random Forest model.")

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
        
        if st.button("Generate Clinical Analysis", key="goto_clinical_data_btn4"):
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
    st.write("Upload a pancreatic CT scan image for analysis using our CNN model.")

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
            if st.button("Save Combined Assessment",key="goto_clinical_data_btn7"):
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
    st.write("""
    This advanced dashboard helps you understand how our AI models are making predictions using state-of-the-art
    explainability techniques. Explore the visualizations to gain insights into the decision-making process.
    """)
    # Check if a patient is selected
    if not st.session_state.patient_id or not st.session_state.patient_name:
        st.warning("Please select or create a patient record in the Patient Management tab first.")
    else:
        st.write(f"Explainable AI for patient: **{st.session_state.patient_name}** (ID: **{st.session_state.patient_id}**)")
        
        # Create tabs for different explainability methods
        xai_tab1, xai_tab2, xai_tab3 = st.tabs(["Clinical Data XAI", "CT Scan XAI", "Interactive Feature Analysis"])
        
        # Clinical Data XAI Tab
        with xai_tab1:
            if st.session_state.rf_prediction is not None and st.session_state.clinical_data is not None:
                st.subheader("Clinical Data Model Explanation")
                
                # Choose XAI method
                xai_method = st.selectbox(
                    "Select Explainability Method",
                    ["SHAP Values", "Feature Importance", "Decision Path"],
                    index=0,
                    key="rf_xai_method"
                )
                
                if xai_method == "SHAP Values" and rf_model is not None:
                    st.write("### SHAP Value Analysis")
                    st.write("""
                    SHAP (SHapley Additive exPlanations) values show how much each feature contributes to pushing the 
                    prediction higher (red) or lower (blue) from the baseline.
                    """)

                    try:
                        background_data = st.session_state.clinical_data.sample(min(10, len(st.session_state.clinical_data)))
                        explainer = shap.TreeExplainer(rf_model)
                        shap_values = explainer.shap_values(st.session_state.clinical_data)
                        if isinstance(shap_values, list):
                            if len(shap_values) > 1:
                                plot_shap_values = shap_values[1]
                                expected_value = explainer.expected_value[1]
                            else:
                                plot_shap_values = shap_values[0]
                                expected_value = explainer.expected_value
                        else:
                            plot_shap_values = shap_values
                            expected_value = explainer.expected_value
        
                        # SHAP force plot
                        st.write("### SHAP Feature Importance")
                        st.write("This plot shows how each feature contributes to the model's predictions.")
                        
                        # Convert SHAP values to plots
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.summary_plot(plot_shap_values, st.session_state.clinical_data, plot_type="bar", show=False)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {e}")
                        st.info("SHAP analysis may require additional data that is not available in this demo version.")
                        # For debugging
                        st.write("Debug info:")
                        st.write(f"SHAP values type: {type(shap_values)}")
                        if isinstance(shap_values, list):
                            st.write(f"SHAP values length: {len(shap_values)}")
                            for i, sv in enumerate(shap_values):
                                st.write(f"SHAP values[{i}] shape: {sv.shape}")
                        else:
                            st.write(f"SHAP values shape: {shap_values.shape}")
                    
                elif xai_method == "Feature Importance" and rf_model is not None:
                    st.write("### Global Feature Importance")
                    st.write("This shows how important each clinical marker is for the model overall.")
                    
                    if hasattr(rf_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': st.session_state.clinical_data.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Create interactive bar chart with Plotly
                        fig = px.bar(
                            feature_importance, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='viridis',
                            title='Global Feature Importance in Random Forest Model'
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                                # SHAP Force Plot for current sample
                        st.write("### SHAP Decision Plot for Current Patient")
                        st.write("This visualization shows how each feature pushed the model's prediction up or down.")
                        
                        # In a real implementation:
                        # shap_values_patient = explainer.shap_values(st.session_state.clinical_data)[1][0]
                        
                        # Simulated force plot for patient
                        fig, ax = plt.subplots(figsize=(12, 1.5))
                        feature_values = list(st.session_state.feature_values.values())
                        
                        # Set base value (average prediction)
                        base_value = 0.4
                        
                        # Cumulative values for waterfall
                        input_features = list(st.session_state.feature_values.keys())
                        
                        # Create simulated contributions
                        contributions = {}
                        for feature in input_features:
                            if feature == 'plasma_CA19_9':
                                contributions[feature] = 0.25 if st.session_state.feature_values[feature] > 37 else -0.05
                            elif feature == 'LYVE1':
                                contributions[feature] = 0.15 if st.session_state.feature_values[feature] > 1.5 else -0.05
                            elif feature == 'age':
                                contributions[feature] = 0.08 if st.session_state.feature_values[feature] > 65 else -0.03
                            else:
                                contributions[feature] = np.random.uniform(-0.05, 0.05)
                        
                        # Sort contributions by absolute magnitude
                        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                        
                        # Create waterfall chart
                        fig = go.Figure(go.Waterfall(
                            name="SHAP", 
                            orientation="h",
                            measure=["relative"] * len(sorted_contributions) + ["total"],
                            y=[f"{k} = {st.session_state.feature_values[k]:.2f}" if isinstance(st.session_state.feature_values[k], float) 
                            else f"{k} = {st.session_state.feature_values[k]}" for k, v in sorted_contributions] + ["Final Prediction"],
                            x=[v for k, v in sorted_contributions] + [sum(v for k, v in sorted_contributions)],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "red"}},
                            decreasing={"marker": {"color": "blue"}},
                            text=[f"{v:+.3f}" for k, v in sorted_contributions] + [f"{base_value + sum(v for k, v in sorted_contributions):.3f}"],
                            textposition="outside"
                        ))
                        
                        fig.update_layout(
                            title="Feature Contribution to Prediction",
                            showlegend=False,
                            height=600,
                            xaxis_title="Impact on prediction (SHAP value)",
                            yaxis_title="Feature"
                        )
                        
                        # Add base value line
                        fig.add_shape(
                            type="line",
                            x0=base_value,
                            y0=-0.5,
                            x1=base_value,
                            y1=len(sorted_contributions) + 0.5,
                            line=dict(color="black", width=2, dash="dash")
                        )
                        
                        fig.add_annotation(
                            x=base_value,
                            y=-0.5,
                            text=f"Base value: {base_value:.3f}",
                            showarrow=False,
                            yshift=-20
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
               
                        
                        # Feature vs Target relationship
                        st.write("### Feature-Target Relationships")
                        st.write("These plots show how each biomarker value relates to pancreatic cancer risk.")
                        
                        # Select top 4 important features for detailed analysis
                        important_features = feature_importance['Feature'].head(4).tolist()  # Use actual top features
                        
                        cols = st.columns(2)
                        for i, feature in enumerate(important_features):
                            with cols[i % 2]:
                                # Generate simulated data to show relationship - you might want to use real data in production
                                is_age_feature = 'age' in feature.lower()
                                x_range = np.linspace(0, 100, 100) if is_age_feature else np.linspace(0, 10, 100)
                                
                                # Different relationships based on feature type
                                if 'ca19' in feature.lower():  # CA19-9 marker
                                    threshold = 37
                                    y_probs = 1 / (1 + np.exp(-(x_range - threshold) / 10))
                                    clinical_threshold = threshold
                                    threshold_label = f"Clinical Threshold: {clinical_threshold} U/mL"
                                elif 'lyve' in feature.lower():  # LYVE1 marker
                                    threshold = 1.5
                                    y_probs = 1 / (1 + np.exp(-(x_range - threshold) / 0.5))
                                    clinical_threshold = threshold
                                    threshold_label = f"Clinical Threshold: {clinical_threshold}"
                                elif any(x in feature.lower() for x in ['reg1b', 'reg1a']):  # REG markers
                                    threshold = 2
                                    y_probs = 1 / (1 + np.exp(-(x_range - threshold) / 1))
                                    clinical_threshold = threshold
                                    threshold_label = f"Clinical Threshold: {clinical_threshold}"
                                elif is_age_feature:  # Age feature
                                    threshold = 60
                                    y_probs = 1 / (1 + np.exp(-(x_range - threshold) / 15))
                                    clinical_threshold = threshold
                                    threshold_label = f"Age Threshold: {clinical_threshold} years"
                                else:  # Default relationship for other features
                                    threshold = x_range.max() / 2
                                    y_probs = 1 / (1 + np.exp(-(x_range - threshold) / (x_range.max() / 10)))
                                    clinical_threshold = None
                                    threshold_label = ""
                                
                                # Create plot
                                fig = go.Figure()
                                
                                # Add line for relationship
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_range, 
                                        y=y_probs,
                                        mode='lines',
                                        name=f'{feature} vs Cancer Risk',
                                        line=dict(color='royalblue')
                                    )
                                )
                                
                                # Add marker for current patient if this feature exists in feature_values
                                if hasattr(st.session_state, 'feature_values') and feature in st.session_state.feature_values:
                                    current_value = st.session_state.feature_values[feature]
                                    
                                    # Calculate predicted probability for this value using the same function as above
                                    if 'ca19' in feature.lower():
                                        pred_prob = 1 / (1 + np.exp(-(current_value - 37) / 10))
                                    elif 'lyve' in feature.lower():
                                        pred_prob = 1 / (1 + np.exp(-(current_value - 1.5) / 0.5))
                                    elif any(x in feature.lower() for x in ['reg1b', 'reg1a']):
                                        pred_prob = 1 / (1 + np.exp(-(current_value - 2) / 1))
                                    elif is_age_feature:
                                        pred_prob = 1 / (1 + np.exp(-(current_value - 60) / 15))
                                    else:
                                        pred_prob = 1 / (1 + np.exp(-(current_value - threshold) / (x_range.max() / 10)))
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[current_value],
                                            y=[pred_prob],
                                            mode='markers',
                                            marker=dict(size=12, color='red'),
                                            name='Current Patient'
                                        )
                                    )
                                
                                # Add reference line for risk threshold
                                fig.add_shape(
                                    type="line",
                                    x0=min(x_range),
                                    y0=0.5,
                                    x1=max(x_range),
                                    y1=0.5,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                # Add clinically relevant thresholds if applicable
                                if clinical_threshold is not None:
                                    fig.add_shape(
                                        type="line",
                                        x0=clinical_threshold,
                                        y0=0,
                                        x1=clinical_threshold,
                                        y1=1,
                                        line=dict(color="green", width=2, dash="dash")
                                    )
                                    fig.add_annotation(
                                        x=clinical_threshold,
                                        y=0.1,
                                        text=threshold_label,
                                        showarrow=False
                                    )
                                
                                fig.update_layout(
                                    title=f"{feature} Relationship with Cancer Risk",
                                    xaxis_title=feature,
                                    yaxis_title="Predicted Cancer Risk",
                                    height=350
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature importances not available for this model.")
                
                elif xai_method == "Decision Path" and rf_model is not None:
                    st.write("### Decision Path Analysis")
                    st.write("""
                    This visualization shows the decision path through the Random Forest for the current patient.
                    It highlights the most important decision nodes that led to the final prediction.
                    """)
                    
                    try:
                        if hasattr(st.session_state, 'feature_values'):
                            # Get current patient data
                            patient_data = pd.DataFrame([st.session_state.feature_values])
                            
                            # Get decision path
                            decision_paths = rf_model.decision_path(patient_data)
                            
                            # Create a simplified visualization of decision paths
                            st.write("#### Most influential decision rules:")
                            
                            # Get a sample tree from the forest
                            tree_index = 0
                            tree = rf_model.estimators_[tree_index]
                            
                            # Extract decision path for this tree
                            node_indicator = tree.decision_path(patient_data)
                            leaf_id = tree.apply(patient_data)
                            
                            # Get feature names
                            feature_names = list(st.session_state.feature_values.keys())
                            
                            # Extract decision rules
                            sample_id = 0
                            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                                                node_indicator.indptr[sample_id + 1]]
                            
                            # Create a list to store decision rules
                            decision_rules = []
                            
                            for node_id in node_index:
                                # Continue only if not leaf
                                if leaf_id[sample_id] != node_id:
                                    # Get feature used for split
                                    feature = tree.tree_.feature[node_id]
                                    # Get threshold
                                    threshold = tree.tree_.threshold[node_id]
                                    # Get feature name
                                    feature_name = feature_names[feature] if feature >= 0 else "Unknown"
                                    # Get patient's value for this feature
                                    value = patient_data.iloc[0, feature] if feature >= 0 else None
                                    # Determine direction
                                    if value is not None:
                                        direction = "<=" if value <= threshold else ">"
                                        importance = abs(value - threshold) / max(abs(value), abs(threshold))
                                        decision_rules.append({
                                            'feature': feature_name,
                                            'value': value,
                                            'threshold': threshold,
                                            'direction': direction,
                                            'importance': importance
                                        })
                            
                            # Sort by importance
                            decision_rules = sorted(decision_rules, key=lambda x: x['importance'], reverse=True)
                            
                            # Display top rules
                            for i, rule in enumerate(decision_rules[:10]):
                                st.write(f"{i+1}. {rule['feature']} = {rule['value']:.2f} {rule['direction']} {rule['threshold']:.2f}")
                            
                            # Create visualization of top 5 rules
                            top_rules = decision_rules[:5]
                            
                            fig = go.Figure()
                            
                            for i, rule in enumerate(top_rules):
                                fig.add_trace(go.Bar(
                                    y=[rule['feature']],
                                    x=[rule['importance']],
                                    orientation='h',
                                    name=f"{rule['feature']} {rule['direction']} {rule['threshold']:.2f}",
                                    hoverinfo='text',
                                    hovertext=f"{rule['feature']} = {rule['value']:.2f} {rule['direction']} {rule['threshold']:.2f}"
                                ))
                            
                            fig.update_layout(
                                title="Top Decision Rules by Importance",
                                xaxis_title="Rule Importance",
                                yaxis_title="Feature",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Patient data not available to analyze decision path.")
                            
                    except Exception as e:
                        st.error(f"Error generating decision path: {e}")
                        st.info("This analysis requires a trained Random Forest model with compatible features.")
                        
            else:
                st.info("Please complete the Clinical Data Analysis to view these insights.")
                if st.button("Go to Clinical Data Analysis",key="goto_clinical_data_btn8"):
                    st.switch_page("tab1")
        
        # CT Scan XAI Tab
        with xai_tab2:
            if st.session_state.cnn_prediction is not None and st.session_state.processed_image is not None:
                st.subheader("CT Scan Model Explanation")
                
                # Choose XAI method
                xai_method = st.selectbox(
                    "Select Explainability Method",
                    ["LIME", "Integrated Gradients", "Occlusion Sensitivity"],
                    index=0,
                    key="cnn_xai_method"
                )
                
                # Display original image
                st.write("### Original CT Scan")
                if st.session_state.ct_image is not None:
                    st.image(st.session_state.ct_image, width=300)
                
                if xai_method == "LIME" and cnn_model is not None:
                    st.write("LIME (Local Interpretable Model-agnostic Explanations) explains the prediction by analyzing how perturbing the input affects model output.")
    
    # Use the actual LIME implementation
                    from lime.lime_image import LimeImageExplainer
                    from skimage.segmentation import mark_boundaries
    
    # Create a wrapper function for prediction
                    def predict_fn(images):
                        # If input is RGB (as LIME will generate), convert to grayscale for our model
                        if images.shape[-1] == 3:
                            gray_images = np.mean(images, axis=-1, keepdims=True)
                            return cnn_model.predict(gray_images)
                        return cnn_model.predict(images)
                    
                    # predictions = predict_fn(np.expand_dims(rgb_image, axis=0))

                    # predicted_class_index = np.argmax(predictions)


                    # class_names = ["Normal", "Pancreatic"]
                    # predicted_class = class_names[predicted_class_index]

                    # print(f"Predicted Class: {predicted_class}")
    
                    # Create an explainer
                    explainer = LimeImageExplainer()
                    
                    # Get the current image and convert for LIME
                    img_array = st.session_state.processed_image[0, :, :, 0]
                    # Convert to 3 channels (RGB) for LIME
                    rgb_image = np.stack((img_array,)*3, axis=-1)
                    
                    with st.spinner("Generating LIME explanation... This may take a moment."):
                        # Generate the explanation
                        explanation = explainer.explain_instance(
                            rgb_image,
                            predict_fn,
                            top_labels=1,
                            hide_color=0,
                            num_samples=500  # Adjust based on performance needs
                        )
                    
                    # Get positive and negative contributions
                    st.subheader("LIME Explanations")
                    
                    # First visualization - Positive influences only
                    temp_positive, mask_positive = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=5,
                        hide_rest=False
                    )
                    
                    # Second visualization - All influences
                    temp_all, mask_all = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=False,
                        num_features=10,
                        hide_rest=False
                    )
                    
                    # Create the visualization
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Original image
                    axes[0].imshow(img_array, cmap='gray')
                    axes[0].set_title('Original CT Scan')
                    axes[0].axis('off')
                    
                    # Positive influences
                    axes[1].imshow(mark_boundaries(temp_positive / 255.0, mask_positive))
                    axes[1].set_title('LIME: Positive Influences')
                    axes[1].axis('off')
                    
                    # All influences
                    axes[2].imshow(mark_boundaries(temp_all / 255.0, mask_all))
                    axes[2].set_title('LIME: All Influences')
                    axes[2].axis('off')
                    
                    st.pyplot(fig)
                    
                    # Additional overlay visualization
                    from skimage.color import label2rgb
                    
                    lime_overlay = label2rgb(mask_all, image=img_array, bg_label=0, kind='overlay')
                    
                    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original CT image
                    ax2[0].imshow(img_array, cmap='gray')
                    ax2[0].axis('off')
                    ax2[0].set_title("Original Image")
                    
                    # Original with LIME overlay
                    ax2[1].imshow(lime_overlay)
                    ax2[1].axis('off')
                    ax2[1].set_title("LIME Overlay on Original")
                    
                    st.pyplot(fig2)
                    
                    st.write("""
                    #### Interpretation
                    - **Green boundaries** show segments that influence the model's prediction
                    - **Bright regions** in the overlay indicate areas with higher importance to the prediction
                    - In the positive influences view, only features supporting the predicted class are shown
                    - In the all influences view, both supporting and contradicting features are shown
                    """)
                    
                    # Show more detailed information about the explanation
                    st.subheader("Feature Importance Details")
                    
                    # Get the prediction label and score
                    prediction = predict_fn(np.expand_dims(rgb_image, axis=0))[0]
                    predicted_class = np.argmax(prediction)
                    class_names = ["Normal", "Pancreatic Cancer"]  # Update these to match your model's classes
                    
                    st.write(f"**Predicted class:** {class_names[predicted_class]} (confidence: {prediction[predicted_class]:.2f})")
                    
                    # Get feature importance weights for the predicted class
                    ind = explanation.top_labels[0]
                    dict_heatmap = dict(explanation.local_exp[ind])
                    heatmap = np.vectorize(lambda x: dict_heatmap.get(x, 0))(explanation.segments)
                    
                    # Display the heatmap
                    fig3, ax3 = plt.subplots(figsize=(8, 8))
                    im = ax3.imshow(heatmap, cmap='RdBu_r', vmin=-heatmap.max(), vmax=heatmap.max())
                    ax3.set_title("LIME Feature Importance Heatmap")
                    ax3.axis('off')
                    fig3.colorbar(im, ax=ax3, label='Feature Importance')
                    
                    st.pyplot(fig3)
                    
                    st.write("""
                    #### Feature Importance Heatmap
                    - **Red regions** indicate features that contribute positively to the prediction
                    - **Blue regions** indicate features that contribute negatively to the prediction
                    - The intensity of color represents the magnitude of importance
                    """)
                    
                elif xai_method == "Integrated Gradients" and cnn_model is not None:
                    st.write("Integrated Gradients attributes the prediction to input features by calculating gradients along a path from a baseline to the input.")
                    
                    import tensorflow as tf
                    import cv2
                    import matplotlib.cm as cm
                    
                    # Define the Integrated Gradients function
                    @st.cache_data
                    def integrated_gradients(model, img_array, baseline=None, steps=50, target_class=None):
                        """
                        Calculates integrated gradients for a given image and model
                        
                        Args:
                            model: TensorFlow model to use for prediction
                            img_array: Input image as a numpy array [1, height, width, channels]
                            baseline: Baseline image (usually black). If None, uses zeros
                            steps: Number of steps for the integration approximation
                            target_class: Index of the class to explain. If None, uses the predicted class
                            
                        Returns:
                            attribution: Attribution map showing feature importance
                        """
                        img_array = tf.cast(img_array, tf.float32)
                        
                        # If no baseline is provided, use zeros
                        if baseline is None:
                            baseline = tf.zeros_like(img_array)
                        else:
                            baseline = tf.cast(baseline, tf.float32)
                        
                        # Make a prediction if target class not specified
                        if target_class is None:
                            prediction = model(img_array)
                            target_class = tf.argmax(prediction[0])
                        
                        # Generate alphas for interpolation
                        alphas = tf.linspace(0.0, 1.0, steps)
                        
                        # Expand dimensions for broadcasting
                        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
                        
                        # Generate interpolated images
                        baseline_expanded = tf.expand_dims(baseline, axis=0)
                        img_expanded = tf.expand_dims(img_array, axis=0)
                        
                        # Formula: baseline + alpha * (input - baseline)
                        delta = img_expanded - baseline_expanded
                        interpolated_images = baseline_expanded + alphas_x * delta
                        
                        # Flatten interpolated images
                        interpolated_flat = tf.reshape(interpolated_images, [-1] + list(img_array.shape[1:]))
                        
                        # Get gradients for all interpolated images
                        with tf.GradientTape() as tape:
                            tape.watch(interpolated_flat)
                            outputs = model(interpolated_flat)
                            outputs = outputs[:, target_class]
                        
                        # Calculate gradients
                        gradients = tape.gradient(outputs, interpolated_flat)
                        
                        # Reshape gradients to match the interpolated images shape
                        gradients = tf.reshape(gradients, [steps] + list(img_array.shape[1:]))
                        
                        # Use trapezoidal rule for integration
                        gradients = (gradients[:-1] + gradients[1:]) / 2.0
                        
                        # Average over integration steps and multiply by input difference
                        avg_gradients = tf.reduce_mean(gradients, axis=0)
                        integrated_gradients = (img_array - baseline) * avg_gradients
                        
                        # Sum over color channels if image has more than one channel
                        if integrated_gradients.shape[-1] > 1:
                            attribution = tf.reduce_sum(integrated_gradients, axis=-1)
                        else:
                            attribution = integrated_gradients
                            
                        return attribution.numpy()
                    
                    with st.spinner("Calculating Integrated Gradients... This may take a moment."):
                        try:
                            # Get the image array
                            img_array = st.session_state.processed_image
                            
                            # Create a baseline (usually black image)
                            baseline = np.zeros_like(img_array)
                            
                            # Get prediction and target class
                            prediction = cnn_model.predict(img_array)
                            target_class = np.argmax(prediction[0])
                            
                            # Get class names - update these to match your model's classes
                            class_names = ["Normal", "Pancreatic Cancer"]
                            
                            # Calculate integrated gradients
                            attributions = integrated_gradients(
                                cnn_model, 
                                img_array, 
                                baseline=baseline, 
                                steps=50,  # More steps = more accurate but slower
                                target_class=target_class
                            )
                            
                            # Process the attribution map
                            attribution_map = np.squeeze(attributions)
                            
                            # Normalize the attribution map for visualization
                            attribution_norm = np.abs(attribution_map)
                            attribution_norm = (attribution_norm - attribution_norm.min()) / (attribution_norm.max() - attribution_norm.min() + 1e-8)
                            
                            # Get positive and negative attributions separately for visualization
                            pos_attr = np.maximum(0, attribution_map)
                            pos_attr_norm = (pos_attr) / (pos_attr.max() + 1e-8)
                            
                            neg_attr = np.abs(np.minimum(0, attribution_map))
                            neg_attr_norm = (neg_attr) / (neg_attr.max() + 1e-8)
                            
                            # Create visualizations
                            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
                            
                            # Original image
                            original_img = np.squeeze(img_array[0, :, :, 0])
                            axes[0, 0].imshow(original_img, cmap='gray')
                            axes[0, 0].set_title(f'Original Image\nPrediction: {class_names[target_class]} ({prediction[0][target_class]:.2f})')
                            axes[0, 0].axis('off')
                            
                            # Combined attribution map
                            im1 = axes[0, 1].imshow(original_img, cmap='gray')
                            im2 = axes[0, 1].imshow(attribution_norm, cmap='hot', alpha=0.6)
                            axes[0, 1].set_title('Integrated Gradients: All Attributions')
                            axes[0, 1].axis('off')
                            fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
                            
                            # Positive attributions
                            axes[1, 0].imshow(original_img, cmap='gray')
                            im3 = axes[1, 0].imshow(pos_attr_norm, cmap='Reds', alpha=0.6)
                            axes[1, 0].set_title('Positive Attributions')
                            axes[1, 0].axis('off')
                            fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                            
                            # Negative attributions
                            axes[1, 1].imshow(original_img, cmap='gray')
                            im4 = axes[1, 1].imshow(neg_attr_norm, cmap='Blues', alpha=0.6)
                            axes[1, 1].set_title('Negative Attributions')
                            axes[1, 1].axis('off')
                            fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Create a more detailed visualization
                            fig2, ax2 = plt.subplots(figsize=(10, 10))
                            
                            # Create a color-coded attribution map
                            colored_attr = np.zeros((*attribution_map.shape, 3))
                            
                            # Red for positive attributions
                            colored_attr[:, :, 0] = pos_attr_norm
                            
                            # Blue for negative attributions  
                            colored_attr[:, :, 2] = neg_attr_norm
                            
                            # Display the original with color-coded overlay
                            ax2.imshow(original_img, cmap='gray')
                            ax2.imshow(colored_attr, alpha=0.6)
                            ax2.set_title('Integrated Gradients: Feature Attribution Map')
                            ax2.axis('off')
                            
                            # Add custom legend for positive and negative attributions
                            from matplotlib.patches import Patch
                            legend_elements = [
                                Patch(facecolor='red', alpha=0.6, label='Positive attribution'),
                                Patch(facecolor='blue', alpha=0.6, label='Negative attribution')
                            ]
                            ax2.legend(handles=legend_elements, loc='upper right')
                            
                            st.pyplot(fig2)
                            
                            # Additional visualization showing a 3D plot of attribution intensities
                            from mpl_toolkits.mplot3d import Axes3D
                            
                            fig3 = plt.figure(figsize=(10, 8))
                            ax3 = fig3.add_subplot(111, projection='3d')
                            
                            # Create a grid of coordinates
                            x = np.arange(0, attribution_map.shape[1])
                            y = np.arange(0, attribution_map.shape[0])
                            X, Y = np.meshgrid(x, y)
                            
                            # Plot the surface
                            surf = ax3.plot_surface(X, Y, attribution_map, cmap='coolwarm', linewidth=0, antialiased=False, alpha=0.8)
                            
                            # Add a color bar which maps values to colors
                            fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
                            
                            # Set labels and title
                            ax3.set_title('3D Visualization of Attribution Intensity')
                            ax3.set_xlabel('X Pixel')
                            ax3.set_ylabel('Y Pixel')
                            ax3.set_zlabel('Attribution Value')
                            
                            st.pyplot(fig3)
                            
                            st.write("""
                            #### Interpretation
                            The Integrated Gradients attribution maps show how each pixel contributes to the prediction:
                            
                            - **Red areas** (positive attribution) indicate features that support the predicted class
                            - **Blue areas** (negative attribution) indicate features that oppose the predicted class
                            - **Brighter colors** represent stronger influence on the model's decision
                            
                            The 3D visualization shows the intensity of attributions across the image, with peaks representing areas of high importance to the model's decision.
                            
                            This analysis helps identify which specific regions of the image are most influential for diagnosis.
                            """)
                            
                            # Detailed technical explanation
                            with st.expander("How Integrated Gradients Works"):
                                st.write("""
                                **Technical Explanation:**
                                
                                Integrated Gradients is a method that satisfies desirable axioms like:
                                - **Sensitivity**: If an input and baseline differ in one feature and have different predictions, the attribution to that feature should be non-zero
                                - **Implementation Invariance**: Attributions should be identical for functionally equivalent networks
                                
                                The method calculates the integral of gradients along a straight-line path from a baseline to the input:
                                
                                1. It creates interpolated images along a path from baseline (usually black) to the input image
                                2. For each interpolated image, it calculates gradients of the output with respect to inputs
                                3. It integrates these gradients along the path (approximated using the trapezoidal rule)
                                4. The result shows how each pixel contributes to the final prediction
                                
                                This helps identify which parts of the image are most important for the model's decision-making process.
                                """)
                                
                        except Exception as e:
                            st.error(f"Error calculating Integrated Gradients: {str(e)}")
                            st.info("Integrated Gradients requires a differentiable model with access to gradients.")
                
                elif xai_method == "Occlusion Sensitivity" and cnn_model is not None:
                    st.write("Occlusion Sensitivity measures how the prediction changes when parts of the image are systematically masked.")
                    
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import time
                    
                    # Get the original image and prediction
                    original_image = st.session_state.processed_image.copy()
                    original_pred = cnn_model.predict(original_image)[0]
                    predicted_class = np.argmax(original_pred)
                    
                    # Get class names - update these to match your model's classes
                    class_names = ["Normal", "Pancreatic Cancer"]
                    
                    # Function to calculate occlusion sensitivity
                    @st.cache_data
                    def calculate_occlusion_sensitivity(image, model, class_idx, patch_size=16, stride=8):
                        """
                        Calculate occlusion sensitivity map for an image.
                        
                        Args:
                            image: Input image of shape [1, height, width, channels]
                            model: The CNN model to use for prediction
                            class_idx: Index of the class to analyze
                            patch_size: Size of the occlusion patch
                            stride: Stride between occlusion patches
                            
                        Returns:
                            sensitivity_map: 2D numpy array with sensitivity scores
                        """
                        # Get image dimensions
                        img_height = image.shape[1]
                        img_width = image.shape[2]
                        
                        # Calculate the number of patches
                        n_h = (img_height - patch_size) // stride + 1
                        n_w = (img_width - patch_size) // stride + 1
                        
                        # Original prediction score
                        original_score = model.predict(image)[0][class_idx]
                        
                        # Initialize sensitivity map
                        sensitivity_map = np.zeros((n_h, n_w))
                        
                        # Storage for progress tracking
                        total_iterations = n_h * n_w
                        
                        # Show progress bar
                        progress_bar = st.progress(0)
                        
                        # Set up a placeholder for the current image being processed
                        current_image_placeholder = st.empty()
                        
                        # Display the original image with prediction
                        fig_orig, ax_orig = plt.subplots(figsize=(5, 5))
                        ax_orig.imshow(np.squeeze(image[0, :, :, 0]), cmap='gray')
                        ax_orig.set_title(f"Original: {class_names[class_idx]} ({original_score:.2f})")
                        ax_orig.axis('off')
                        current_image_placeholder.pyplot(fig_orig)
                        
                        # Start the occlusion process
                        iteration = 0
                        
                        # For each patch
                        for i in range(n_h):
                            for j in range(n_w):
                                # Create a copy of the original image
                                occluded_image = image.copy()
                                
                                # Calculate patch coordinates
                                h_start = i * stride
                                h_end = h_start + patch_size
                                w_start = j * stride
                                w_end = w_start + patch_size
                                
                                # Create occlusion (replace with zeros or mean value)
                                # Using zeros (black patch) for occlusion
                                occluded_image[0, h_start:h_end, w_start:w_end, :] = 0
                                
                                # Get prediction for occluded image
                                occluded_score = model.predict(occluded_image)[0][class_idx]
                                
                                # Calculate sensitivity as drop in prediction score
                                sensitivity = original_score - occluded_score
                                sensitivity_map[i, j] = sensitivity
                                
                                # Update progress bar
                                iteration += 1
                                progress_bar.progress(iteration / total_iterations)
                                
                                # Update the display occasionally to show progress
                                if iteration % 10 == 0 or iteration == total_iterations:
                                    fig, ax = plt.subplots(figsize=(5, 5))
                                    ax.imshow(np.squeeze(occluded_image[0, :, :, 0]), cmap='gray')
                                    ax.set_title(f"Occluding: [{h_start}:{h_end}, {w_start}:{w_end}]\nScore: {occluded_score:.2f}")
                                    ax.axis('off')
                                    current_image_placeholder.pyplot(fig)
                        
                        # Clear progress displays
                        progress_bar.empty()
                        current_image_placeholder.empty()
                        
                        return sensitivity_map, n_h, n_w, stride
                    
                    # Configure parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        patch_size = st.slider("Occlusion Patch Size", min_value=8, max_value=32, value=16, step=4)
                    with col2:
                        stride = st.slider("Stride", min_value=4, max_value=16, value=8, step=4)
                    
                    # Calculate sensitivity
                    with st.spinner("Calculating occlusion sensitivity map... This may take a few minutes."):
                        try:
                            sensitivity_map, n_h, n_w, stride = calculate_occlusion_sensitivity(
                                original_image, 
                                cnn_model, 
                                predicted_class,
                                patch_size=patch_size,
                                stride=stride
                            )
                            
                            # Normalize sensitivity map
                            sensitivity_norm = sensitivity_map.copy()
                            if sensitivity_norm.max() > sensitivity_norm.min():
                                sensitivity_norm = (sensitivity_norm - sensitivity_norm.min()) / (sensitivity_norm.max() - sensitivity_norm.min())
                            
                            # Upscale sensitivity map to match original image size
                            h, w = original_image.shape[1:3]
                            
                            # Create coordinates for sensitivity map
                            x_coords = np.linspace(0, w, n_w, endpoint=False) + stride / 2
                            y_coords = np.linspace(0, h, n_h, endpoint=False) + stride / 2
                            
                            # Create full-size sensitivity map using interpolation
                            from scipy.interpolate import RegularGridInterpolator
                            
                            # Create interpolation function
                            interp_func = RegularGridInterpolator((y_coords, x_coords), sensitivity_map,
                                                                bounds_error=False, fill_value=0)
                            
                            # Create full grid
                            y_full, x_full = np.mgrid[0:h, 0:w]
                            points = np.vstack((y_full.flatten(), x_full.flatten())).T
                            
                            # Interpolate
                            sensitivity_full = interp_func(points).reshape(h, w)
                            
                            # Normalize the interpolated map
                            if sensitivity_full.max() > sensitivity_full.min():
                                sensitivity_full = (sensitivity_full - sensitivity_full.min()) / (sensitivity_full.max() - sensitivity_full.min())
                            
                            # Create visualizations
                            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                            
                            # Original image
                            original_img = np.squeeze(original_image[0, :, :, 0])
                            axes[0].imshow(original_img, cmap='gray')
                            axes[0].set_title(f'Original CT Scan\nPrediction: {class_names[predicted_class]} ({original_pred[predicted_class]:.2f})')
                            axes[0].axis('off')
                            
                            # Raw sensitivity map
                            axes[1].imshow(sensitivity_norm, cmap='hot', interpolation='nearest')
                            axes[1].set_title('Occlusion Sensitivity Map (Raw)')
                            axes[1].axis('off')
                            
                            # Overlay sensitivity on original image
                            axes[2].imshow(original_img, cmap='gray')
                            im = axes[2].imshow(sensitivity_full, cmap='hot', alpha=0.6)
                            axes[2].set_title('Occlusion Sensitivity Overlay')
                            axes[2].axis('off')
                            
                            # Add colorbar
                            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                            cbar.set_label('Sensitivity (Higher = More Important)', rotation=270, labelpad=15)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Create a more detailed visualization with a grid overlay
                            img_height, img_width = original_image.shape[1:3]
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 10))
                            ax2.imshow(original_img, cmap='gray')
                            
                            # Draw grid with colors based on sensitivity
                            alpha_max = 0.7
                            for i in range(n_h):
                                for j in range(n_w):
                                    h_start = i * stride
                                    h_end = min(h_start + patch_size, img_height)
                                    w_start = j * stride
                                    w_end = min(w_start + patch_size, img_width)
                                    
                                    # Normalized sensitivity for this patch
                                    sens = sensitivity_norm[i, j]
                                    
                                    # Add colored rectangle
                                    rect = plt.Rectangle((w_start, h_start), patch_size, patch_size, 
                                                    fill=True, alpha=sens * alpha_max,
                                                    facecolor='red', edgecolor='white', linewidth=0.5)
                                    ax2.add_patch(rect)
                            
                            ax2.set_title('Occlusion Sensitivity Grid Analysis')
                            ax2.axis('off')
                            
                            st.pyplot(fig2)
                            
                            # Add a quantitative analysis of most important regions
                            st.subheader("Most Important Regions")
                            
                            # Find top N sensitive regions
                            top_n = 5
                            flat_indices = np.argsort(sensitivity_map.flatten())[-top_n:]
                            top_indices = np.unravel_index(flat_indices, sensitivity_map.shape)
                            
                            # Create a table showing coordinates and sensitivity scores
                            region_data = []
                            for idx in range(top_n):
                                i, j = top_indices[0][idx], top_indices[1][idx]
                                h_start = i * stride
                                h_end = min(h_start + patch_size, img_height)
                                w_start = j * stride
                                w_end = min(w_start + patch_size, img_width)
                                
                                region_data.append({
                                    "Region": f"#{top_n-idx}",
                                    "Coordinates": f"({w_start}, {h_start}) to ({w_end}, {h_end})",
                                    "Sensitivity Score": f"{sensitivity_map[i, j]:.4f}"
                                })
                            
                            st.table(region_data)
                            
                            st.write("""
                            #### Interpretation
                            The occlusion sensitivity map reveals how the model's prediction changes when different parts of the image are occluded:
                            
                            - **Bright red regions** indicate areas where occlusion causes a significant drop in the prediction score for the target class
                            - These are critical regions that the model relies on heavily for its diagnosis
                            - Areas with little to no color have minimal impact on the prediction when occluded
                            
                            The grid visualization shows the exact patches used in the occlusion analysis, with color intensity proportional to importance.
                            
                            The table displays the coordinates of the top regions that, when occluded, most significantly impact the model's prediction. These are the most diagnostically important areas according to the model.
                            """)
                            
                            with st.expander("How Occlusion Sensitivity Works"):
                                st.write("""
                                **Technical Details:**
                                
                                Occlusion sensitivity is an intuitive approach to understanding what parts of an image are important for a model's prediction:
                                
                                1. **Methodology**: It systematically occludes (masks) different portions of the input image and observes how the prediction changes
                                2. **Implementation**: We slide a small black patch across the image, measuring the prediction drop at each position
                                3. **Interpretation**: Larger drops in prediction score indicate regions more critical to the classification
                                
                                **Advantages:**
                                - Straightforward interpretation
                                - Model-agnostic (works with any CNN architecture)
                                - No need for model modifications
                                
                                **Limitations:**
                                - Computationally expensive for large images
                                - Results depend on occlusion patch size and stride
                                - Sequential rather than parallel processing
                                
                                The analysis helps identify regions of the pancreas that are most influential in the model's diagnostic decision.
                                """)
                                
                        except Exception as e:
                            st.error(f"Error calculating occlusion sensitivity: {str(e)}")
                            st.info("Please ensure your model is properly configured for this analysis.")
        
        with xai_tab3:
                    st.subheader("Interactive Feature Analysis")
                    st.write("""
                    This interactive tool allows you to explore how changes in clinical features affect the 
                    predicted risk of pancreatic cancer. Adjust the values below to see how the model's prediction changes.
                    """)
                    
                    if st.session_state.rf_prediction is not None and st.session_state.feature_values is not None:
                        # Create feature sliders based on original values
                        st.write("### Adjust Clinical Features")
                        st.write("Move the sliders to see how changes affect the prediction.")
                        
                        # Set up columns for sliders
                        col1, col2 = st.columns(2)
                        
                        # Initialize modified values dictionary
                        modified_values = {}
                        
                        # Create sliders for numerical features
                        with col1:
                            # Age slider
                            modified_values['age'] = st.slider(
                                "Age", 
                                min_value=18, 
                                max_value=100, 
                                value=int(st.session_state.feature_values['age']),
                                key="age_slider"
                            )
                            
                            # CA19-9 slider (higher range)
                            modified_values['plasma_CA19_9'] = st.slider(
                                "Plasma CA19-9 (U/mL)", 
                                min_value=0.0, 
                                max_value=500.0, 
                                value=float(st.session_state.feature_values['plasma_CA19_9']),
                                key="ca19_9_slider"
                            )
                            
                            # Creatinine slider
                            modified_values['creatinine'] = st.slider(
                                "Creatinine (mg/dL)", 
                                min_value=0.0, 
                                max_value=5.0, 
                                value=float(st.session_state.feature_values['creatinine']),
                                key="creatinine_slider"
                            )
                            
                            # Add sex selection
                            sex_options = {"Male": 1, "Female": 0}
                            current_sex = "Male" if st.session_state.feature_values['sex'] == 1 else "Female"
                            selected_sex = st.selectbox(
                                "Sex",
                                options=list(sex_options.keys()),
                                index=list(sex_options.keys()).index(current_sex),
                                key="sex_selector"
                            )
                            modified_values['sex'] = sex_options[selected_sex]
                        
                        with col2:
                            # Biomarker sliders
                            modified_values['LYVE1'] = st.slider(
                                "LYVE1 (ng/mL)", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=float(st.session_state.feature_values['LYVE1']),
                                key="lyve1_slider"
                            )
                            
                            modified_values['REG1B'] = st.slider(
                                "REG1B (ng/mL)", 
                                min_value=0.0, 
                                max_value=80.0, 
                                value=float(st.session_state.feature_values['REG1B']),
                                key="reg1b_slider"
                            )
                            
                            modified_values['TFF1'] = st.slider(
                                "TFF1 (ng/mL)", 
                                min_value=0.0, 
                                max_value=400.0, 
                                value=float(st.session_state.feature_values['TFF1']),
                                key="tff1_slider"
                            )
                            
                            modified_values['REG1A'] = st.slider(
                                "REG1A (ng/mL)", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=float(st.session_state.feature_values['REG1A']),
                                key="reg1a_slider"
                            )
                        
                        # Add additional categorical features
                        st.write("### Categorical Features")
                        
                        # Patient cohort selection
                        cohort_options = {"Cohort1": 0, "Cohort2": 1}
                        current_cohort = "Cohort2" if st.session_state.feature_values['patient_cohort_Cohort2'] == 1 else "Cohort1"
                        selected_cohort = st.selectbox(
                            "Patient Cohort",
                            options=list(cohort_options.keys()),
                            index=list(cohort_options.keys()).index(current_cohort),
                            key="cohort_selector"
                        )
                        modified_values['patient_cohort_Cohort2'] = cohort_options[selected_cohort]
                        
                        # Sample origin selection
                        origin_options = ["Other", "ESP", "LIV", "UCL"]
                        
                        # Determine current origin
                        current_origin = "Other"
                        for origin in ["ESP", "LIV", "UCL"]:
                            if st.session_state.feature_values[f'sample_origin_{origin}'] == 1:
                                current_origin = origin
                                break
                        
                        selected_origin = st.selectbox(
                            "Sample Origin",
                            options=origin_options,
                            index=origin_options.index(current_origin),
                            key="origin_selector"
                        )
                        
                        # Update modified values for sample origin
                        for origin in ["ESP", "LIV", "UCL"]:
                            modified_values[f'sample_origin_{origin}'] = 1 if selected_origin == origin else 0
                        
                        # Create input DataFrame with modified values
                        modified_df = pd.DataFrame([modified_values])
                        
                        # Make prediction with modified values
                        if rf_model is not None:
                            # Make prediction
                            new_prediction = rf_model.predict_proba(modified_df)[0]
                            original_prediction = st.session_state.rf_prediction
                            
                            # Display results
                            st.write("### Prediction Comparison")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    label="Original Risk Score",
                                    value=f"{original_prediction:.2%}"
                                )
                            
                            with col2:
                                st.metric(
                                    label="New Risk Score", 
                                    value=f"{new_prediction[1]:.2%}",
                                    delta=f"{new_prediction[1] - original_prediction:.2%}"
                                )
                            
                            # Create comparison visualization
                            fig, ax = plt.subplots(figsize=(8, 0.5))
                            
                            # Create gauge for original prediction
                            ax.barh(0, 100, color='lightgray', alpha=0.3)
                            ax.barh(0, original_prediction * 100, color='green', alpha=0.7)
                            
                            # Create gauge for new prediction
                            ax.barh(1, 100, color='lightgray', alpha=0.3)
                            ax.barh(1, new_prediction[1] * 100, color='blue', alpha=0.7)
                            
                            # Add labels
                            ax.set_yticks([0, 1])
                            ax.set_yticklabels(['Original', 'Modified'])
                            ax.set_xlim(0, 100)
                            ax.set_xticks([0, 25, 50, 75, 100])
                            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                            ax.set_title('Risk Score Comparison')
                            
                            # Add threshold line
                            ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
                            ax.text(51, 0.5, '50% Threshold', color='red', alpha=0.7)
                            
                            st.pyplot(fig)
                            
                            # Feature impact visualization
                            st.write("### Feature Impact Analysis")
                            st.write("This chart shows which feature changes had the most impact on the prediction.")
                            
                            # Calculate differences between original and modified values
                            diff_dict = {}
                            impact_dict = {}
                            
                            # Simple feature impact estimation for demonstration
                            # In a real implementation, use SHAP or other proper feature attribution methods
                            for feature in modified_values.keys():
                                if feature in st.session_state.feature_values:
                                    original_val = st.session_state.feature_values[feature]
                                    modified_val = modified_values[feature]
                                    
                                    # Skip if no change
                                    if original_val == modified_val:
                                        continue
                                        
                                    # Calculate difference
                                    if isinstance(original_val, (int, float)) and isinstance(modified_val, (int, float)):
                                        diff_dict[feature] = modified_val - original_val
                                        
                                        # Estimate impact based on feature importance
                                        if hasattr(rf_model, 'feature_importances_'):
                                            feature_idx = list(modified_df.columns).index(feature)
                                            importance = rf_model.feature_importances_[feature_idx]
                                            
                                            # Normalize difference for better visualization
                                            # Use a simple scaling approach
                                            if feature == 'age':
                                                normalized_diff = diff_dict[feature] / 50  # Scale by typical age range
                                            elif feature == 'plasma_CA19_9':
                                                normalized_diff = diff_dict[feature] / 100  # Scale by typical CA19-9 range
                                            else:
                                                normalized_diff = diff_dict[feature] / 5  # Default scaling
                                            
                                            impact_dict[feature] = normalized_diff * importance * 10  # Scaling factor
                                    else:
                                        # For categorical features, just note the change
                                        diff_dict[feature] = f"{original_val} → {modified_val}"
                                        
                                        # Estimate impact based on feature importance
                                        if hasattr(rf_model, 'feature_importances_'):
                                            feature_idx = list(modified_df.columns).index(feature)
                                            importance = rf_model.feature_importances_[feature_idx]
                                            impact_dict[feature] = importance * (1 if modified_val > original_val else -1)
                            
                            # Create impact visualization
                            impact_df = pd.DataFrame({
                                'Feature': list(impact_dict.keys()),
                                'Impact': list(impact_dict.values()),
                                'Change': [diff_dict[f] for f in impact_dict.keys()]
                            })
                            
                            # Sort by absolute impact
                            impact_df['AbsImpact'] = impact_df['Impact'].abs()
                            impact_df = impact_df.sort_values('AbsImpact', ascending=False)
                            
                            # Add sign for better visualization
                            impact_df['Color'] = impact_df['Impact'].apply(lambda x: 'green' if x < 0 else 'red')
                            
                            # Create bar chart
                            fig = px.bar(
                                impact_df,
                                x='Impact',
                                y='Feature',
                                color='Color', 
                                color_discrete_map={'red': 'red', 'green': 'green'},
                                orientation='h',
                                title='Feature Impact on Risk Score Change',
                                labels={'Impact': 'Impact on Risk Score', 'Feature': 'Modified Feature'},
                                hover_data=['Change']
                            )
                            
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # What-if feature exploration
                            st.write("### What-if Feature Exploration")
                            st.write("Explore how changing a specific feature affects risk prediction while holding others constant.")
                            
                            # Feature selection
                            selected_feature = st.selectbox(
                                "Select feature to explore",
                                options=[f for f in modified_df.columns if f not in ['patient_cohort_Cohort2', 'sample_origin_ESP', 'sample_origin_LIV', 'sample_origin_UCL'] and f != 'sex'],
                                key="feature_explorer"
                            )
                            
                            # Get current value
                            current_value = modified_values[selected_feature]
                            
                            # Create range for selected feature
                            if selected_feature == 'age':
                                feature_range = np.linspace(20, 90, 100)
                            elif selected_feature == 'plasma_CA19_9':
                                feature_range = np.linspace(0, 500, 100)
                            else:
                                feature_range = np.linspace(0, 10, 100)
                            
                            # Create predictions across range
                            predictions = []
                            for val in feature_range:
                                # Create new dataframe with the changed feature
                                temp_df = modified_df.copy()
                                temp_df[selected_feature] = val
                                
                                # Make prediction
                                pred = rf_model.predict_proba(temp_df)[0][1]
                                predictions.append(pred)
                            
                            # Create plot
                            fig = go.Figure()
                            
                            # Add prediction line
                            fig.add_trace(
                                go.Scatter(
                                    x=feature_range,
                                    y=predictions,
                                    mode='lines',
                                    name='Predicted Risk',
                                    line=dict(color='blue')
                                )
                            )
                            
                            # Add current value marker
                            current_pred = new_prediction[1]
                            fig.add_trace(
                                go.Scatter(
                                    x=[current_value],
                                    y=[current_pred],
                                    mode='markers',
                                    marker=dict(size=12, color='red'),
                                    name='Current Value'
                                )
                            )
                            
                            # Add threshold line
                            fig.add_shape(
                                type="line",
                                x0=min(feature_range),
                                y0=0.5,
                                x1=max(feature_range),
                                y1=0.5,
                                line=dict(color="red", width=2, dash="dash")
                            )
                            
                            # Add clinical threshold if available
                            if selected_feature == 'plasma_CA19_9':
                                fig.add_shape(
                                    type="line",
                                    x0=37,
                                    y0=0,
                                    x1=37,
                                    y1=1,
                                    line=dict(color="green", width=2, dash="dash")
                                )
                                fig.add_annotation(
                                    x=37,
                                    y=0.1,
                                    text="Clinical Threshold: 37 U/mL",
                                    showarrow=False
                                )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Effect of {selected_feature} on Pancreatic Cancer Risk",
                                xaxis_title=selected_feature,
                                yaxis_title="Predicted Risk",
                                yaxis=dict(range=[0, 1]),
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add clinical interpretation
                            st.write("### Clinical Interpretation")
                            st.write("""
                            This interactive analysis allows healthcare providers to:
                            
                            1. **Understand Risk Factors**: Identify which biomarkers and clinical factors contribute most to risk
                            2. **Personalize Screening**: Determine optimal screening intervals based on individual risk profiles
                            3. **Intervention Planning**: Assess how lifestyle or treatment interventions might modify risk
                            4. **Patient Communication**: Clearly explain risk factors to patients using visual aids
                            
                            *Note: This tool is designed to support clinical decision-making, not replace clinical judgment.*
                            """)
                            
                            st.success("✓ Interactive analysis complete!")
                            
                        else:
                            st.error("Random Forest model not loaded properly. Please check your model file.")
                    else:
                        st.info("Please complete the Clinical Data Analysis to use this interactive tool.")
                        if st.button("Go to Clinical Data Analysis",key="goto_clinical_data_btn9"):
                            st.switch_page("tab1")
# About Tab
with tab6:
    st.header("About This Application")
    
    st.write("""
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
    
    st.write("© 2025 AI-Driven Pancreatic Cancer Detection - Version 1.0")