# AI-Driven Early Detection of Pancreatic Cancer

An AI/ML system for early detection of pancreatic cancer using multimodal data and explainable models (SHAP, LIME).

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Dataset & Preprocessing](#dataset-preprocessing)
* [Models & Explainability](#models-explainability)
* [Installation & Usage](#installation-usage)
* [Project Structure](#project-structure)
* [How to Contribute](#how-to-contribute)
* [Future Work](#future-work)
* [License](#license)
* [Author / Contact](#author-contact)

## Overview

Pancreatic cancer is one of the deadliest cancers due to its late detection and rapid progression. This project aims to **leverage multimodal patient data** (clinical, imaging, laboratory, etc.) and **machine learning / deep learning** techniques to identify early-onset risk factors and predict the presence of pancreatic cancer. Furthermore, the models incorporate **explainability tools** such as SHAP and LIME to provide transparency and insight into predictions.

## Features

* Pretrained classification models (e.g., Random Forest, CNN) for pancreatic cancer detection.
* Explainable AI modules using SHAP & LIME to interpret model decisions.
* Modular Python scripts for data ingestion, preprocessing, model training, evaluation and deployment.
* Ready-to-use model files (e.g., `random_forest_pancreatic_model.pkl`, `pancreas_cnn_model.h5`).
* Prototype application (`app.py`) for demonstration/prediction.

## Tech Stack

* **Programming Language**: Python
* **Libraries/Frameworks**: scikit-learn, TensorFlow/Keras (or PyTorch if applicable), SHAP, LIME
* **Data Format**: CSV, image formats (if imaging modality used)
* **Deployment**: Simple Flask/Streamlit app (`app.py`) for inference demo
* **Version Control**: Git & GitHub

## Dataset & Preprocessing

* The `patient_data` folder contains raw or processed clinical datasets.
* The `test_data` folder contains sample/test cases for inference.
* Preprocessing scripts (`strategy.py`, etc.) handle cleaning, transformation, feature‐engineering and splitting of data.
* Imaging modalities (if used) are processed with appropriate augmentation, normalization and resizing.

## Models & Explainability

* `random_forest_pancreatic_model.pkl` — a Random Forest classifier trained on structured patient data.
* `pancreas_cnn_model.h5` — a convolutional neural network model possibly used on imaging data (CT/MRI) of pancreas.
* Explainability scripts (`proto.py`, `protos.py`, `prototype.py`) demonstrate how to use SHAP or LIME to visualize feature/voxel importance for predictions.
* The `final.py`, `final2.py`, `new.py` scripts likely reflect final versions or experiment variants.

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/JeremiahRanen7/AI-Driven-Early-Detection-of-Pancreatic-Cancer.git  
   cd AI-Driven-Early-Detection-of-Pancreatic-Cancer  
   ```
2. Set up a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv  
   source venv/bin/activate    # On Windows: venv\Scripts\activate  
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt  
   ```

   *(If a `requirements.txt` file is not present, create one with necessary packages: scikit-learn, tensorflow, shap, lime, flask, pandas, numpy, etc.)*
4. Run preprocessing and model training (or directly use pretrained models):

   ```bash
   python strategy.py  
   python final.py  
   ```
5. Launch the prototype app for inference:

   ```bash
   python app.py  
   ```

   Then open `http://127.0.0.1:5000` (or whichever port) in your browser to input patient data and get predictions with explainability visuals.

## Project Structure

```
AI-Driven-Early-Detection-of-Pancreatic-Cancer/
│
├── patient_data/             # Clinical, demographic, imaging datasets
├── test_data/                # Sample input data for inference/testing
├── models/                   # Pretrained model files
│   ├── random_forest_pancreatic_model.pkl  
│   └── pancreas_cnn_model.h5  
├── __pycache__/              
├── app.py                    # Application for demo/prediction & explainability  
├── final.py                  # Final pipeline script  
├── final2.py                 # Variant or next version  
├── new.py                    # Experimental script  
├── proto.py                  # Prototype explainability script  
├── protos.py                 # Alternate prototype script  
├── prototype.py              # Another prototype  
└── strategy.py               # Data processing / modeling strategy script  
```

## How to Contribute

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`
3. Make your changes and commit: `git commit -m "Add <feature>"`
4. Push to your fork: `git push origin feature/YourFeatureName`
5. Open a Pull Request to the main repo.
6. Ensure your code follows PEP8 standards, includes comments, and passes any tests you add.

## Future Work

* Expand dataset to include larger multicenter cohorts and more imaging modalities (CT/MRI/Ultrasound).
* Integrate other AI techniques (e.g., Transformers for imaging, multimodal fusion models).
* Develop a full web interface/dashboard for visualizing explainability results.
* Implement continuous learning/online training with incoming patient data.
* Deploy as a web service or mobile app with secure backend and API endpoints.
* Validation/benchmarking with clinical collaborators, regulatory compliance and usability studies.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this work under conditions stated in the license.

## Author / Contact

**Jeremiah Ranen R**
Bachelor’s in AI & Data Science — KPR Institute of Engineering and Technology, Coimbatore.

**Ram Eshuwar Parimalam K.P.C**
Bachelor’s in AI & Data Science — KPR Institute of Engineering and Technology, Coimbatore.

**Ankush Sivankutty**
Bachelor’s in Biomedical Engineering — KPR Institute of Engineering and Technology, Coimbatore.

---

**Thank you for exploring this project!**
If you use or build upon it, please ⭐ the repo and share feedback/issues.

