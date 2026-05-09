# Pulsar Star Detection using Machine Learning & Deep Learning

An end-to-end machine learning project for detecting pulsar stars from radio signal data using advanced classification techniques, imbalance handling, and deep learning.

This project compares traditional gradient boosting models with neural networks to build a highly accurate pulsar classification system capable of handling imbalanced astronomical datasets.

## Project Overview

Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation. Detecting pulsars is a highly challenging classification problem because pulsar signals are extremely rare compared to non-pulsar noise.

In this project, I built a complete ML pipeline that:

Cleans and preprocesses astronomical signal data, 
Handles severe class imbalance using SMOTE, 
Applies robust feature scaling, 
Trains and evaluates: 
XGBoost Classifier, 
Deep Neural Network (MLP), 
Compares model performance using classification metrics and ROC-AUC


Dataset Used: HTRU2 Pulsar Dataset

Features include statistical characteristics extracted from integrated pulse profiles and DM-SNR curves.

### Features
Mean of integrated profile
Standard deviation of integrated profile
Excess kurtosis
Skewness
DM-SNR statistical features
### Target
0 → Non-pulsar
1 → Pulsar

## Tech Stack
- Machine Learning
- Python
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Data Processing
- Pandas
- NumPy
- SMOTE (Imbalanced-learn)
- Visualization
- Matplotlib
- Seaborn


## Model Performance
XGBoost Results
Metric	Score
Accuracy	97%
ROC-AUC	0.9775
Pulsar Recall	92%
Pulsar F1-Score	84%

Deep Learning (MLP) Results
Metric	Score
Accuracy	97%
Pulsar Recall	93%
Pulsar F1-Score	87%

The neural network achieved stronger minority-class detection performance while maintaining high overall accuracy.

## Key Learnings
- Handling class imbalance is critical for rare-event detection problems
- SMOTE significantly improved recall for pulsar detection
- RobustScaler helped stabilize training against extreme outliers
- Deep learning models can outperform traditional ML models on minority class recognition
- Proper evaluation metrics matter more than accuracy in imbalanced datasets
  
## Future Improvements
- Hyperparameter tuning with Optuna
- Ensemble stacking between XGBoost and Neural Networks
- Model deployment using FastAPI
- Real-time pulsar prediction dashboard
- Explainability using SHAP values



Dataset citation - R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
