# Pulsar Classification Project

### Overview

This project focuses on classifying pulsar candidates from an imbalanced astronomical dataset using machine learning techniques. The dataset contains statistical features derived from pulsar signals and non-pulsar signals.

### Data Preprocessing and Visualization

- Assigned meaningful column names to the dataset based on official dataset descriptions to improve clarity.
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class, thus mitigating bias during model training.
- Extracted advanced features using Fourier Transform to capture frequency-domain characteristics and Wavelet Transform for time-frequency analysis on signal attributes like mean integrated profile and dispersion measure.
- Engineered additional features such as the ratio of mean to standard deviation of dispersion measures (DM_Approx).
- Visualized feature distributions before and after applying different scaling techniques including StandardScaler, MinMaxScaler, and RobustScaler to better understand feature behavior and support model training.


### Models Implemented

- Random Forest Classifier
- XGBoost Classifier
- Long Short-Term Memory (LSTM) Neural Network

Tree-based models were chosen due to their effectiveness with tabular data containing skewed statistical features, while LSTM was included for comparative purposes.

### Model Evaluation

Performance metrics included Accuracy, Precision, Recall, F1-Score, and ROC-AUC score.
Due to the inherent class imbalance, ROC-AUC and class-wise metrics were emphasized over simple accuracy.

Performance Summary
Model	Accuracy	ROC-AUC
Random Forest	~0.98	~0.97
XGBoost	~0.98	~0.98

Both tree-based models showed excellent classification capability, while LSTM underperformed due to the dataset's non-sequential nature.

### Observations

Tree-based models effectively captured nonlinear relationships and handled skewed data distributions.
LSTM models did not add predictive value given the statistical, non-temporal dataset.
Default decision thresholds were used without calibration or probability tuning.

### Conclusion and Future Work

This project highlights the applicability of classical machine learning methods to astronomical signal classification tasks. Future work may involve:

- Threshold tuning and model calibration
- Interpretability analysis with feature attribution methods
- Exploring ensemble methods or hybrid models

### Tools & Technologies

- Python (Pandas, NumPy)
- Scikit-learn
- Imbalanced-learn (SMOTE)
- PyWavelets
- XGBoost
- TensorFlow / Keras
- Matplotlib / Seaborn


Dataset citation - R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
