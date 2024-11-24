# classificationprobelm
Classification problem using Breast cancer dataset

# Breast Cancer Prediction using Machine Learning
This project demonstrates the use of various machine learning algorithms to classify breast cancer as malignant or benign using the Breast Cancer dataset. It includes steps for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment.

# Overview
Breast cancer detection is critical in improving early diagnosis and treatment outcomes. This project uses machine learning techniques to predict the likelihood of malignancy based on clinical features of tumors.

# Objective
To build and evaluate machine learning models for binary classification of breast cancer into malignant or benign categories, and identify the best-performing model.

# Dataset
The dataset used in this project is the "Breast Cancer Dataset" available from the UCI Machine Learning Repository. It contains 569 samples of tumor measurements, with 30 features each, and a target variable (`Diagnosis`).

# Data Features
- Diagnosis: Target variable (Malignant or Benign)
- Features: 30 continuous variables related to tumor properties (e.g., radius, texture, smoothness)

## Project Workflow

# 1. Data Collection
- Dataset is sourced from the UCI repository.
- It is preloaded into the `data/` directory.

# 2. Data Preprocessing
- Handling Missing Values: Verified no missing values.
- Outlier Detection: Used boxplots and z-scores to identify and remove outliers.
- Skewness Correction: Applied log transformations where necessary.

# 3. Exploratory Data Analysis (EDA)
- Visualized data using:
  - Histograms for distribution analysis
  - Boxplots to detect outliers
  - Heatmaps for feature correlation
  - Pairplots for multivariate analysis

# 4. Feature Engineering
- Encoded the categorical target variable (`Diagnosis`) using label encoding.
- Retained highly correlated features after correlation analysis.

# 5. Train-Test Split
- Splitted the dataset into training (80%) and testing (20%) sets.

# 6. Feature Scaling
- Applied Min-Max Scaling to normalize the feature values.

# 7. Model Training and Evaluation
The following classifiers were trained:
1. Logistic Regression : Logistic regression is a statistical model that predicts the probability of a binary outcome using a sigmoid function. It is a linear                               model that estimates the relationship between the features and the log odds of the target class.
                        : Breast cancer detection often involves distinguishing between two classes (malignant or benign), making logistic regression an ideal  
                          choice. It works well for linearly separable data and provides interpretable coefficients for feature importance.

2. Decision Tree Classifier : Decision trees split the data into subsets based on the most significant feature at each node. The tree grows by creating branches 
                              that maximize the information gain or minimize impurity
                            : This algorithm can handle both categorical and continuous features and works well with non-linear relationships. It is interpretable, 
                              which is crucial for understanding feature impacts in medical datasets.

4. Random Forest Classifier : Random forests are an ensemble method that combines multiple decision trees to improve prediction accuracy. It reduces overfitting by 
                              averaging the results of various trees, which are trained on random subsets of the data and features.
                            : Breast cancer datasets are often high-dimensional with numerous features. Random forests can handle this complexity effectively and 
                              are robust against overfitting, making them a strong candidate.

6. Support Vector Machine (SVM) : SVM creates a decision boundary (hyperplane) that separates the data points of different classes with the maximum margin. It 
                                  uses kernel tricks to handle non-linearly separable data.
                                : Breast cancer datasets might not always have a clear linear separation. SVM can effectively handle this with kernels like RBF or 
                                  polynomial, making it suitable for high-dimensional and complex datasets.

8. k-Nearest Neighbors (k-NN) : k-NN classifies a data point based on the majority class of its k nearest neighbors. The distance metric determines "nearness."
                              : k-NN is simple and intuitive, performing well on smaller datasets where computation cost is not an issue. However, it can be   
                                sensitive to noise and feature scaling, requiring careful preprocessing.

# Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

# 8. Best Model Selection
The Random Forest Classifier achieved the highest performance with:
- Accuracy: 96.7%
- F1-Score: 96.5%


