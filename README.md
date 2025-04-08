# Heart Disease Prediction Using Machine Learning

## Overview
This project investigates the application of machine learning techniques for early detection of heart disease using clinical data. The study leverages a dataset of 303 individuals with 14 attributes to predict the presence or absence of heart disease. Various machine learning algorithms, including Random Forest, Logistic Regression, Naïve Bayes, XGBoost, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN), are implemented and evaluated based on metrics such as accuracy and F1-score.

The goal is to provide insights into the potential of machine learning for precise heart disease detection, aiding clinical decision-making and personalized treatment planning.

## Authors
- **Abhishek Vardhanapu** - Department of Computer Science and Engineering, SRM University – AP, Andhra Pradesh, India  
  Email: abhishek_vardhanapu@srmap.edu.in
- **Mohammad Fayaz Shaik** - Department of Computer Science and Engineering, SRM University – AP, Andhra Pradesh, India

## Abstract
The foremost reason behind mortality and hospital admittance across the globe is heart disease. Thanks to the advancements in technology and computer engineering, the identification and swift, efficient treatment of heart conditions have become feasible. This research paper investigates the role of machine learning in early heart disease detection using diverse medical data. The study employs algorithms like Random Forest, Logistic Regression, Naïve Bayes, XGBoost, Support Vector Machines, and K-Nearest Neighbors on medical records to predict heart disease based on clinical attributes. Performance evaluation includes accuracy, F1 score, and feature importance analysis. Results underscore machine learning's potential for precise heart disease detection, aiding clinical decisions and personalized treatment.

**Keywords**: Random Forest, Naïve Bayes, Support Vector Machine, Logistic Regression, XGBoost, K-Nearest Neighbors, F1 score.

Within the dataset, there exists information from 303 individuals, with each individual characterized by 14 pertinent attributes. Additionally, a target variable is included, denoting the presence or absence of heart disease. Through this work, we aim to contribute to advances in heart disease diagnosis using machine learning techniques by providing a better insight of the potential of algorithms for personal early diagnosis and treatment planning [6]. The results of our study may have implications for healthcare professionals in optimizing decision-making processes and, ultimately, reducing the burden of heart disease on global health.

## I. Introduction
Heart disease continues to be a leading cause of mortality worldwide [1,3], posing a significant public health challenge. Early and accurate detection of heart disease is crucial for timely medical intervention and improved patient outcomes. In recent years, machine learning techniques have shown great potential in aiding healthcare [2], especially in diagnosing heart disease. In this research paper, we present a comprehensive study on heart disease detection using a machine learning approach, specifically employing various Machine Learning algorithms [4,5] with a dataset obtained from Kaggle.

## II. Literature Review
In the paper [7], the objective was to create a predictive framework for identifying heart disease through the utilization of machine learning methods, employing a range of parameters associated with cardiac health. After preprocessing the dataset, three distinct supervised machine learning classification algorithms namely, the J48 Classifier, Naive Bayes, and Random Forest were employed using the Weka machine learning software. The outcomes yielded by the random forest classifier and its associated confusion matrix demonstrate the durability of the approach. The findings of this investigation could potentially serve as an auxiliary instrument for cardiologists in establishing a trustworthy heart disease diagnosis. A. Kumari and A.K. Mehta [8] aimed to forecast cardiovascular disease through the implementation of seven distinct machine learning algorithms. Furthermore, they sought to enhance the precision of underperforming algorithms by employing ensemble techniques like AdaBoost and the voting ensemble approach. An efficient prediction system was created by [9] which can predict stroke and heart-related diseases accurately as well as determining the severity level of the disease with a shorter computation time.

In [10], a comparison was drawn between the decision tree and multilayer perceptron approaches. These techniques outlined in this study offer support to healthcare experts in gauging risk probabilities. With its cost-efficient nature, the suggested system stands as a viable substitute for the current framework. In the study by [11,12], the authors proposed a model for heart disease prediction and imminent heart disease identification using algorithms such as Logistic Regression, SVM, Multinomial Naive Bayes, Random Forest, and Decision Tree. With the help of multimodal strategy, they were able to achieve higher accuracy while reducing processing time. Within the publication [13] from Seema G., Kalpna G., and Nitin G., they employed the algorithms most in demand on the Cleveland Heart Disease dataset. The ensuing outcomes were juxtaposed to ascertain the most fitting algorithm for identifying and classifying coronary heart disease. The simulation outcomes underscore that, among the algorithms tested, the Naive Bayes algorithm outperforms in terms of accuracy for forecasting coronary disease based on the heart disease dataset.

The objective of the study outlined in [14] involved the comparison and assessment of six distinct machine learning algorithms to achieve optimal outcomes in predicting heart conditions. Through this examination, it was determined that the Xgboost method exhibited a 91.3% increase in accuracy when predicting heart disease. Notably, for the prediction of heart disease, the process of feature selection holds the potential to yield more valuable attributes and ultimately lead to more effective results. By [15,16], an automated mechanism was developed with the intention of early-stage heart disease prognosis. As a result, medical practitioners could enhance their diagnostic precision, and individuals could proactively monitor their health issues via this automated framework. Their findings indicated that for improved heart disease prediction outcomes, a preferable approach for the future would involve the application of search algorithms to select features, followed by the utilization of machine learning methods for prediction.

## III. Dataset
The dataset utilized in this study has been obtained from Kaggle, a widely recognized platform renowned for its role in disseminating and uncovering datasets tailored for diverse machine learning and data science undertakings. This dataset contains information from 303 individuals and encompasses 14 attributes, each playing a crucial role in heart disease detection.

**Table 1. Attributes and their descriptions**  
*(Note: The specific table is not included here; refer to the original paper `HEART_DISEASE_PREDICTION.pdf` for details.)*

This dataset provides a comprehensive set of attributes that capture essential information related to heart health and risk factors for heart disease. Each record represents a unique individual, and the target variable serves as the ground truth label for the presence or absence of heart disease.

## IV. Proposed Methodology
The research aims to develop an efficient heart disease detection model using different machine learning algorithms. The methodology involves the following key steps:

**Fig.1. Prediction Process**  
*(Refer to the paper for the figure.)*

1. **Dataset Acquisition and Exploration**: Obtain the heart disease dataset from Kaggle, encompassing data of 303 individuals and comprising 14 attributes, the dataset also incorporates a target variable that signifies the existence or nonexistence of heart disease.
2. **Data Preprocessing**: Perform data preprocessing to handle missing values, outliers, and noisy data. Ensure the dataset's cleanliness and reliability to avoid biases in the model training process.
3. **Feature Selection using Correlation**: Perform a correlation analysis to identify the key factors that strongly affect the detection of heart disease. Choose the informative attributes with the aim of diminishing dimensionality and enhancing the model's effectiveness.
4. **Dataset Split**: Utilize the sklearn library to partition the preprocessed dataset into a training set, accounting for 75% of the data, and a test set, encompassing 25% of the data. This split ensures that the model learns from a substantial amount of data while being evaluated on unseen instances to assess its generalization ability.
5. **Model Selection and Implementation**: Incorporate a range of machine learning algorithms into the implementation, encompassing Random Forest, Logistic Regression, Naïve Bayes, XGBoost, Support Vector Machines, and K-Nearest Neighbors. Tailor the configuration of each algorithm to their individual parameters and preferences.
6. **Model Training and Evaluation**: Individually train every algorithm using the training set and subsequently assess their efficacy on the distinct test set. Calculate essential performance measures such as accuracy and F1-score, enabling the quantification of each algorithm's competence in accurately categorizing instances denoting the presence or absence of heart disease.
7. **Results Analysis and Comparison**: Analyze and compare the accuracy and F1-score achieved by each algorithm. Interpret the results to determine which algorithm demonstrates the highest predictive power for heart disease detection on this specific dataset. Consider not only the individual scores but also the trade-offs between precision and specificity.

## V. Results
The target variable indicates whether heart disease is present or not (1: present, 0: absent). We visualized the distribution using a countplot, which showed the proportion of heart disease cases compared to non-heart disease cases. If there was a class imbalance, we addressed it using appropriate techniques. The countplot results provided insights into the dataset's class distribution and its potential impact on the model's performance.

**Fig. 2. No. of people with and without disease**  
*(Refer to the paper for the figure.)*

Once the correlation analysis was applied to the dataset, we generated a heatmap to visually represent the correlations among various attributes. In the heatmap, brighter colors depict positive correlations, whereas darker colors indicate negative correlations. A correlation coefficient of 1 indicates an absolute positive correlation, whereas -1 indicates an absolute negative correlation. A coefficient of 0 signifies the absence of correlation between the attributes. By examining the heatmap, we acquire a better understanding of the interactions between attributes within the dataset. This understanding is instrumental in subsequent phases such as model training and employing varied approaches for the detection of heart disease.

**Fig. 3. Correlation Matrix**  
*(Refer to the paper for the figure.)*

After conducting the correlation analysis, the subsequent stage in the data exploration process entailed generating histograms for all attributes present within the dataset. These histograms, serving as graphical representations, provide a clear depiction of the distribution of values for each attribute. This visualization offers valuable insights into the extent of spread, concentration, and frequency of occurrences exhibited by the data.

A histogram was created for each of the attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) to provide a visual representation of their respective data distributions. In these histograms, the x-axis portrays the attribute's value range, while the y-axis illustrates the frequency of occurrences within each specific value range.

**Fig. 4. Histograms**  
*(Refer to the paper for the figure.)*

Upon completing the training phase using the 75% portion of the training set and subsequently subjecting the model to an evaluation on the separate 25% test set, an analysis of the model's performance can be conducted to gauge its predictive capabilities accurately, we obtained confusion matrices as follows:

a) **K-Nearest Neighbors** –  
**Fig. 5. Confusion Matrix of KNN**  
The accuracy score for the K-Nearest Neighbors (KNN) model in heart disease detection was calculated as follows:  
= 54 / 76 ≈ 0.71 or 71%

b) **Logistic Regression** –  
**Fig. 6. Confusion Matrix of LR**  
The accuracy score for the Logistic Regression (LR) model in heart disease detection was calculated as follows:  
= (34 + 30) / (34 + 30 + 5 + 7) = 0.841

c) **Random Forest** –  
**Fig. 7. Confusion Matrix of RF**  
The accuracy score for the Random Forest (RF) model in heart disease detection was calculated as follows:  
= (33 + 29) / (33 + 29 + 6 + 8) ≈ 0.803

d) **Support Vector Machine** –  
**Fig. 8. Confusion Matrix of SVM**  
The accuracy score for the Support Vector Machine (SVM) model in heart disease detection was calculated as follows:  
= (33 + 17) / (33 + 17 + 18 + 8) ≈ 0.617

e) **Naïve Bayes** –  
**Fig. 9. Confusion Matrix of NB**  
The accuracy score for the Naïve Bayes (NB) model in heart disease detection was calculated as follows:  
= (30 + 29) / (30 + 29 + 6 + 11) ≈ 0.791

f) **XGBoost** –  
**Fig. 10. Confusion Matrix of XGBoost**  
The accuracy score for the XGBoost model in heart disease detection was calculated as follows:  
= (32 + 28) / (32 + 28 + 7 + 9) ≈ 0.791

In addition to calculating accuracy, we also derived the F1-score, precision, and specificity using the following formulas:  
*(Note: Formulas and detailed calculations are in the paper.)*

After calculating we organized and presented these values in a table as follows:  
a) Training set – 75% and Test set – 25%  
b) Training set – 80% and Test set – 20%  
c) Training set – 85% and Test set – 15%  
d) Training set – 90% and Test set – 10%  
e) Training set – 70% and Test set – 30%

## VI. Conclusion
In conclusion, our study has provided valuable insights into the application of various machine learning algorithms for heart disease detection. We conducted extensive experiments using different training and testing set ratios, including 70% training and 30% testing, 80% training and 20% testing, 90% training and 10% testing, and 85% training and 15% testing, 75% training and 25% testing to evaluate algorithm performance across a range of scenarios.

Our research underscores the paramount importance of selecting the most appropriate algorithm for heart disease detection, recognizing that each algorithm has its unique strengths and limitations. The decision regarding this is critical for optimizing predictive performance.

## Dataset Details
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets) and contains:
- 303 individuals
- 14 attributes (e.g., age, sex, cholesterol levels, etc.)
- Target variable: 1 (heart disease present) or 0 (absent)

For a detailed description of attributes, refer to the research paper (`HEART_DISEASE_PREDICTION.pdf`).

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `xgboost`

Install dependencies using:
```bash
pip install -r requirements.txt
