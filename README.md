📊 Bank Marketing Campaign – Machine Learning Project
Overview

This project applies machine learning techniques to predict whether a customer will subscribe to a bank term deposit, based on data from real marketing campaigns.
The goal is to build an end-to-end ML pipeline that not only achieves strong predictive performance, but also provides actionable business insights.

The project covers data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and business recommendations.

Dataset

Source: UCI Machine Learning Repository – Bank Marketing Dataset

Records: 41,188 customers

Target variable:

y → whether the customer subscribed to a term deposit (yes/no)

The dataset contains demographic, financial, and campaign-related features such as age, job, balance, contact method, and previous campaign outcomes.

Project Structure
bank-marketing-ml/

│── data/

│   ├── bank-full.csv

│   └── processed/

│       ├── X_train.csv

│       ├── X_test.csv

│       ├── y_train.csv

│       ├── y_test.csv

│       └── high_probability_customers.csv

│── 01_data_cleaning_and_eda.ipynb

│── 02_feature_engineering.ipynb

│── 03_machine_learning_models.ipynb

│── 04_insights_and_recommendations.ipynb

│── models/

│   └── best_model.joblib

│── visuals/

│   ├── term_deposit_distribution.png

│   ├── roc_curves_comparison.png

│   ├── feature_importance.png

│   └── confusion_matrix_logistic.png

│── README.md


Methodology
1. Data Cleaning & Exploratory Data Analysis

Verified absence of missing values

Analyzed target distribution and class imbalance

Explored categorical and numerical feature relationships

Generated visual insights for business understanding

2. Feature Engineering

Removed duration to prevent data leakage, as it is only known after the call ends

One-hot encoded categorical variables

Scaled numerical features

Applied stratified train/test split due to class imbalance

3. Machine Learning Models

The following models were trained and compared:

Logistic Regression (baseline)

Random Forest

Gradient Boosting (final selected model)

Evaluation metrics focused on:

Precision, Recall, F1-score

ROC-AUC (primary metric due to class imbalance)

4. Model Selection

Gradient Boosting was selected as the final model, achieving the best balance between recall and ROC-AUC while maintaining stable performance on the minority class.

Key Insights

Customers contacted via cellular channels show higher subscription probability.

A successful previous campaign strongly increases the likelihood of subscription.

Balance and campaign-related variables play a significant role in customer decisions.

Certain job and education segments respond better to marketing efforts.

Business Recommendations

Target customers with a predicted subscription probability above 70%.

Prioritize cellular contact methods for high-value segments.

Reduce repeated contact attempts for customers with low predicted probability.

Use predictive scoring to optimise marketing costs and increase conversion rates.

Tools & Technologies

Python (pandas, numpy)

scikit-learn

matplotlib & seaborn

Jupyter Notebook

Git & GitHub

Author

Marta Valero
Data Analysis & Machine Learning
Focused on practical, business-driven data solutions
