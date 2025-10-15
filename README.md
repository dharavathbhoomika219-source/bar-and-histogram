# customer demographic 
Decision Tree Classifier: Predicting Customer Purchase Behavior
📋 Project Overview
This project focuses on building a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.
The dataset used is the Bank Marketing Dataset from the UCI Machine Learning Repository
By analyzing customer attributes such as age, job, education, marital status, and previous campaign outcomes, this model helps identify potential buyers — a valuable insight for marketing and sales strategy optimization.
🎯 Objective:
The main goal is to:
Predict whether a customer will subscribe to a term deposit (yes or no).Explore data preprocessing, feature selection, and model evaluation techniques.Gain insights into the factors influencing customer purchasing decisions.
Dataset Description:
Source: Bank Marketing Data Set - UCI Machine Learning Repository
The dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution.
Key Features
| Feature   | Description                                                             |
| --------- | ----------------------------------------------------------------------- |
| age       | Age of the customer                                                     |
| job       | Type of job (e.g., admin, technician, blue-collar)                      |
| marital   | Marital status (e.g., single, married, divorced)                        |
| education | Education level                                                         |
| default   | Has credit in default?                                                  |
| balance   | Average yearly balance (in euros)                                       |
| housing   | Has a housing loan?                                                     |
| loan      | Has a personal loan?                                                    |
| contact   | Contact communication type                                              |
| month     | Last contact month of year                                              |
| duration  | Last contact duration, in seconds                                       |
| campaign  | Number of contacts performed during this campaign                       |
| previous  | Number of contacts performed before this campaign                       |
| poutcome  | Outcome of the previous marketing campaign                              |
| **y**     | Target variable: has the client subscribed a term deposit? (`yes`/`no`) |
Technologies Used:
Python 3.x
Pandas – Data manipulation and analysis
NumPy – Numerical computing
Matplotlib / Seaborn – Data visualization
Scikit-learn – Model building and evaluation
🚀 Project Workflow:
Importing Libraries and Dataset
Load the Bank Marketing dataset using Pandas.
Data Exploration & Cleaning
Handle missing values.
Explore categorical and numerical variables.
Visualize feature distributions and relationships.
Data Preprocessing:
Encode categorical variables using One-Hot Encoding.
Split data into training and testing sets.
Apply feature scaling if required.
Model Building
Train a Decision Tree Classifier using Scikit-learn.
Tune hyperparameters (e.g., max_depth, min_samples_split).
Model Evaluation
Evaluate performance using metrics such as:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Visualize the decision tree structure.
Insights
Identify key factors influencing customer purchase decisions.
Generate feature importance rankings.
📊 Results:
The Decision Tree model achieved strong predictive accuracy.
Feature importance analysis showed that contact duration, previous campaign outcomes, and customer age were among the top predictors.
The model can help marketing teams target customers more effectively.
🧠 Future Improvements:
Implement ensemble models such as Random Forests or Gradient Boosting.
Use cross-validation for more robust performance estimates.
Optimize hyperparameters using GridSearchCV.
Compare results with other classification algorithms (e.g., Logistic Regression, SVM).
🗂️ Repository Structure:
├── data/
│   └── bank.csv
├── notebooks/
│   └── Decision_Tree_Classifier.ipynb
├── src/
│   └── model.py
├── README.md
├── requirements.txt
└── results/
    └── confusion_matrix.png
💻 How to Run:
Clone this repository:
git clone https://github.com/your-username/decision-tree-customer-purchase.git
cd decision-tree-customer-purchase
Install dependencies:
pip install -r requirements.txt
Run the Jupyter notebook:
jupyter notebook notebooks/Decision_Tree_Classifier.ipynb
🏆 Acknowledgments
Dataset: UCI Machine Learning Repository – Bank Marketing Data Set
Scikit-learn documentation for Decision Tree Classifier examples
📧 Contact
Created by Dharavath Bhoomika
For questions or suggestions, feel free to reach out via dharavathbhoomika219@gmail.com
