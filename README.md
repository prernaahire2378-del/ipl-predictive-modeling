# IPL Match Winner Prediction – Predictive Modeling

## 1. Project Overview

This project focuses on building a machine learning classification model to predict the winning IPL team using historical match-related data.

The main objective is to compare multiple machine learning models, evaluate their performance using suitable classification metrics, handle class imbalance, perform hyperparameter tuning, and select the best-performing model based on the Macro F1-score.

The project follows an end-to-end machine learning workflow, including data preparation, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, oversampling, and final model selection.

---

## 2. Business Question

Can we predict the winning IPL team using historical match-related information and machine learning techniques?

The objective is to identify a machine learning model that can classify the winning team across multiple IPL team categories while maintaining balanced performance across both majority and minority classes.

---

## 3. Problem Type

This project is a **Multiclass Classification** problem because the target variable `winner` contains multiple IPL team categories.

### Target Variable (y)

- `winner`

The target variable represents the IPL team that won the match.

### Features (X)

The following input features were used for prediction:

- `season`
- `city`
- `match_type`
- `venue`
- `team1`
- `team2`
- `toss_winner`
- `toss_decision`
- `super_over`

The target column `winner` was separated from the input features before model training.

---

## 4. Dataset

The project uses historical IPL match data containing information related to IPL matches, teams, venues, seasons, toss decisions, and match outcomes.

The target variable used for prediction is:

- `winner` – The team that won the match.

The data loading and preprocessing steps are included within the predictive modeling notebook to keep the project self-contained.

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

- Loaded the IPL dataset.
- Checked the dataset structure and data types.
- Checked for missing values.
- Cleaned and prepared the relevant features.
- Separated input features (`X`) and target variable (`y`).
- Split the data into training and testing sets.
- Performed preprocessing after the train-test split to avoid data leakage.
- Applied categorical encoding to categorical features.
- Applied feature scaling where required.
- Fitted preprocessing transformations on the training data only.
- Applied the fitted transformations to the test data using transform-only mode.

The preprocessing workflow was designed to ensure that information from the test set was not used during model training.

---

## 6. Train-Test Split

The dataset was divided into training and testing subsets before fitting the preprocessing transformations.

The training data was used for:

- Model training
- Cross-validation
- Hyperparameter tuning
- Oversampling

The test data was kept separate and was used only for final model evaluation.

This approach helps provide a more reliable estimate of how the final model performs on unseen data.

---

## 7. Class Distribution and Imbalance

The target variable contains multiple IPL teams with different numbers of observations.

The class distribution was analyzed before model training to identify potential class imbalance.

Some teams had significantly fewer observations than others. Therefore, Random Oversampling was evaluated as a method to provide minority classes with greater representation during model training.

Oversampling was applied only to the training data.

### Original Training Shape

`(872, 9)`

### Resampled Training Shape

`(2185, 9)`

After oversampling, each target class in the training data had 115 samples.

The test set remained untouched and was not oversampled so that final evaluation was performed on the original test distribution.

---

## 8. Models Trained

The following classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Tuned Logistic Regression
5. Logistic Regression with Random Oversampling

The models were evaluated using multiple classification metrics.

The primary metric used for model selection was **Macro F1-score** because this is a multiclass classification problem with imbalanced class distributions.

---

## 9. Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

---

## 10. Model Comparison

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5917 | 0.5555 | 0.4712 | 0.4878 |
| Decision Tree | 0.3853 | 0.2552 | 0.2314 | 0.2167 |
| Random Forest | 0.5183 | 0.4724 | 0.5229 | 0.4759 |
| Tuned Logistic Regression | 0.5183 | 0.5344 | 0.5906 | **0.5526** |
| Logistic Regression + Oversampling | 0.5780 | 0.4889 | 0.5637 | 0.5079 |

The Tuned Logistic Regression model achieved the highest Macro F1-score.

---

## 11. Cross-Validation

5-fold cross-validation was performed.

### Original Logistic Regression

- Mean Macro F1: `0.3831`
- Standard Deviation: `0.0345`

### Tuned Logistic Regression

- Best Cross-Validation Macro F1: `0.4554`

---

## 12. Hyperparameter Tuning

GridSearchCV was used.

### Best Parameters

- `C = 100`
- `solver = 'lbfgs'`

### Best Cross-Validation Macro F1

`0.4554`

---

## 13. Oversampling

Random Oversampling was applied only on the training data.

### Logistic Regression + Oversampling

- Accuracy: `0.5780`
- Precision: `0.4889`
- Recall: `0.5637`
- Macro F1: `0.5079`

---

## 14. Final Classification Report

### Tuned Logistic Regression

- Accuracy: `0.5183`
- Macro Precision: `0.5344`
- Macro Recall: `0.5906`
- Macro F1: `0.5526`

---

## 15. Final Confusion Matrix

A confusion matrix was generated to visualize the classification performance of the final model and identify correctly classified and misclassified IPL teams.

---

## 16. Final Model Selection

The **Tuned Logistic Regression** model was selected because it achieved the highest Macro F1-score among all evaluated models.

---

## 17. Final Model Recommendation

The Tuned Logistic Regression model is recommended as the final predictive model for IPL winner prediction because it provides the most balanced performance across all classes.

---

## 18. Results Summary

| Metric | Value |
|---------|------:|
| Accuracy | 51.83% |
| Macro Precision | 53.44% |
| Macro Recall | 59.06% |
| Macro F1-score | **55.26%** |

### Final Model

**Tuned Logistic Regression**

---

## 19. Key Findings

- Logistic Regression provided a strong baseline.
- Hyperparameter tuning improved Macro F1-score.
- Oversampling improved minority class prediction.
- Decision Tree showed the weakest performance.
- Tuned Logistic Regression achieved the best overall balanced performance.

---

## 20. Limitations

- Some IPL teams have fewer historical observations.
- Class imbalance affects prediction quality.
- Historical match data cannot perfectly predict future outcomes.
- Additional player statistics and match conditions could further improve performance.

---

## 21. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Google Colab
- GitHub

---

## 22. Project Structure

```text
ipl-predictive-modeling/
│
├── IPL_Predictive_Modeling.ipynb
├── requirements.txt
└── README.md
```

---

## 23. Reproducibility

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook.
4. Run all cells sequentially.

---

## 24. Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting IPL match winners using multiclass classification.

Different machine learning models were compared, cross-validation and hyperparameter tuning were performed, and class imbalance was addressed using Random Oversampling.

Among all evaluated models, the **Tuned Logistic Regression** model achieved the highest **Macro F1-score (0.5526)** and was selected as the final recommended model for predicting IPL match winners.
