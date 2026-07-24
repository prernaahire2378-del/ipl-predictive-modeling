# IPL Match Winner Prediction – Predictive Modeling

## 1. Project Overview

This project focuses on building a machine learning classification model to predict the winning IPL team based on match-related features.

The main goal is to compare different machine learning models, evaluate their performance using Macro F1-score, improve the best-performing model through hyperparameter tuning, and validate its performance using cross-validation.

---

## 2. Business Question

Can we predict the winning IPL team using historical match-related information and machine learning techniques?

This project aims to identify the model that performs best at classifying IPL match winners across multiple teams.

---

## 3. Problem Type

This is a **Multiclass Classification** problem because the target variable `winner` contains multiple IPL team categories.

### Target Variable (y)
- `winner`

### Features (X)
The remaining relevant match-related columns used for prediction.

The target column `winner` was separated from the input features before model training.

---

## 4. Dataset

The project uses historical IPL match data.

The dataset contains information related to IPL matches, including match and team-related features.

The target variable is:

- `winner` – The team that won the match.

The data was loaded and cleaned within the predictive modeling notebook to keep the project self-contained.

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

- Loaded the IPL dataset.
- Checked data types and missing values.
- Cleaned the dataset and prepared relevant features.
- Separated features (`X`) and target (`y`).
- Split the data into training and testing sets before fitting encoders or scalers.
- Used categorical encoding for categorical features.
- Used numerical scaling where required.
- Applied the fitted preprocessing steps to the test data using transform-only mode.
- Ensured that preprocessing was performed without data leakage.

The train-test split was performed before fitting preprocessing transformations.

---

## 6. Class Distribution and Imbalance

The target variable contains multiple IPL teams with different numbers of observations.

The class distribution was checked before model training.

The minority classes were identified, and oversampling was applied to the training data only.

### Original Training Shape

`(872, 9)`

### Resampled Training Shape

`(2185, 9)`

After oversampling, each class in the training data had 115 samples.

The test set was not oversampled to ensure that the final evaluation represented the original data distribution.

---

## 7. Models Trained

The following classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Tuned Logistic Regression
5. Logistic Regression with Oversampling

The main evaluation metric used for model comparison was **Macro F1-score** because this is a multiclass classification problem with imbalanced class distributions.

---

## 8. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Model Comparison

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5917 | 0.5555 | 0.4712 | 0.4878 |
| Decision Tree | 0.3853 | 0.2552 | 0.2314 | 0.2167 |
| Random Forest | 0.5183 | 0.4724 | 0.5229 | 0.4759 |
| Tuned Logistic Regression | 0.5183 | 0.5344 | 0.5906 | **0.5526** |
| Logistic Regression + Oversampling | 0.5780 | 0.4889 | 0.5637 | 0.5079 |

---

## 9. Cross-Validation

5-fold cross-validation was performed to evaluate the stability of the best-performing model.

### Original Logistic Regression

- Mean Macro F1: `0.3831`
- Standard Deviation: `0.0345`

### Tuned Logistic Regression

- Best Cross-Validation Macro F1: `0.4554`

The tuned model showed an improvement over the original Logistic Regression model during cross-validation.

---

## 10. Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV`.

The search was performed using a pipeline containing preprocessing and the Logistic Regression model.

The tuning process used **Macro F1-score** as the scoring metric.

### Best Parameters

- `C = 100`
- `solver = 'lbfgs'`

### Best Cross-Validation Macro F1

`0.4554`

The tuned Logistic Regression model achieved a higher Macro F1-score than the original Logistic Regression model.

---

## 11. Oversampling

Because the target classes were imbalanced, oversampling was applied to the training data only.

The purpose was to give minority classes more representation during model training.

### Logistic Regression with Oversampling

- Accuracy: `0.5780`
- Precision: `0.4889`
- Recall: `0.5637`
- Macro F1: `0.5079`

The oversampling approach improved the Macro F1-score compared with the original Logistic Regression model but did not outperform the tuned Logistic Regression model.

---

## 12. Final Model Selection

The **Tuned Logistic Regression** model was selected as the final model.

It achieved the highest test-set Macro F1-score among the evaluated models:

**Test Macro F1-score: 0.5526**

The model was selected because Macro F1-score gives equal importance to each class and is more appropriate for this multiclass classification problem with imbalanced team representation.

The tuned model also improved the Macro F1-score from `0.4878` to `0.5526` compared with the original Logistic Regression model.

---

## 13. Final Model Recommendation

The Tuned Logistic Regression model is recommended as the final model for this project because it achieved the highest Macro F1-score of `0.5526` on the test set. Hyperparameter tuning improved the model's balanced classification performance across the different IPL team classes. Although the model does not predict every team equally well, it provided the best overall Macro F1-score among the tested approaches and was therefore selected as the final model.

---

## 14. Limitations

- Some IPL teams have very few historical observations.
- Team names have changed over different IPL seasons, which creates separate categories for some teams.
- The dataset contains an imbalance between frequently occurring and less frequently occurring teams.
- The model performance may be affected by the limited number of samples for some teams.
- Historical match patterns may not always represent future IPL outcomes.

---

## 15. Conclusion

This project demonstrated a complete machine learning workflow for a multiclass classification problem.

The workflow included data preparation, train-test splitting, categorical encoding, feature scaling, model training, model comparison, cross-validation, hyperparameter tuning, oversampling, and final model selection.

Among the evaluated models, **Tuned Logistic Regression** achieved the best test Macro F1-score of **0.5526** and was selected as the final model.
