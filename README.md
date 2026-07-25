# 🏏 IPL Match Winner Prediction – Predictive Modeling Pipeline

## 📌 Project Overview

This project develops and evaluates a complete machine learning pipeline to predict the winning team in an Indian Premier League (IPL) match using historical IPL match data. The project follows an end-to-end predictive modeling workflow, beginning with business problem definition and concluding with model selection and evaluation.

The primary objective is to compare multiple machine learning classification algorithms and identify the model that provides the best balanced predictive performance across all IPL teams.

The project includes:

- Business problem framing
- Data understanding and preprocessing
- Feature engineering and preparation
- Train-test splitting
- Categorical encoding
- Numerical feature scaling
- Model training
- Model evaluation
- Confusion matrix analysis
- Class imbalance analysis
- Oversampling
- Cross-validation
- Hyperparameter tuning
- Final model recommendation

---

# 📊 Business Problem

Can historical IPL match information be used to accurately predict the winning team of an IPL match?

The project formulates this as a supervised multiclass classification problem where the objective is to predict the match winner using historical match-related information.

---

# 🎯 Problem Type

**Machine Learning Task:** Multiclass Classification

The target variable contains multiple IPL teams, making this a multiclass classification problem rather than binary classification.

---

# 📂 Dataset

The project uses the historical IPL **matches.csv** dataset.

The dataset contains match-level information including:

- Season
- City
- Match Date
- Match Type
- Team 1
- Team 2
- Toss Winner
- Toss Decision
- Venue
- Result
- Winning Team
- Other match-related information

---

# 🎯 Target Variable

```text
winner
```

The **winner** column represents the IPL team that won the match.

---

# 📥 Feature Variables

The remaining selected columns from `matches.csv` are used as predictor variables.

Examples include:

- Season
- City
- Team1
- Team2
- Toss Winner
- Toss Decision
- Venue
- Match Type
- Other relevant match information

The target variable was separated from the input features before model training.

---

# ⚙️ Data Preparation

The following preprocessing steps were performed before training the models:

- Loaded the IPL dataset
- Inspected dataset structure
- Checked data types
- Checked missing values
- Cleaned invalid records
- Selected relevant features
- Removed unnecessary columns
- Separated target and predictors
- Identified categorical and numerical variables
- Prepared features for machine learning

The preprocessing workflow was designed to prevent data leakage.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets before any preprocessing transformations.

This ensures that:

- Encoders are fitted only on training data.
- Scalers are fitted only on training data.
- Oversampling is performed only on training data.
- Test data remains completely unseen during model training.

This approach prevents information leakage from the testing data.

---

# 🔤 Categorical Encoding

Since machine learning algorithms require numerical input, categorical variables were encoded before model training.

The encoder was:

- Fitted only on training data
- Applied to testing data using transform only
- Applied consistently after oversampling

No information from the testing dataset was used while learning the encoding.

---

# 📏 Numerical Feature Scaling

Numerical features were standardized for models that are sensitive to feature magnitude.

Scaling was performed by:

- Fitting the scaler only on training data
- Applying the fitted scaler to testing data

This ensures that no information from the testing dataset influences the preprocessing process.

---

# ⚖️ Class Distribution

The distribution of IPL winners was examined before training.

The dataset contains multiple IPL teams with unequal numbers of observations.

Some franchises have significantly more historical wins than others, creating class imbalance.

Because of this imbalance, **Macro F1-score** was selected as the primary evaluation metric.

Macro F1 gives equal importance to every IPL team regardless of how many matches they have won historically.

---

# 🔄 Class Rebalancing Using Oversampling

To improve minority class representation, oversampling was performed **only on the training dataset**.

The testing dataset was left unchanged to preserve realistic model evaluation.

### Original Training Shape

```text
(872, 9)
```

### Resampled Training Shape

```text
(2185, 9)
```

Each IPL team contained **115 samples** after oversampling.

### Encoded Training Shape

```text
(2185, 181)
```

### Encoded Testing Shape

```text
(218, 181)
```

Oversampling improved minority class representation during model training.

---

# 📈 Primary Evaluation Metric

The primary metric used throughout the project is:

## **Macro F1-score**

Macro F1-score is preferred because:

- The problem is multiclass.
- The classes are imbalanced.
- Each IPL team is treated equally regardless of frequency.

Additional reported metrics include:

- Accuracy
- Precision
- Recall
- Confusion Matrix

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Tuned Logistic Regression
5. Logistic Regression with Oversampling

---

# 📊 Model Evaluation

Each model was evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

Macro F1-score was used as the primary model selection metric.

---

# 📋 Model Comparison

| Model | Accuracy | Precision | Recall | Macro F1 |
|--------|---------:|----------:|--------:|----------:|
| Logistic Regression | 0.5917 | 0.5555 | 0.4712 | 0.4878 |
| Decision Tree | 0.3853 | 0.2552 | 0.2314 | 0.2167 |
| Random Forest | 0.5183 | 0.4724 | 0.5229 | 0.4759 |
| Tuned Logistic Regression | 0.5183 | 0.5344 | 0.5906 | **0.5526** |
| Logistic Regression + Oversampling | 0.5780 | 0.4889 | 0.5637 | 0.5079 |

---

# 🏆 Model Ranking

| Rank | Model | Macro F1 |
|------|------------------------------|----------:|
| 1 | Tuned Logistic Regression | **0.5526** |
| 2 | Logistic Regression + Oversampling | 0.5079 |
| 3 | Logistic Regression | 0.4878 |
| 4 | Random Forest | 0.4759 |
| 5 | Decision Tree | 0.2167 |

The tuned Logistic Regression model achieved the best balanced predictive performance.

---

# 🔍 Confusion Matrix Analysis

A confusion matrix was generated for the final tuned Logistic Regression model.

The confusion matrix compares:

- Actual winning teams
- Predicted winning teams

Diagonal entries represent correctly classified matches, while off-diagonal values represent incorrect predictions.

The analysis indicates that teams with larger historical datasets are predicted more accurately than teams with fewer observations.

---

# 📑 Final Classification Report

| Metric | Score |
|---------|------:|
| Accuracy | 0.5183 |
| Macro Precision | 0.5344 |
| Macro Recall | 0.5906 |
| Macro F1 | **0.5526** |

---

# 🔄 Cross-Validation

Five-fold cross-validation was performed using Macro F1-score.

## Original Logistic Regression

Mean Macro F1

```text
0.3831
```

Standard Deviation

```text
0.0345
```

## Tuned Logistic Regression

Best Cross-Validation Macro F1

```text
0.4554
```

Test Macro F1

```text
0.5526
```

The tuned model demonstrated improved generalization compared with the baseline Logistic Regression.

---

# 🔗 Pipeline for Cross-Validation

A Scikit-learn Pipeline was used during cross-validation and hyperparameter tuning.

The pipeline ensured that:

- Encoding occurred separately within each fold.
- Scaling occurred separately within each fold.
- No preprocessing information leaked between folds.

This resulted in a more reliable estimation of model performance.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter optimization was performed using **GridSearchCV**.

### Parameters Searched

- C
- solver

### Optimization Metric

Macro F1-score

### Best Parameters

```text
C = 100
solver = lbfgs
```

### Best Cross-Validation Score

```text
0.4554
```

Hyperparameter tuning improved the overall balanced classification performance.

---

# 🔄 Oversampling Results

Oversampling was applied only to the training data.

### Logistic Regression + Oversampling

| Metric | Score |
|---------|------:|
| Accuracy | 0.5780 |
| Precision | 0.4889 |
| Recall | 0.5637 |
| Macro F1 | 0.5079 |

Oversampling improved the original Logistic Regression model but did not outperform the tuned Logistic Regression model.

---

# ✅ Final Model Selection

The **Tuned Logistic Regression** model was selected as the final model because it achieved the highest Macro F1-score among all evaluated models.

### Final Model

**Tuned Logistic Regression**

### Test Macro F1-score

```text
0.5526
```

---

# 🚀 Final Recommendation

Based on the experimental results, the Tuned Logistic Regression model is recommended for deployment.

Reasons include:

- Highest Macro F1-score
- Better balanced classification across IPL teams
- Improved performance after hyperparameter tuning
- Better overall generalization compared with the baseline models

Although oversampling improved the baseline Logistic Regression model, it did not surpass the tuned model.

---

# ⚠️ Project Limitations

- Some IPL teams have relatively few historical observations.
- Team names have changed across IPL seasons.
- Historical franchises are treated as separate categories.
- Target classes are imbalanced.
- Minority classes remain challenging to classify accurately.
- Historical trends may not perfectly represent future IPL matches.
- The model should be considered a predictive analytics exercise rather than a guarantee of future match outcomes.

---

# 📝 Conclusion

This project demonstrates a complete end-to-end machine learning workflow for multiclass classification using historical IPL match data.

The workflow includes:

- Business problem definition
- Data preparation
- Feature selection
- Train-test splitting
- Categorical encoding
- Numerical feature scaling
- Model training
- Model evaluation
- Confusion matrix analysis
- Class imbalance analysis
- Oversampling
- Cross-validation
- Pipeline-based preprocessing
- Hyperparameter tuning
- Final model selection

Five machine learning approaches were evaluated.

Among all evaluated models, the **Tuned Logistic Regression** model achieved the highest **Macro F1-score of 0.5526**, making it the recommended model for predicting IPL match winners.

---

# 📁 Project Structure

```
├── matches.csv
├── IPL_Predictive_Modeling_Updated.ipynb
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

---

# 📦 Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

1. Clone the repository.

```bash
git clone https://github.com/your-username/IPL-Match-Winner-Prediction.git
```

2. Navigate to the project directory.

```bash
cd IPL-Match-Winner-Prediction
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook or Google Colab.

5. Open:

```text
IPL_Predictive_Modeling_Updated.ipynb
```

6. Run all cells sequentially to reproduce the complete predictive modeling workflow.

---

# 🔁 Reproducibility

The project uses fixed random seeds wherever applicable to ensure reproducible results.

All preprocessing transformations—including encoding, scaling, oversampling, and model training—are performed within the appropriate training workflow to prevent data leakage and enable reproducibility.
