# Comment Category Prediction Challenge

Project code for the Kaggle competition [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge) for the Term 1 2026 Machine Learning Practice Project.
`
## Project Overview

Online platforms struggle to moderate harmful content at scale. This project tackles that problem by training a pipeline that classifies comments into four labels (Labels 0 through 3), where Label 0 represents the most neutral content and Label 3 represents the most explicitly violent or harmful language.

The dataset includes both text features (comment body) and structured features (upvotes, downvotes, interaction flags, emoticon signals, and demographic metadata such as race, gender, religion, and disability mentions). The final model is a stacking ensemble that combines the predictions of six base classifiers.

## Problem Statement

Given a comment and its associated metadata, predict one of four target classes:

- **Label 0** -- Generally neutral or political content
- **Label 1** -- Content related to race, identity, or group-based topics
- **Label 2** -- More polarizing political or social commentary
- **Label 3** -- Explicitly violent or harmful language

This is a 4-class imbalanced classification problem evaluated using **macro F1 score**.

## Dataset

The training data contains comments from online discussions with the following feature groups:

| Feature Group | Features |
|---|---|
| Temporal Data | `created_date` and derived features |
| Vote features | `upvote`, `downvote` |
| Numeric/interaction | `if_1`, `if_2`, `emoticon_1`, `emoticon_2`, `emoticon_3` |
| Categorical (demographics) | `race`, `gender`, `religion`, `disability` |
| Text | `comment` |

Groups were used as a splitting criterion during cross-validation to prevent data leakage across folds. A `StratifiedGroupKFold` splitter was used for initial model benchmarking and hyperparameter tuning. `StratifiedKFold` was used for feature selection, and final model training and evaluation.

## Exploratory Data Analysis

Several label biases were identified across demographic feature groups:

- **Gender**: Transgender-related comments showed the highest concentration of Label 1 (52.73%) and lowest Label 0 (27.97%), a distinct polarization compared to male and female groups.
- **Religion**: Comments mentioning Islam (63.57% Label 1) and Judaism (54.10% Label 1) showed the strongest Label 1 dominance. Christianity and Hinduism aligned more closely with non-religious comment distributions.
- **Race**: Clear label shifts were observed depending on the racial group mentioned in the comment.

N-gram frequency analysis revealed that Label 3 comments contain explicitly violent phrases (e.g., "kill people", "shoot kill", "death penalty"), while Label 0 and Label 2 comments share more political vocabulary (e.g., "fake news", "climate change", "donald trump").

UMAP and t-SNE projections of comment text embeddings were also computed to visualize class separability in reduced dimensions.

The generated plots can be found in the plots/ directory, organized by the comment IDs used in Section 1.

## Preprocessing Pipeline

A `ColumnTransformer` was used to apply different preprocessing strategies to each feature group:

| Transformer Name | Features | Steps |
|---|---|---|
| `log` | `upvote`, `downvote` | `log1p` transform + `RobustScaler` |
| `num` | `if_1`, `if_2`, `emoticon_1/2/3` | `SimpleImputer(constant, 0)` + `PowerTransformer(yeo-johnson)` |
| `cat` | `race`, `gender`, `religion`, `disability` | `SimpleImputer(constant)` + `OrdinalEncoder` |
| `text` | `comment` | Custom text cleaning + `TfidfVectorizer` |
| `date` | Date-derived columns | `RobustScaler` |

A `SelectKBest` feature selection step with `chi2` scoring was applied after vectorization for tree-based models.

## Models

Six classifiers were trained and tuned using `GridSearchCV` with macro F1 scoring:

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `C=5.5`, `solver=saga`, `class_weight=balanced`, `max_iter=1500` |
| SGD Classifier | Tuned regularization and learning rate |
| XGBoost | `objective=multi:softprob`, `eval_metric=mlogloss` |
| LightGBM | `num_leaves=120`, `lambda_l1=0.05`, `lambda_l2=0.15`, `min_child_samples=10` |
| CatBoost | `loss_function=MultiClassOneVsAll`, `auto_class_weights=SqrtBalanced` |
| Random Forest | `class_weight=balanced_subsample`, `oob_score=macro_f1` |

All models were wrapped in `sklearn.pipeline.Pipeline` with the appropriate `ColumnTransformer` preprocessor.

## Final Model

A `StackingClassifier` was trained on top of the six base models using a `LogisticRegression` meta-learner with `predict_proba` as the stacking method. Cross-validation was performed using the same `StratifiedKFold` splitter to maintain class distribution across folds.

Training the full stacking classifier took approximately 10 hours on Kaggle.

## Feature Importance

Feature importance was explored during local development using LOFO and permutation importance. The results were not reproducible within Kaggle's 12-hour runtime limit, so this step was excluded from the final submission notebook.

## Evaluation Metric

All models were evaluated using **macro F1 score** to account for class imbalance across the four target labels. The strongest individual model achieved roughly 0.82 macro F1, and the stacking classifier improved this to about 0.84 on the test set.

## Notes on improving model performance

- Improve text cleaning with spaCy and NLTK: This project was limited in the scope of external libraries it could use, but lemmatization and a stronger stopwords list to filter vocabulary could improve both score and interpretability.
- Use high regularization for text features: Lower max_df, increase min_df, and reduce vocabulary size for tree-based models to cut noise and overfitting.
- Reduce the number of models: The final ensemble includes three boosting models with similar error patterns. Depending on resource and time limits, reduce the number of boosting models to one. If training is time-limited but RAM/VRAM are not a concern, prefer XGBoost. If resources are constrained but time is not, prefer LightGBM or CatBoost.
- Train using One-vs-One Classifiers: This allows each sub-classifier to focus on a single pairwise boundary, which can improve separation for overlapping or imbalanced classes at the cost of longer training time.
- Try alternative meta-learners: Using an alternative meta-classifier like a highly regularized LightGBM or Ridge Classifier can help map non-linear boundaries present in the probabilities produced by the sub-classifiers.
- Threshold Tuning: Calibrating the meta-classifier and tuning per-class thresholds on validation data may improve minority class performance.

## Dependencies

- Python 3.14
- scikit-learn 1.8
- XGBoost
- LightGBM
- CatBoost
- pandas, numpy
- matplotlib, seaborn, plotly
- UMAP, joblib