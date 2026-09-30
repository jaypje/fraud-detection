# Credit Card Fraud Detection

A binary classifier that flags fraudulent credit card transactions, built on the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, ~0.17% fraud).

The focus of this project is **honest evaluation on extremely imbalanced data**: avoiding leakage, choosing metrics that don't reward a useless model, tuning the decision threshold on validation only, and touching the test set once.

## Results

Random Forest, decision threshold 0.15 (chosen on validation), evaluated once on a held-out test set of 42,559 transactions (71 fraud cases):

| Metric | Value |
|---|---|
| Fraud precision | 0.773 |
| Fraud recall | 0.817 |
| Fraud F1 | 0.795 |
| Macro-F1 | 0.897 |
| AUC-PR | 0.803 |
| AUC-ROC | 0.961 |

58 frauds caught, 13 missed, 17 false alarms.

For comparison, an "always predict legit" baseline scores 99.8% accuracy but a macro-F1 of only 0.50, which is why accuracy is not used here.

**Model comparison (validation AUC-PR):** Random Forest 0.853, XGBoost 0.822, Logistic Regression 0.697. With only ~70 fraud cases per split, the Random Forest vs XGBoost gap is small and should be read as "comparable".

![Confusion matrix and precision-recall curve](figures/confusion_matrix_and_pr_curve_random_forest.png)

## Approach

1. **Cleaning:** removed 1,081 exact duplicate rows before splitting, so the model can't memorize rows that appear in both train and test.
2. **Split:** stratified 70/15/15 train/validation/test, so each split keeps the same fraud rate.
3. **Baseline:** majority-class predictor, to show why accuracy is misleading.
4. **Models:** Logistic Regression (scaled features), Random Forest and XGBoost, all using class weighting instead of SMOTE (the V1-V28 features are anonymized, so synthetic points aren't interpretable).
5. **Model selection:** by validation AUC-PR, since ROC-AUC looks deceptively good when negatives dominate.
6. **Threshold tuning:** minimized an assumed cost ($100 per missed fraud, $5 per false alarm) on validation, with a sensitivity check across four cost ratios. The chosen threshold (0.15) held for 3 of 4.
7. **Final evaluation:** test set used once.
8. **Analysis:** calibration curve, error analysis by transaction amount, SHAP explainability, and a confidence-based routing experiment.

## Key findings

- **Drivers (SHAP):** the model relies most on V12, V14, V4, V10 and V3. Low values of V12, V14, V10 and V3, and high values of V4, push predictions toward fraud.
- **Missed fraud:** missed cases have a higher mean amount than caught ones (~$248 vs ~$104), but a low median (~$30), so a few large misses drive the average.
- **Calibration:** reasonable at the extremes, too sparse to judge in the middle. Probabilities are fine for ranking and thresholding but not as exact risk estimates.

![SHAP summary](figures/shap_summary.png)

## Limitations

- Random split instead of time-based (the data covers only ~2 days), so concept drift is not tested.
- Cost assumptions are illustrative, not from real chargeback or customer-friction data.
- Only 71 fraud cases in the test set, so all test metrics carry wide uncertainty.
- No hyperparameter tuning beyond class weighting.
- Features are PCA-anonymized, so drivers like V14 can't be mapped to real-world transaction attributes.

## Next steps

1. Bootstrapped confidence intervals on test metrics.
2. RandomizedSearchCV on Random Forest and XGBoost.
3. Compare class weighting against SMOTE.
4. Amount-scaled false-negative cost, to reflect real financial loss.
5. Time-based validation on data spanning weeks or months.

## Run it yourself

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv` (the file is too large for GitHub, so it is not included).
2. Create an empty `figures/` folder if it doesn't exist.
3. Open `fraud_detection.ipynb` and run all cells.

## Repo structure

```
.
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
├── data/        # put creditcard.csv here (not committed)
└── figures/     # plots saved by the notebook
```

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib.
