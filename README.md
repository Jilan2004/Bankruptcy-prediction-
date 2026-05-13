# Corporate Bankruptcy Prediction — CatBoost Model & Web Application

A machine learning project for predicting corporate bankruptcy among publicly listed American companies (NYSE/NASDAQ, 1999–2018). Built as part of a bachelor's thesis at Romanian-American University, Bucharest.

The project trains a CatBoost gradient boosting classifier, applies probability calibration and threshold tuning, and deploys the model through a web application where users can input a company's financial data and receive a bankruptcy risk assessment.

---

## Results Summary

| Model | AUC | Brier Score | Failed F1 |
|---|---|---|---|
| Logistic Regression | 0.752 | 0.237 | 0.058 |
| Decision Tree | 0.735 | 0.222 | 0.080 |
| Random Forest | 0.804 | 0.186 | 0.079 |
| CatBoost (uncalibrated) | 0.824 | 0.111 | 0.125 |
| **CatBoost + Isotonic Calibration** | **0.825** | **0.025** | **0.188** |

Classification threshold set to **0.4** (raises recall on bankrupt companies from 51% to 70%).

---

## Project Structure

```
Bankruptcy_thesis/
│
├── app.py                        # Web application (Streamlit)
│
├── data/
│   └── american_bankruptcy_dataset.csv   # ! Not included — see below
│
├── model/
│   ├── catboost_bankruptcy.cbm           # ! Not included — see below
│   ├── app_metadata.joblib               # ! Not included — see below
│   ├── isotonic_calibrator.joblib        # ! Not included — see below
│   └── platt_calibrator.joblib           # ! Not included — see below
│
├── notebooks/
│   ├── step1_explore_data.ipynb          # EDA
│   ├── step2_train_model.ipynb           # Model training
│   ├── step3_imbalance.ipynb             # Class imbalance handling
│   ├── step4_calibration.ipynb           # Probability calibration
│   ├── step5_baselines.ipynb             # Baseline comparisons
│   ├── step6_hyperparameter_tuning.ipynb # Grid search tuning
│   ├── step7_walkforward.ipynb           # Walk-forward validation
│   └── step8_altman_zscore.ipynb         # Altman Z-Score comparison
│
└── test_real_post2018.csv        # Real company test data (Apple, Macy's, JC Penney)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jilan2004/Bankruptcy-prediction-.git
cd Bankruptcy-prediction-
```

### 2. Install dependencies

```bash
pip install catboost scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```

### 3. Download the dataset

The dataset is published by Lombardo et al. (2022) and is available at:

 https://github.com/sowide/bankruptcy_dataset

Download `american_bankruptcy_dataset.csv` and place it inside the `data/` folder.

### 4. Train the model

Run the notebooks in order:

```bash
jupyter notebook
```

Open and run `step1_explore_data.ipynb` through `step8_altman_zscore.ipynb` in sequence. After running `step2_train_model.ipynb`, the trained model files will be saved to the `model/` folder automatically.

---

## Running the App

Once the model files are in the `model/` folder, launch the web application:

```bash
streamlit run app.py
```

The app will open in your browser. You can either:
- Enter a company's financial data manually
- Upload a CSV file with multiple companies

The app outputs a bankruptcy probability and risk classification (Low / Medium / High) for each company.

---

## Dataset

- **Source**: Lombardo et al. (2022) — *Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks*. Future Internet, 14(8), 244. MDPI.
- **Companies**: 8,262 publicly listed US companies (NYSE/NASDAQ)
- **Observations**: 78,682 firm-year records
- **Period**: 1999–2018
- **Class balance**: 93.4% alive, 6.6% bankrupt
- **Features**: 18 financial variables (raw values from annual reports) + industry division code

---

## Real Company Test Data (test_real_post2018.csv)

This file contains financial data for three real companies from after the training period (post-2018), used to evaluate how the model performs on unseen real-world data. All figures are sourced from official annual reports and SEC 10-K filings.

| Company | Fiscal Year | Source | Expected Risk | Model Prediction |
|---|---|---|---|---|
| Apple Inc. | FY2020 (ended Sept 2020) | Apple 10-K, SEC EDGAR | Low | Low ✅ |
| Macy's Inc. | FY2019 (ended Feb 2020) | Macy's 10-K, SEC EDGAR | Medium | Medium ✅ |
| J.C. Penney | FY2019 (ended Feb 2020) | JC Penney earnings press release (GlobeNewswire, Feb 2020) | High | Medium ⚠️ |

JC Penney filed for Chapter 11 bankruptcy in May 2020, four months after this fiscal year ended. The model correctly identified financial distress (Medium risk) but did not predict imminent bankruptcy — a known limitation of models that rely solely on annual financial statements and cannot detect short-term liquidity crises or debt maturity concentrations.

---

## Key Design Decisions

- **Temporal split** — train ≤ 2011, validation 2012–2014, test ≥ 2015. Prevents data leakage from future years into training.
- **scale_pos_weight = 6** — handles class imbalance without discarding training data (unlike random undersampling).
- **Threshold = 0.4** — lowers the decision boundary to improve recall on bankrupt companies at the cost of slightly lower precision.
- **Isotonic calibration** — maps raw CatBoost scores to calibrated probabilities (Brier score reduced from 0.111 to 0.025).
- **X14 removed** — Total Current Liabilities (X14) was found to be identical to Total Liabilities (X17) across all 78,682 rows, indicating a data extraction error. X14 is excluded from user input and automatically derived from X17 at inference time.

---

## Citation

If you use this project or the dataset, please cite the original dataset paper:

```
Lombardo, G.; Pellegrino, M.; Adosoglou, G.; Cagnoni, S.; Pardalos, P.M.; Poggi, A. (2022).
Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.
Future Internet, 14(8), 244. MDPI.
https://www.mdpi.com/1999-5903/14/8/244
```

---

## Author

**Jilan Kourdi**  
Bachelor's Thesis — Romanian-American University, Bucharest  
Faculty of Computer Science for Business Management  
GitHub: [Jilan2004](https://github.com/Jilan2004)
