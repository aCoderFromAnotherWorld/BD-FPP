# Temporal Model Evaluation & Comparison Strategy

## Executive Summary

This evaluation analyzes five machine learning models (**Gradient Boosting**, **Linear Regression**, **M5P**, **Random Forest**, and **XGBoost**) across three distinct temporal forecasting horizons: **`target_7`** (short-term), **`target_14`** (medium-term), and **`target_30`** (long-term). Models were evaluated across multiple commodities using five core metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Pearson Correlation Coefficient ($R$), Relative Absolute Error (RAE %), and Mean Absolute Percentage Error (MAPE %).

---

## 1. Overall Model Rankings & Performance Summary

The metrics below represent the macro-average performance across all commodities and temporal horizons ($7$, $14$, and $30$ days):

| Rank | Model | Pearson $R$ ↑ | RMSE ↓ | MSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | **0.4515** | **13.0515** | **510.32** | 99.83% | 7.74% |
| **2** | **XGBoost** | **0.4423** | 14.5950 | 672.08 | 95.64% | 7.96% |
| **3** | **Gradient Boosting** | **0.4250** | 13.1409 | 537.74 | **91.57%** | **6.95%** |
| **4** | **Linear Regression** | **0.4124** | 14.4094 | 512.98 | 148.51% | 9.52% |
| **5** | **M5P** | **0.3591** | 19.2541 | 1685.00 | 122.47% | 11.31% |

---

## 2. Performance Breakdown by Target Horizon

### **Short-Term Horizon: `target_7` (7-Day Forecast)**
* **Top Performer:** **Gradient Boosting** achieved the strongest linear correlation ($R = 0.5697$) and lowest percentage errors (RAE: $73.57\%$, MAPE: $5.51\%$).
* **Runner-Up:** Linear Regression ($R = 0.5204$, RMSE: $11.71$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.5697** | **10.4263** | **73.57%** | **5.51%** |
| **Linear Regression** | 0.5204 | 11.7067 | 133.25% | 8.33% |
| **XGBoost** | 0.5181 | 11.9779 | 81.96% | 6.39% |
| **Random Forest** | 0.5040 | 11.0515 | 103.41% | 7.26% |
| **M5P** | 0.4595 | 20.4367 | 116.76% | 11.47% |

---

### **Medium-Term Horizon: `target_14` (14-Day Forecast)**
* **Top Performer:** **Random Forest** took the lead in correlation ($R = 0.4573$), outperforming ensemble tree peers in maintaining directional predictive strength over 2 weeks.
* **Runner-Up:** Linear Regression ($R = 0.4238$) / Gradient Boosting (lowest RMSE: $12.43$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.4573** | 13.0623 | 97.74% | 8.62% |
| **Linear Regression** | 0.4238 | 13.5227 | 142.51% | 9.34% |
| **Gradient Boosting** | 0.4170 | **12.4272** | **93.77%** | **7.10%** |
| **XGBoost** | 0.4077 | 15.3655 | 107.21% | 9.48% |
| **M5P** | 0.3239 | 20.3865 | 127.81% | 13.57% |

---

### **Long-Term Horizon: `target_30` (30-Day Forecast)**
* **Top Performer:** **XGBoost** demonstrated superior resilience to variance decay over extended horizons, recording the highest correlation ($R = 0.4012$) and low relative errors (RAE: $97.75\%$).
* **Runner-Up:** Random Forest ($R = 0.3931$, lowest MAPE: $7.33\%$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.4012** | 16.4415 | **97.75%** | 8.03% |
| **Random Forest** | 0.3931 | **15.0408** | 98.34% | **7.33%** |
| **M5P** | 0.2938 | 16.9392 | 122.86% | 8.89% |
| **Linear Regression** | 0.2931 | 17.9989 | 169.75% | 10.88% |
| **Gradient Boosting** | 0.2883 | 16.5693 | 107.36% | 8.23% |

---

## 3. Best Model Declaration & Detailed Justification

### **Winner: Random Forest**

**Random Forest** is selected as the overall best model across all temporal horizons.

#### **Why Random Forest Won:**

1. **Highest Overall Correlation ($R = 0.4515$):**
   Random Forest achieved the highest average Pearson correlation across all target horizons and commodities, indicating that its predictions consistently track real-world price directionality better than competing models.

2. **Lowest Global Error Footprint ($\text{RMSE} = 13.0515$, $\text{MSE} = 510.32$):**
   Random Forest achieved the lowest overall RMSE among all models, demonstrating superior stability and minimizing large variance outliers.

3. **Balanced Temporal Multi-Step Stability:**
   Unlike Gradient Boosting (which degraded sharply from $R = 0.5697$ at 7 days down to $R = 0.2883$ at 30 days) or Linear Regression (which suffered high RAE above $140\%$), Random Forest retained consistent strength across $7$, $14$, and $30$ days ($R = 0.5040 \rightarrow 0.4573 \rightarrow 0.3931$).

4. **Robustness to Non-Linear Price Volatility:**
   By averaging multiple decision trees trained on bootstrapped subsets, Random Forest effectively prevented overfitting—a key advantage over M5P (which suffered high MSEs) and standalone regression models.