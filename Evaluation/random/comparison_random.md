# Random Split Model Evaluation & Comparison Strategy

## Executive Summary

This evaluation analyzes five machine learning models (**Gradient Boosting**, **Linear Regression**, **M5P**, **Random Forest**, and **XGBoost**) trained and tested under a **random split strategy** (80% train / 20% test) across three distinct target horizons: **`target_7`** (short-term), **`target_14`** (medium-term), and **`target_30`** (long-term). Models were evaluated across multiple commodities using five core metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Pearson Correlation Coefficient ($R$), Relative Absolute Error (RAE %), and Mean Absolute Percentage Error (MAPE %).

---

## 1. Overall Model Rankings & Performance Summary

The metrics below represent the macro-average performance across all commodities and target horizons ($7$, $14$, and $30$ days) using the random split dataset:

| Rank | Model | Pearson $R$ ↑ | RMSE ↓ | MSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **XGBoost (Random)** | **0.7887** | 9.7512 | 383.69 | **42.96%** | **3.97%** |
| **2** | **Random Forest (Random)** | 0.7693 | **9.2606** | **355.23** | 53.16% | 4.42% |
| **3** | **Gradient Boosting (Random)** | 0.6572 | 12.0758 | 477.86 | 66.61% | 6.54% |
| **4** | **M5P (Random)** | 0.6061 | 12.3232 | 532.60 | 66.73% | 6.31% |
| **5** | **Linear Regression (Random)** | 0.5506 | 14.0134 | 614.86 | 75.45% | 8.12% |

---

## 2. Performance Breakdown by Target Horizon

### **Short-Term Horizon: `target_7` (7-Day Forecast)**

* **Top Performer:** **XGBoost (Random)** achieved the highest linear correlation ($R = 0.8387$), lowest relative error (RAE: $38.51\%$), and lowest percentage error (MAPE: $3.65\%$).
* **Runner-Up:** **Random Forest (Random)** ($R = 0.8147$, lowest RMSE: $7.8050$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| --- | --- | --- | --- | --- |
| **XGBoost (Random)** | **0.8387** | 7.8759 | **38.51%** | **3.65%** |
| **Random Forest (Random)** | 0.8147 | **7.8050** | 51.85% | 4.34% |
| **Gradient Boosting (Random)** | 0.7233 | 10.8160 | 53.70% | 5.35% |
| **M5P (Random)** | 0.7099 | 11.6913 | 58.34% | 5.84% |
| **Linear Regression (Random)** | 0.6511 | 12.8200 | 64.58% | 6.91% |

---

### **Medium-Term Horizon: `target_14` (14-Day Forecast)**

* **Top Performer:** **Random Forest (Random)** took the lead in Pearson correlation ($R = 0.7812$) and overall absolute variance error (lowest RMSE: $10.3595$).
* **Runner-Up:** **XGBoost (Random)** ($R = 0.7642$, lowest RAE: $44.47\%$, lowest MAPE: $4.29\%$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| --- | --- | --- | --- | --- |
| **Random Forest (Random)** | **0.7812** | **10.3595** | 53.33% | 4.55% |
| **XGBoost (Random)** | 0.7642 | 11.5186 | **44.47%** | **4.29%** |
| **Gradient Boosting (Random)** | 0.6710 | 12.5523 | 63.26% | 6.67% |
| **Linear Regression (Random)** | 0.5792 | 14.3972 | 73.34% | 8.18% |
| **M5P (Random)** | 0.5589 | 13.6175 | 70.85% | 6.96% |

---

### **Long-Term Horizon: `target_30` (30-Day Forecast)**

* **Top Performer:** **XGBoost (Random)** dominated the 30-day forecast with the highest correlation ($R = 0.7632$), lowest RAE ($45.91\%$), and lowest MAPE ($3.97\%$).
* **Runner-Up:** **Random Forest (Random)** ($R = 0.7120$, lowest RMSE: $9.6172$).

| Model | Pearson $R$ ↑ | RMSE ↓ | RAE (%) ↓ | MAPE (%) ↓ |
| --- | --- | --- | --- | --- |
| **XGBoost (Random)** | **0.7632** | 9.8592 | **45.91%** | **3.97%** |
| **Random Forest (Random)** | 0.7120 | **9.6172** | 54.29% | 4.35% |
| **Gradient Boosting (Random)** | 0.5774 | 12.8591 | 82.86% | 7.59% |
| **M5P (Random)** | 0.5495 | 11.6607 | 70.99% | 6.14% |
| **Linear Regression (Random)** | 0.4216 | 14.8230 | 88.42% | 9.26% |

---

## 3. Best Model Declaration & Detailed Justification

### **Winner: XGBoost (Random)**

**XGBoost (Random)** is selected as the overall best model for the randomly split dataset.

#### **Why XGBoost Won:**

1. **Highest Overall Correlation ($R = 0.7887$):**
XGBoost achieved the highest average Pearson correlation coefficient across all commodities and target horizons, outperforming Random Forest ($0.7693$) and significantly outpacing Gradient Boosting ($0.6572$), M5P ($0.6061$), and Linear Regression ($0.5506$).
2. **Superior Relative Accuracy ($\text{RAE} = 42.96\%$, $\text{MAPE} = 3.97\%$):**
XGBoost achieved the lowest relative error footprint across the board, reducing Mean Absolute Percentage Error below $4.0\%$ and Relative Absolute Error down to $42.96\%$ (compared to Random Forest's $53.16\%$ RAE and $4.42\%$ MAPE).
3. **Multi-Horizon Dominance (`target_7` & `target_30`):**
XGBoost ranked #1 in both the short-term 7-day horizon ($R = 0.8387$, MAPE $= 3.65\%$) and the long-term 30-day horizon ($R = 0.7632$, MAPE $= 3.97\%$). It maintained exceptional performance even as the prediction window expanded.
4. **Advanced Gradient Boosting Dynamics on Randomly Sampled Data:**
By leveraging second-order Taylor expansion for gradient optimization, explicit $\text{L1}$ and $\text{L2}$ regularization penalties, and efficient tree pruning, XGBoost effectively fits complex multi-feature relationships without overfitting the random split training set.