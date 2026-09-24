# BD-FPP: A Multi-Region Bangladeshi Food-Price Dataset with Engineered Lag Features for Time-Series Commodity Price Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Weka 3.8](https://img.shields.io/badge/Weka-3.8-orange.svg)](https://www.cs.waikato.ac.nz/ml/weka/)
[![Dataset Version](https://img.shields.io/badge/Dataset%20Version-v2.2-green.svg)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Institution](https://img.shields.io/badge/KUET-CSE%204112-red.svg)](https://www.kuet.ac.bd/)

**Course:** CSE 4112: Machine Learning Laboratory  
**Department:** Department of Computer Science and Engineering  
**Institution:** Khulna University of Engineering & Technology (KUET), Khulna-9203, Bangladesh  
**Supervisors:** Dr. Muhammad Aminul Haque, Nabil Faiyaz Sadi  
**Group:** B2 (Batch 2K21)  

---

## 👥 Authors & Team Contributions (CRediT Taxonomy)

| Roll | Student Name | Overall Contribution | Primary Roles (CRediT) | Artifacts / Workflows Owned |
| :---: | :--- | :---: | :--- | :--- |
| **2107091** | **Md. Sabith** | 20% | Conceptualization, Methodology, Software (Linear Regression), Writing | `Evaluation/temporal/moa_lag_evaluation.ipynb`, Linear Regression Weka logs, LaTeX report drafting |
| **2107100** | **Abu Hasanat Soykot** | 20% | Conceptualization, Data curation, Software (M5P), Validation | `Preprocessing/moa_again_process.ipynb`, M5P evaluation buffers, dataset curation |
| **2107106** | **Newaz Mohammad Hamim** | 20% | Methodology, Software (Random Forest), Visualization | `time_series_figure.ipynb`, macro & overlay trajectory visualizations, `Report/figures/` |
| **2107108** | **Tajnoor Sultana** | 20% | Investigation, Software (GBR), Formal analysis, App Development | GBR evaluation logs, multi-horizon error ablation, `gui.py` desktop Tkinter application |
| **2107115** | **Ahsanul Islam Emon** | 20% | Data curation, Software (XGBoost), Visualization, Project admin | XGBoost Weka scripts, cross-model comparison tables, `requirements.txt` |

---

## 📌 Executive Summary

**BD-FPP (Bangladeshi Food-Price Panel)** is a spatially disaggregated panel of daily retail commodity prices across all eight administrative divisions of Bangladesh. The panel is enriched with:
1. **Synchronized Macroeconomic Coupling:** Daily central bank USD/BDT reference exchange rates from Bangladesh Bank to support currency pass-through modeling on domestic staple foods.
2. **Within-Series Feature Engineering:** Four ready-made historical price features (`Price_T-1`, `Price_T-7`, `Price_T-30`, `Price_30d_MA`) computed strictly forward in time within each independent series to eliminate cross-series leakage.
3. **Direct Multi-Horizon Targets:** Three direct forward price forecasting targets (`target_7d`, `target_14d`, `target_30d`), enabling systematic empirical analysis of predictive skill decay over 7-, 14-, and 30-day forecast horizons.
4. **Leakage-Safe Partitioning Protocols:** Dual 80/20 train/test evaluation splits (strictly chronological holdout vs. interleaved random shuffle) that allow researchers to empirically quantify temporal data leakage in retail price time series.

The model-ready dataset comprises **25,207 verified records** spanning **18 commodities**, **8 divisions**, and **295 distinct calendar dates** between 16 June 2025 and 20 August 2026.

---

## 📊 Dataset Scope & Specifications

| Dimension | Specification |
| :--- | :--- |
| **Geographic Coverage** | 8 Administrative Divisions of Bangladesh: *Barisal, Chattagram, Dhaka, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet* |
| **Commodity Coverage** | 18 Essential Retail Commodities: *Ata (Packet), Ata (loose)-White, Beef, Broiler chicken, Egg Farm-Red, Garlic (Imported), Garlic (local)-Big Size, Ginger (Imported), Green Chili (Local), Lentils-Desi-Whole, Milk, Onion (local), Palm Oil, Pangash (big), Potato (Holland)-Red, Rice-Fine, Rice-Medium, Soybean Oil (loose)* |
| **Temporal Window** | 16 June 2025 – 20 August 2026 (295 unique dates) |
| **Total Observations** | 25,207 complete records with full 30-day lag history and 30-day forward targets |
| **Primary Data Source** | Department of Agricultural Marketing (DAM), Ministry of Agriculture ([moa-services.com](https://moa-services.com/market-directory/product-wise-market-price-report)) |
| **Macroeconomic Feed** | Bangladesh Bank Daily Reference Exchange Rate Bulletin (USD/BDT: 119.50 – 122.80) |
| **Release Formats** | CSV (`Preprocessing/`), Weka ARFF (`ARFF Files/`), serialized models (`Model/`), prediction buffers (`Buffer/`, `Evaluation/`) |

---

## 📐 Data Dictionary (15 Attributes)

| Attribute / Field | Type | Range / Format | Description | Missing Rule |
| :--- | :---: | :--- | :--- | :---: |
| `division` | Nominal | 8 administrative divisions | Division-level geographical jurisdiction | Drop / Not permitted |
| `commodity_name` | Nominal | 18 canonical items | Standardized retail commodity label | Drop / Not permitted |
| `retail_unit` | Nominal | `Kg`, `Liter`, `4 Pcs` | Standard transaction unit of observation | Drop / Not permitted |
| `year` | Numeric | 2025 – 2026 | Calendar observation year | None permitted |
| `month` | Numeric | 1 – 12 | Calendar observation month | None permitted |
| `day` | Numeric | 1 – 31 | Calendar observation day | None permitted |
| `rate` | Numeric | 119.50 – 122.80 | Bangladesh Bank daily USD/BDT reference exchange rate | None permitted |
| `average_price` | Numeric | 14.50 – 1,440.00 BDT | Current average retail market price per unit | None permitted |
| `Price_T-1` | Numeric | BDT / retail unit | 1-day lagged price ($p_{s, t-1}$) within series $s$ | Boundary dropped |
| `Price_T-7` | Numeric | BDT / retail unit | 7-day lagged price ($p_{s, t-7}$) within series $s$ | Boundary dropped |
| `Price_T-30` | Numeric | BDT / retail unit | 30-day lagged price ($p_{s, t-30}$) within series $s$ | Boundary dropped |
| `Price_30d_MA` | Numeric | BDT / retail unit | 30-day moving average: $\frac{1}{30}\sum_{k=1}^{30} p_{s, t-k}$ | Boundary dropped |
| `target_7d` | Numeric | BDT / retail unit | 7-day ahead forward price level ($p_{s, t+7}$) | Supervised target |
| `target_14d` | Numeric | BDT / retail unit | 14-day ahead forward price level ($p_{s, t+14}$) | Supervised target |
| `target_30d` | Numeric | BDT / retail unit | 30-day ahead forward price level ($p_{s, t+30}$) | Supervised target |

---

## 🧪 Benchmark Machine Learning Models & Results

Five baseline regressors were evaluated under both protocols across all 18 commodities and all 3 forecasting horizons:
1. **Linear Regression (LR)** — Standard Ordinary Least Squares baseline
2. **M5P Model Tree** — Piecewise linear decision tree with regression equations at terminal leaves
3. **Random Forest (RF)** — Ensemble bagging of 100 random decision trees
4. **Gradient Boosting Regressor (GBR)** — Sequentially boosted residual trees
5. **XGBoost Regressor (XGBR)** — Scaled extreme gradient boosting

### Macro-Averaged Benchmark Comparison

*(Metrics macro-averaged with equal weight across all 18 commodities to prevent high-priced goods like Beef from dominating)*

| Model | Chronological $R$ | Chronological RAE (%) | Chronological RMSE | Random $R$ | Random RAE (%) | Random RMSE | Key Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.4515** | 99.83% | **13.05** | 0.7712 | 45.43% | 14.65 | Best overall chronological correlation and lowest RMSE; bounded predictions prevent extrapolation drift. |
| **XGBoost** | 0.4423 | 95.64% | 14.60 | **0.7887** | **42.96%** | **13.56** | Top performer under random split; shows largest generalization gap ($2.23\times$ RAE increase under chronological split). |
| **Gradient Boosting** | 0.4250 | **91.57%** | 13.14 | 0.7672 | 46.40% | 14.77 | Lowest relative absolute error and lowest MAPE (6.95%) under true forward holdout. |
| **Linear Regression** | 0.4124 | 148.51% | 14.41 | 0.5898 | 80.98% | 19.34 | Global hyperplane baseline; vulnerable to macroeconomic regime drift in out-of-sample periods. |
| **M5P Model Tree** | 0.3591 | 122.47% | 19.25 | 0.5701 | 81.39% | 20.25 | Piecewise linear rules; sensitive to isolated regional boundary shifts at short forecast horizons. |

> **⚠️ The Temporal Data Leakage Finding:** Randomly shuffling time series data artificially inflates Pearson correlation $R$ by +0.14 to +0.35 and cuts relative absolute error (RAE) by more than half ($42.96\%$ vs. $95.64\%$). This confirms that random cross-validation leaks neighboring future observations into training folds. Chronological partitioning is essential for realistic evaluation.

---

## 🗂 Repository Structure

```
BD-FPP/
├── ARFF Files/                          # Weka-ready ARFF format datasets
│   ├── moa_train_80_lag.arff            # Chronological 80% train split
│   ├── moa_test_20_lag.arff             # Chronological 20% test split
│   ├── moa_train_random_80_lag.arff     # Randomly shuffled 80% train split
│   └── moa_test_random_20_lag.arff      # Randomly shuffled 20% test split
├── Buffer/                              # Raw Weka run logs and execution traces
│   ├── temporal/                        # Logs for chronological protocol (target_7, 14, 30)
│   └── random/                          # Logs for random shuffle protocol (target_7, 14, 30)
├── Data Scrapper/                       # Web harvesting scripts
│   ├── scrape_moa.py                    # Daily scraper for DAM MOA market price bulletin
│   └── requirements.txt                 # Scraper dependencies
├── Evaluation/                          # Metric extraction and per-commodity analysis
│   ├── temporal/                        # Chronological holdout evaluation notebooks & metrics CSV
│   └── random/                          # Random split evaluation notebooks & metrics CSV
├── Model/                               # Serialized Weka & Scikit-learn trained models (30 models)
│   ├── temporal/                        # Serialized .model binaries for 5 models × 3 horizons
│   └── random/                          # Serialized .model binaries for 5 models × 3 horizons
├── Preprocessing/                       # Primary dataset cleaning and lag engineering
│   ├── moa_again_process.ipynb          # Master processing pipeline notebook
│   ├── moa_final_dataset_with_lag.csv   # Released 25,207-row multi-horizon master dataset
│   ├── moa_train_80_lag.csv             # Chronological training split (20,165 rows)
│   ├── moa_test_20_lag.csv              # Chronological test split (5,042 rows)
│   ├── moa_train_random_80_lag.csv      # Random training split (20,165 rows)
│   └── moa_test_random_20_lag.csv       # Random test split (5,042 rows)
├── Report/                              # Official Course Dataset Report
│   ├── 91_100_106_108_115.pdf           # Full 53-page compiled submission report
│   └── dump.rar                         # LaTeX sources, figures, and build archives
├── Slide/                               # Presentation materials
│   └── 91_100_106_108_115.pptx          # Final project defense presentation slide deck
├── timeseries-figures/                  # Visual trajectories & diagnostic plots
│   ├── Cross-Model/                     # Single-axis multi-model overlays
│   ├── Multi Horizon/                   # Horizon ablation trajectory curves
│   ├── Per Commodity Plot/              # Per-commodity individual time-series charts
│   └── Report_Ready_Figures/            # 15 macro grids (18 commodities) + 9 cross-model overlays
├── gui.py                               # Interactive Tkinter desktop forecasting application
├── requirements.txt                     # Python dependencies
├── time_series_figure.ipynb             # Script to generate all time-series figures
├── links.txt                            # External resource and data source links
└── README.md                            # Repository documentation (this file)
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aCoderFromAnotherWorld/BD-FPP.git
cd BD-FPP
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Interactive Desktop Forecasting Application (`gui.py`)
```bash
python gui.py
```
**Features of the GUI application:**
- **Split Protocol Selection:** Choose between `Temporal` (real forward forecasting) and `Random` models.
- **Model Selector:** Choose among Linear Regression, M5P Model Tree, Random Forest, GBR, and XGBoost.
- **Interactive Query:** Pick any division, commodity, and future target date.
- **Greedy Multi-Hop Horizon Planning:** Automatically decomposes forecasting intervals beyond 7 days into greedy 30-day, 14-day, and 7-day jumps, appending synthetic intermediate predictions and updating lag features sequentially.
- **Visual Trajectory:** Renders historical prices alongside forward projections and uncertainty intervals in an embedded Matplotlib window.

---

## 📓 Running the Notebooks

1. **Feature Engineering & Preprocessing:**
   Open and execute `Preprocessing/moa_again_process.ipynb` to inspect data cleaning, date sorting, within-series lag calculation, and ARFF generation.
2. **Evaluation & Performance Diagnostics:**
   Open `Evaluation/temporal/moa_lag_evaluation.ipynb` and `Evaluation/random/moa_lag_evaluation.ipynb` to reproduce the per-commodity metric aggregations and macro tables.
3. **Time-Series Trajectory Figures:**
   Open `time_series_figure.ipynb` to regenerate the full suite of macro overviews and cross-model comparison plots.

---

## 📜 Citation & License

This dataset and code are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

If using BD-FPP in academic work or reports, please cite:
```bibtex
@misc{bdfpp2026,
  title={BD-FPP: A Multi-Region Bangladeshi Food-Price Dataset with Engineered Lag Features for Time-Series Commodity Price Forecasting},
  author={Sabith, Md. and Soykot, Abu Hasanat and Hamim, Newaz Mohammad and Sultana, Tajnoor and Emon, Ahsanul Islam},
  year={2026},
  howpublished={CSE 4112 Machine Learning Laboratory, Department of Computer Science and Engineering, Khulna University of Engineering \& Technology (KUET)},
  url={https://github.com/aCoderFromAnotherWorld/BD-FPP}
}
```
