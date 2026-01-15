# ⚡ URJA AI - Nepal Electricity Load Forecasting

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A machine learning web application that predicts monthly electricity demand (GWh) for Nepal using historical data from the **Nepal Electricity Authority (2012-2025)**.

**Author:** Subodh  
**Last Updated:** January 2026

![Demand Analysis](models/nepal_2024/demand_analysis.png)

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Data Source** | Nepal Electricity Authority (NEA) |
| **Time Period** | August 2012 – December 2025 |
| **Total Records** | 161 monthly observations |
| **Min Demand** | 323.54 GWh (May 2013) |
| **Max Demand** | 1,224.56 GWh (September 2025) |
| **Total Growth** | 278% increase over 13+ years |
| **CAGR** | ~8.5% compound annual growth rate |

### Annual Demand Growth

| Fiscal Year | Total (TWh) | YoY Growth |
|-------------|-------------|------------|
| 2012/13 | 4.74 | - |
| 2013/14 | 5.02 | +5.93% |
| 2014/15 | 5.38 | +7.17% |
| 2015/16 | 5.79 | +7.62% |
| 2016/17 | 7.06 | +21.93% |
| 2017/18 | 7.41 | +4.87% |
| 2018/19 | 7.87 | +6.18% |
| 2019/20 | 7.89 | +0.36% ⚠️ COVID |
| 2020/21 | 8.97 | +13.60% |
| 2021/22 | 10.69 | +19.10% 🚀 |
| 2022/23 | 11.84 | +10.80% |
| 2023/24 | 12.70 | +7.29% |
| 2024/25 | 12.38 | -2.52% |
| 2025/26* | 5.58* | (partial) |

*Partial year data

### 📈 Historical Monthly Demand Trends

| Period | Avg Monthly (GWh) | Peak Month | Key Events |
|--------|-------------------|------------|------------|
| 2012-2013 | 324-395 | Sep 2013 (395 GWh) | Baseline period, load shedding era |
| 2014-2015 | 420-480 | Aug 2015 (480 GWh) | Nepal earthquake (Apr 2015) |
| 2016-2017 | 520-650 | Jul 2017 (650 GWh) | Load shedding ends, demand surge |
| 2018-2019 | 610-720 | Jul 2019 (720 GWh) | Industrial growth period |
| 2020-2021 | 650-850 | Jul 2021 (850 GWh) | COVID recovery, rapid growth |
| 2022-2023 | 900-1,100 | Sep 2023 (1,100 GWh) | Peak growth phase |
| 2024-2025 | 1,050-1,225 | Sep 2025 (1,225 GWh) | Current highest demand |

**Key Observations:**
- 🔋 Demand nearly **quadrupled** from 324 GWh (2012) to 1,225 GWh (2025)
- 📊 **Monsoon months** (Jul-Sep) consistently show peak demand due to irrigation
- 🏭 **Industrial expansion** post-2016 drove exponential growth
- ⚡ **Load shedding elimination** (2017) unlocked suppressed demand

---

## 🤖 Model Performance & Accuracy

### Current Metrics

| Metric | Value | What It Means |
|--------|-------|---------------|
| **R² Score** | 95.47% | Model explains 95%+ of variance in data |
| **MAPE** | 2.85% | Predictions are within ±3% of actual values |
| **MAE** | 12.85 GWh | Average prediction error |
| **RMSE** | 18.42 GWh | Root mean squared error |
| **CV RMSE** | 0.34 | Cross-validation consistency |

### Understanding the Metrics

#### 📈 R² Score (Coefficient of Determination)
- **What:** Measures how well predictions match actual values (0-100%)
- **Our Score:** 95.47% - Very good fit due to:
  - Extended dataset with 161 data points (50% more than before)
  - Comprehensive feature engineering
  - Time series cross-validation
  - Gradient Boosting with optimized hyperparameters
- **Context:** 95%+ R² indicates strong pattern recognition

#### 📊 MAPE (Mean Absolute Percentage Error)
- **What:** Average percentage difference between predicted and actual
- **Our Score:** 2.85% - **Very Good!**
- **Interpretation:** 
  - < 1% = Excellent forecasting
  - 1-5% = Very good forecasting ✅ (We're here!)
  - 5-10% = Good forecasting
  - > 10% = Needs improvement
- **Real-world:** If actual demand is 1,000 GWh, our prediction will be ~971-1,029 GWh

#### 📉 MAE (Mean Absolute Error)
- **What:** Average absolute difference in GWh
- **Our Score:** 12.85 GWh
- **Context:** Out of average monthly demand of ~800 GWh, error is ~1.6% - very acceptable

#### 🎯 RMSE (Root Mean Squared Error)
- **What:** Like MAE but penalizes larger errors more heavily
- **Our Score:** 18.42 GWh
- **Use:** Slightly higher than MAE indicates some variation, but still within acceptable range

### Model Strengths

```
✅ KEY FEATURES:
• Extended dataset with 161 samples (2012-2025)
• Historical data from 2012-2016 included
• Enhanced feature engineering (19 features)
• Time series cross-validation (5 splits)
• Gradient Boosting with optimized parameters
• Rolling statistics for trend capture
• Proper seasonal encoding for Nepal's patterns
• Realistic test set evaluation (holdout last 12 months)
```

### Top Feature Importances

| Feature | Importance | Description |
|---------|------------|-------------|
| lag_12 | 84.36% | Same month last year |
| lag_1 | 8.48% | Previous month |
| trend_squared | 1.90% | Quadratic trend |
| rolling_mean_12 | 1.71% | 12-month average |
| lag_2 | 1.27% | Two months ago |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13 or higher (tested on 3.13)
- pip package manager
- 500MB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/urja-ai.git
cd urja-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Option 1: Run Jupyter Notebook
jupyter notebook URJA_AI.ipynb
# Run all cells

# Option 2: The model is pre-trained in models/nepal_2024/
```

### Start Web Application

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## 🌐 Web Interface Features

### Dashboard Cards
- **Latest Demand:** Most recent monthly electricity demand
- **Model Accuracy:** R² score percentage
- **Forecast Error:** MAPE percentage
- **Data Range:** Number of historical records

### Interactive Charts
1. **Historical + Forecast Chart**
   - View 2, 3, 5 years or all historical data
   - Generate 6, 12, or 24 month forecasts
   - Hover for exact values

2. **Annual Demand Bar Chart**
   - Fiscal year breakdown (Nepal: Jul-Jun)
   - Total demand in TWh

### Forecast Summary
- Average predicted demand
- Peak month prediction
- Minimum month prediction
- Forecast date range

---

## 📁 Project Structure

```
urja-ai/
│
├── 📄 app.py                           # Flask web server
├── 📓 URJA_AI.ipynb                    # Training notebook (2012-2025)
├── 📋 requirements.txt                 # Python dependencies
├── 📜 LICENSE                          # MIT License
├── 📖 README.md                        # This file
│
├── 📂 data/
│   └── nepal_electricity_demand.csv    # 161 monthly records (2012-2025)
│
├── 📂 models/nepal_2024/
│   ├── nepal_load_forecast_model.joblib # Trained model
│   ├── scaler_X.pkl                    # Feature scaler
│   ├── scaler_y.pkl                    # Target scaler
│   ├── metrics.json                    # Performance metrics
│   ├── config.json                     # Model configuration
│   ├── forecast_12months.csv           # Pre-generated forecast
│   ├── demand_analysis.png             # Data visualization
│   ├── training_results.png            # Model evaluation plots
│   └── forecast_plot.png               # Forecast visualization
│
└── 📂 templates/
    └── index.html                      # Web interface
```

---

## 🔧 Technical Details

### Algorithm
- **Model:** GradientBoostingRegressor (scikit-learn)
- **Estimators:** 200 trees
- **Max Depth:** 5
- **Learning Rate:** 0.1

### Features (19 total)

| Category | Features |
|----------|----------|
| **Time** | month, year, quarter, time_idx |
| **Cyclical** | month_sin, month_cos (captures seasonality) |
| **Season** | season (1-4 for Nepal's climate) |
| **Lag** | lag_1, lag_2, lag_3, lag_6, lag_12 |
| **Rolling** | rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_std_6, rolling_std_12 |
| **Trend** | trend, trend_squared |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/forecast?months=12` | GET | Generate forecast |
| `/api/historical?months=36` | GET | Historical data |
| `/api/metrics` | GET | Model performance |
| `/api/annual` | GET | Annual totals by fiscal year |
| `/api/status` | GET | System health check |

---

## 📈 Key Insights from Data

### Seasonal Patterns
- **Peak Months:** June, July, August (monsoon + irrigation + cooling)
- **Low Months:** November, December (winter, less agricultural activity)
- **Variation:** ~40% difference between peak and trough months

### Growth Trends
- **Pre-COVID:** Steady 5-6% annual growth
- **COVID Impact:** Only 0.36% growth in 2019/20
- **Post-COVID Boom:** 13-19% growth in 2020-2022
- **Recent:** Stabilizing at 7-11% annual growth

### Demand Drivers (Nepal-specific)
1. **Agriculture:** Irrigation pumps during monsoon
2. **Industry:** Manufacturing growth
3. **Residential:** Electrification reaching rural areas
4. **Export:** Selling surplus hydropower to India

---

## 🔮 Sample Forecast Output

```
12-MONTH ELECTRICITY DEMAND FORECAST
====================================
August 2025:    1,277.55 GWh
September 2025: 1,277.54 GWh
October 2025:   1,273.70 GWh
November 2025:  1,087.60 GWh  ← Winter dip
December 2025:  1,129.59 GWh
January 2026:   1,309.56 GWh
February 2026:  1,312.36 GWh  ← Peak
March 2026:     1,155.22 GWh
April 2026:     1,295.67 GWh
May 2026:       1,301.97 GWh
June 2026:      1,308.99 GWh
July 2026:      1,276.40 GWh

Average: 1,250.51 GWh/month
```

---

## 📚 Data Source

**Nepal Electricity Authority (NEA)** - Official government electricity data

Referenced from: [Investopaper - An Analytical Study on Electricity Demand in Nepal](https://www.investopaper.com/news/an-analytical-study-on-electricity-demand-in-nepal/)

The dataset includes:
- Monthly electricity demand in GWh
- Fiscal year classification (Nepal follows mid-July to mid-July)
- 9 complete fiscal years of data

---

## 🛠️ Dependencies

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
gunicorn>=21.0.0
python-dotenv>=1.0.0
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Issues & Support

For bugs, questions, or suggestions:
- Open an [Issue](https://github.com/adhikarisubodh9991/urja-ai/issues)
- Check existing issues first

---

## 🙏 Acknowledgments

- Nepal Electricity Authority for the data
- Investopaper for the analysis and data compilation
- scikit-learn team for the ML library
- Flask team for the web framework

---

<p align="center">
  Made with ⚡ for Nepal's energy future
</p>
