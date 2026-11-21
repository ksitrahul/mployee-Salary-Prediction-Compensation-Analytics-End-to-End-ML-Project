# Employee Salary Prediction & Compensation Analytics (Jupyter + Streamlit)

## Key Upgrades Included
- **Outlier-Resilient Modeling (`HuberRegressor`)**: Handles right-tail executive/outlier salaries (`> 100 LPA`) without skewing standard salary predictions (`R² = 0.9079`, `MAE = 2.71 LPA`, `Median AE = 2.23 LPA`).
- **80% Confidence Interval Estimation**: Dual `GradientBoostingRegressor` quantile models (`alpha=0.10` and `alpha=0.90`) compute realistic low-to-high compensation bands.
- **What-If Career Simulator**: Simulates salary uplift from promotions, upskilling, advanced degrees, and relocation.
- **HR Batch Payroll Equity Audit**: Flags underpaid employees (retention risk) and above-market outliers against predicted confidence bands.
## Quickstart
```bash
pip install -r requirements.txt
jupyter notebook salary_prediction_pipeline.ipynb
streamlit run app.py
```
