# Diamond Price Prediction Using Machine Learning with GUI

**Problem statement**

Predict the selling price of diamonds from their physical and qualitative attributes (e.g., carat, cut, color, clarity, dimensions). The application provides an interactive GUI for data exploration, preprocessing, model training and price prediction to make the workflow accessible to non-programmers.

---

**Dataset description**

- Source: `Data/diamonds.csv` (commonly used diamonds dataset)
- Typical columns: `price` (target), `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`, etc.
- Notes: the app expects the file at `data/diamonds.csv` or `Data/diamonds.csv` in the project root.

---

**Methodology**

1. Data loading and exploratory preview (first 10 rows shown in the GUI).
2. Basic preprocessing:
   - Drop rows with missing essential values (simple strategy; can be extended).
   - Encode categorical features (`cut`, `color`, `clarity`) via one-hot encoding.
   - Separate features (`X`) and target (`y` = `price`).
   - Split into train/test sets (80/20, `random_state=42`).
   - Scale numeric features (e.g., `carat`, `depth`, `table`, `x`, `y`, `z`) using `StandardScaler` fit on training data.
3. Models trained in the app:
   - Linear Regression
   - Random Forest Regressor (default: 100 trees)
4. Evaluation metrics: **R2**, **RMSE**, **MAE**. Results are shown in the GUI comparison table.
5. Prediction: user can supply input values (carat, cut, color, clarity, depth, table, x, y, z) and the app uses the best-performing trained model (by R2) to predict price.

---

**Tools & libraries**

- GUI: `customtkinter`, built on top of `tkinter`
- Data handling: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine learning: `scikit-learn` (models, preprocessing, metrics)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

**Results**

After training (Modeling tab) the app shows a comparison table with columns: **Model Name | R2 Score | RMSE | MAE**. Example (values will vary depending on dataset/version):

| Model Name | R2 Score | RMSE | MAE |
|---|---:|---:|---:|
| LinearRegression | 0.85 | 1234.56 | 987.65 |
| RandomForestRegressor | 0.92 | 823.45 | 654.32 |

> The actual values above are placeholders. Run **Preprocess Data** → **Train Models** in the app to compute real results on your copy of the dataset.

---

**Conclusion**

This mini-project demonstrates an end-to-end workflow for tabular regression problems with a simple GUI: data loading, preprocessing, exploratory preview, statistical plots, model training, evaluation and interactive prediction. The app is a good base for further improvements such as advanced imputation, feature engineering, hyperparameter tuning, cross-validation and model persistence/export.

---

**Usage**

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python src/app.py`
3. Use the **Load Dataset** button to load `diamonds.csv`.
4. Optionally explore statistical plots in **Statistical Analysis**.
5. Click **Preprocess Data**, then **Modeling** → **Train Models**.
6. Use the **Predict Price** inputs to make predictions with the best model.

---

**License**

MIT-style — adapt as needed.
