"""Basic CustomTkinter application window for Diamond Price Prediction GUI.

- Dark modern theme
- Window size: 1200x700
- Title: "Diamond Price Prediction – ML GUI"

No machine learning code included yet.
"""

import customtkinter as ctk
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Use modern dark appearance
ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
ctk.set_default_color_theme("dark-blue")  # Prebuilt themes: "blue", "dark-blue", "green"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Diamond Price Prediction – ML GUI")
        self.geometry("1200x700")
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkLabel(self, text="Diamond Price Prediction – ML GUI", font=("Helvetica", 20, "bold"))
        header.grid(row=0, column=0, padx=20, pady=10, sticky="n")

        # Controls frame (Load button + info)
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(2, weight=0)

        load_btn = ctk.CTkButton(controls, text="Load Dataset", command=self.load_data)
        load_btn.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")

        preprocess_btn = ctk.CTkButton(controls, text="Preprocess Data", command=self.preprocess_data)
        preprocess_btn.grid(row=0, column=2, padx=(10, 0), pady=5, sticky="e")

        info_frame = ctk.CTkFrame(controls)
        info_frame.grid(row=0, column=1, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)

        # Variables to display dataset info
        self.row_var = tk.StringVar(value="Rows: -")
        self.col_var = tk.StringVar(value="Columns: -")
        self.columns_var = tk.StringVar(value="Column names: -")
        self.split_var = tk.StringVar(value="Train/Test: - / -")

        rows_lbl = ctk.CTkLabel(info_frame, textvariable=self.row_var, anchor="w")
        rows_lbl.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        cols_lbl = ctk.CTkLabel(info_frame, textvariable=self.col_var, anchor="w")
        cols_lbl.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        colsnames_lbl = ctk.CTkLabel(info_frame, textvariable=self.columns_var, anchor="w", wraplength=900, justify="left")
        colsnames_lbl.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        split_lbl = ctk.CTkLabel(info_frame, textvariable=self.split_var, anchor="w")
        split_lbl.grid(row=3, column=0, sticky="w", padx=8, pady=2)

        rows_lbl = ctk.CTkLabel(info_frame, textvariable=self.row_var, anchor="w")
        rows_lbl.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        cols_lbl = ctk.CTkLabel(info_frame, textvariable=self.col_var, anchor="w")
        cols_lbl.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        colsnames_lbl = ctk.CTkLabel(info_frame, textvariable=self.columns_var, anchor="w", wraplength=900, justify="left")
        colsnames_lbl.grid(row=2, column=0, sticky="w", padx=8, pady=2)

        # Content area with tabs: Preview and Statistical Analysis
        content_frame = ctk.CTkFrame(self, corner_radius=8)
        content_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Tabview with Preview and Statistical Analysis tabs
        tabview = ctk.CTkTabview(content_frame, width=1000)
        tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        tabview.add("Preview")
        tabview.add("Statistical Analysis")

        # ---- Preview tab ----
        preview_tab = tabview.tab("Preview")
        preview_label = ctk.CTkLabel(preview_tab, text="Preview (first 10 rows)", font=("Helvetica", 14, "bold"))
        preview_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        table_frame = ctk.CTkFrame(preview_tab)
        table_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Use ttk.Treeview for tabular preview
        self.tree = ttk.Treeview(table_frame, columns=(), show="headings")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # ---- Statistical Analysis tab ----
        stats_tab = tabview.tab("Statistical Analysis")
        stats_tab.grid_rowconfigure(0, weight=0)
        stats_tab.grid_rowconfigure(1, weight=1)
        stats_tab.grid_columnconfigure(0, weight=1)

        btn_frame = ctk.CTkFrame(stats_tab, fg_color="transparent")
        btn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        hist_btn = ctk.CTkButton(btn_frame, text="Histogram: Price", command=self.plot_histogram)
        hist_btn.grid(row=0, column=0, padx=6, pady=6)
        box_btn = ctk.CTkButton(btn_frame, text="Boxplot: Price vs Cut", command=self.plot_boxplot)
        box_btn.grid(row=0, column=1, padx=6, pady=6)
        scatter_btn = ctk.CTkButton(btn_frame, text="Scatter: Carat vs Price", command=self.plot_scatter)
        scatter_btn.grid(row=0, column=2, padx=6, pady=6)

        # Plot display area
        plot_frame = ctk.CTkFrame(stats_tab)
        plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_container = plot_frame

        # ---- Modeling tab ----
        tabview.add("Modeling")
        model_tab = tabview.tab("Modeling")
        model_tab.grid_rowconfigure(0, weight=0)
        model_tab.grid_rowconfigure(1, weight=0)
        model_tab.grid_rowconfigure(2, weight=1)
        model_tab.grid_columnconfigure(0, weight=1)

        model_btn_frame = ctk.CTkFrame(model_tab, fg_color="transparent")
        model_btn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        train_btn = ctk.CTkButton(model_btn_frame, text="Train Models (Linear, RF)", command=self.train_models)
        train_btn.grid(row=0, column=0, padx=6, pady=6)

        results_label = ctk.CTkLabel(model_tab, text="Model Results", font=("Helvetica", 14, "bold"))
        results_label.grid(row=1, column=0, padx=10, pady=(6, 0), sticky="w")

        results_frame = ctk.CTkFrame(model_tab)
        results_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(results_frame, columns=("model","r2","rmse","mae"), show="headings")
        self.results_tree.heading("model", text="Model Name")
        self.results_tree.heading("r2", text="R2 Score")
        self.results_tree.heading("rmse", text="RMSE")
        self.results_tree.heading("mae", text="MAE")
        self.results_tree.column("model", width=220)
        self.results_tree.column("r2", width=140)
        self.results_tree.column("rmse", width=140)
        self.results_tree.column("mae", width=140)

        r_vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        r_hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=r_vsb.set, xscrollcommand=r_hsb.set)

        self.results_tree.grid(row=0, column=0, sticky="nsew")
        r_vsb.grid(row=0, column=1, sticky="ns")
        r_hsb.grid(row=1, column=0, sticky="ew")

        self.results_df = None

        # ---- Prediction controls ----
        pred_frame = ctk.CTkFrame(model_tab)
        pred_frame.grid(row=3, column=0, padx=10, pady=(6,10), sticky="ew")
        pred_frame.grid_columnconfigure(0, weight=1)
        pred_frame.grid_columnconfigure(1, weight=1)

        # Input vars
        self.input_vars = {
            'carat': tk.StringVar(value='0.0'),
            'cut': tk.StringVar(value='None'),
            'color': tk.StringVar(value='None'),
            'clarity': tk.StringVar(value='None'),
            'depth': tk.StringVar(value='0.0'),
            'table': tk.StringVar(value='0.0'),
            'x': tk.StringVar(value='0.0'),
            'y': tk.StringVar(value='0.0'),
            'z': tk.StringVar(value='0.0'),
        }

        left = ctk.CTkFrame(pred_frame, fg_color='transparent')
        left.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        right = ctk.CTkFrame(pred_frame, fg_color='transparent')
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)

        # Numeric inputs
        ctk.CTkLabel(left, text='Carat').grid(row=0, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['carat']).grid(row=0, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(left, text='Depth').grid(row=1, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['depth']).grid(row=1, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(left, text='Table').grid(row=2, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['table']).grid(row=2, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(left, text='x').grid(row=3, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['x']).grid(row=3, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(left, text='y').grid(row=4, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['y']).grid(row=4, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(left, text='z').grid(row=5, column=0, sticky='w')
        ctk.CTkEntry(left, textvariable=self.input_vars['z']).grid(row=5, column=1, sticky='ew', padx=6)

        # Categorical inputs (OptionMenus)
        ctk.CTkLabel(right, text='Cut').grid(row=0, column=0, sticky='w')
        self.cut_menu = ctk.CTkOptionMenu(right, values=['None'], variable=self.input_vars['cut'])
        self.cut_menu.grid(row=0, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(right, text='Color').grid(row=1, column=0, sticky='w')
        self.color_menu = ctk.CTkOptionMenu(right, values=['None'], variable=self.input_vars['color'])
        self.color_menu.grid(row=1, column=1, sticky='ew', padx=6)
        ctk.CTkLabel(right, text='Clarity').grid(row=2, column=0, sticky='w')
        self.clarity_menu = ctk.CTkOptionMenu(right, values=['None'], variable=self.input_vars['clarity'])
        self.clarity_menu.grid(row=2, column=1, sticky='ew', padx=6)

        predict_btn = ctk.CTkButton(pred_frame, text='Predict Price', command=self.predict_price)
        predict_btn.grid(row=1, column=0, columnspan=2, pady=8)

        self.prediction_var = tk.StringVar(value='Predicted price: -')
        self.prediction_lbl = ctk.CTkLabel(pred_frame, textvariable=self.prediction_var, font=("Helvetica", 12, "bold"))
        self.prediction_lbl.grid(row=2, column=0, columnspan=2, pady=(0,6))

        # Data holder
        self.data = None
        self._canvas = None

    def load_data(self):
        """Load dataset from data/diamonds.csv (or Data/diamonds.csv) and update UI with basic info."""
        candidates = [Path("data/diamonds.csv"), Path("Data/diamonds.csv")]
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                except Exception as e:
                    messagebox.showerror("Load error", f"Error reading {p}: {e}")
                    return
                self.data = df
                self.row_var.set(f"Rows: {len(df)}")
                self.col_var.set(f"Columns: {len(df.columns)}")
                self.columns_var.set("Column names: " + ", ".join(df.columns.astype(str).tolist()))
                self.populate_table(df)
                self.update_category_options()
                return

        messagebox.showerror("File not found", "Could not find 'diamonds.csv' in 'data/' or 'Data/' directory.")

    def preprocess_data(self):
        """Encode categorical features (cut, color, clarity), split into train/test (80/20), and scale numeric features."""
        if self.data is None:
            messagebox.showerror("No data", "Please load a dataset first.")
            return
        # Required columns
        required = {"price", "cut", "color", "clarity"}
        missing = required - set(self.data.columns)
        if missing:
            messagebox.showerror("Missing columns", f"Missing required columns: {', '.join(sorted(missing))}")
            return
        # Work on a copy
        df = self.data.copy()
        # Drop rows with missing essential values (price, carat, or categorical features)
        essential = ["price", "carat", "cut", "color", "clarity"]
        df = df.dropna(subset=[c for c in essential if c in df.columns])

        # Separate X and y
        y = df["price"].copy()
        X = df.drop(columns=["price"]).copy()

        # One-hot encode categorical features
        cat_cols = [c for c in ["cut", "color", "clarity"] if c in X.columns]
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Identify numeric columns to scale
        numeric_candidates = ["carat", "depth", "table", "x", "y", "z"]
        numeric_cols = [c for c in numeric_candidates if c in X.columns]
        if not numeric_cols:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # store numeric columns for later prediction
        self.numeric_cols = numeric_cols

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numeric columns
        scaler = StandardScaler()
        if numeric_cols:
            X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train.loc[:, numeric_cols])
            X_test.loc[:, numeric_cols] = scaler.transform(X_test.loc[:, numeric_cols])

        # Store preprocessing artifacts
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.preprocessed = True

        # Update UI
        self.split_var.set(f"Train/Test: {len(X_train)} / {len(X_test)}")
        messagebox.showinfo("Preprocessing complete", f"Preprocessing finished.\nTrain/Test sizes: {len(X_train)} / {len(X_test)}")

    def populate_table(self, df):
        """Populate the Treeview with the first 10 rows of dataframe."""
        if not hasattr(self, 'tree'):
            return
        preview = df.head(10)
        cols = [str(c) for c in preview.columns]
        # configure columns in treeview
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="w", minwidth=50)
        # clear previous rows
        for r in self.tree.get_children():
            self.tree.delete(r)
        # insert new rows
        for _, row in preview.iterrows():
            values = [row[col] if pd.notnull(row[col]) else "" for col in cols]
            values = [str(v) for v in values]
            self.tree.insert("", "end", values=values)

    # -------------------- Statistical plotting --------------------
    def plot_histogram(self):
        """Plot histogram of diamond prices."""
        if self.data is None:
            messagebox.showerror("No data", "Please load a dataset first.")
            return
        if "price" not in self.data.columns:
            messagebox.showerror("Missing column", "Column 'price' not found in dataset.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.data["price"].dropna(), bins=30, kde=True, ax=ax, color="#4c72b0")
        ax.set_title("Histogram of Diamond Prices")
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        self.show_plot(fig)

    def plot_boxplot(self):
        """Plot boxplot of price grouped by cut."""
        if self.data is None:
            messagebox.showerror("No data", "Please load a dataset first.")
            return
        if not {"price", "cut"}.issubset(self.data.columns):
            messagebox.showerror("Missing column", "Columns 'price' and/or 'cut' not found in dataset.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="cut", y="price", data=self.data, ax=ax)
        ax.set_title("Boxplot of Price by Cut")
        ax.set_xlabel("Cut")
        ax.set_ylabel("Price")
        self.show_plot(fig)

    def plot_scatter(self):
        """Plot scatter of carat vs price."""
        if self.data is None:
            messagebox.showerror("No data", "Please load a dataset first.")
            return
        if not {"carat", "price"}.issubset(self.data.columns):
            messagebox.showerror("Missing column", "Columns 'carat' and/or 'price' not found in dataset.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x="carat", y="price", data=self.data, ax=ax, s=25, color="#dd8452")
        ax.set_title("Scatter: Carat vs Price")
        ax.set_xlabel("Carat")
        ax.set_ylabel("Price")
        self.show_plot(fig)

    def show_plot(self, fig):
        """Embed a matplotlib Figure into the plot container (clears previous)."""
        # remove existing canvas if present
        if hasattr(self, "_canvas") and self._canvas is not None:
            try:
                self._canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self._canvas = None
        # draw and embed
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self._canvas = canvas
        # close the figure in matplotlib to free memory
        plt.close(fig)

    def train_models(self):
        """Train Linear Regression and Random Forest, evaluate on test set, and store results."""
        if not getattr(self, 'preprocessed', False):
            messagebox.showerror("No preprocessed data", "Please preprocess the data before training models.")
            return
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        results = []

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"model": "LinearRegression", "r2": r2, "rmse": rmse, "mae": mae})

        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"model": "RandomForestRegressor", "r2": r2, "rmse": rmse, "mae": mae})

        # store trained models for later prediction
        self.trained_models = {"LinearRegression": lr, "RandomForestRegressor": rf}

        self.results_df = pd.DataFrame(results)
        self.populate_results_table()
        messagebox.showinfo("Training complete", "Models trained and evaluated. Results updated in Modeling tab.")

    def populate_results_table(self):
        """Populate the results Treeview with stored metrics."""
        if self.results_df is None:
            return
        # clear previous
        for r in self.results_tree.get_children():
            self.results_tree.delete(r)
        for _, row in self.results_df.iterrows():
            self.results_tree.insert("", "end", values=(row['model'], f"{row['r2']:.4f}", f"{row['rmse']:.4f}", f"{row['mae']:.4f}"))

    def update_category_options(self):
        """Update option menu values for categorical inputs based on loaded data."""
        if self.data is None:
            return
        for col, menu in [("cut", getattr(self, 'cut_menu', None)), ("color", getattr(self, 'color_menu', None)), ("clarity", getattr(self, 'clarity_menu', None))]:
            if menu is None:
                continue
            vals = sorted(self.data[col].dropna().unique().astype(str).tolist()) if col in self.data.columns else ['None']
            if not vals:
                vals = ['None']
            # ensure current value is in options
            current = self.input_vars[col].get() if col in self.input_vars else None
            menu.configure(values=vals)
            if current not in vals:
                self.input_vars[col].set(vals[0])

    def predict_price(self):
        """Use best-performing trained model to predict price from input fields."""
        if not hasattr(self, 'results_df') or self.results_df is None:
            messagebox.showerror("No models", "No trained models found. Please train models first.")
            return
        if not hasattr(self, 'trained_models'):
            messagebox.showerror("No models", "No trained models found. Please train models first.")
            return
        # Determine best model by highest R2
        best_row = self.results_df.loc[self.results_df['r2'].idxmax()]
        best_model_name = best_row['model']
        model = self.trained_models.get(best_model_name)
        if model is None:
            messagebox.showerror("Model error", "Best model not available for prediction.")
            return
        # Build input row aligned with training columns
        X_cols = list(self.X_train.columns)
        x_row = pd.DataFrame(columns=X_cols, index=[0])
        x_row.loc[0] = 0.0
        # Fill numeric features
        for num in getattr(self, 'numeric_cols', []):
            if num in self.input_vars:
                try:
                    val = float(self.input_vars[num].get())
                except Exception:
                    val = 0.0
                x_row.loc[0, num] = val
        # Handle categorical one-hot columns
        for cat in ['cut', 'color', 'clarity']:
            sel = self.input_vars[cat].get() if cat in self.input_vars else None
            for col in X_cols:
                if col.startswith(f"{cat}_"):
                    suffix = col.split("_", 1)[1]
                    x_row.loc[0, col] = 1.0 if sel == suffix else 0.0
        # Scale numeric columns
        if hasattr(self, 'numeric_cols') and self.numeric_cols:
            try:
                x_row.loc[:, self.numeric_cols] = self.scaler.transform(x_row.loc[:, self.numeric_cols])
            except Exception as e:
                messagebox.showerror("Scaling error", f"Error applying scaler: {e}")
                return
        # Ensure column order
        x_row = x_row[X_cols]
        # Predict
        try:
            pred = model.predict(x_row)[0]
            self.prediction_var.set(f"Predicted price: {pred:,.2f}")
        except Exception as e:
            messagebox.showerror("Prediction error", f"Prediction failed: {e}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
