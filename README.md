# Diamond Price Prediction System

## Overview
The Diamond Price Prediction System is a comprehensive machine learning application that predicts diamond prices based on their physical and qualitative attributes (carat weight, cut, color, clarity, dimensions, etc.). The system features an interactive GUI built with CustomTkinter, allowing users to explore datasets, preprocess data, train multiple ML models, and make real-time price predictions without requiring programming knowledge.

## Features
- **Interactive GUI Dashboard** – Modern dark-themed interface with intuitive controls
- **Data Exploration** – Load and visualize diamond datasets with statistical analysis
- **Data Preprocessing** – Automated feature scaling and data preparation
- **Multiple ML Models** – Linear Regression and Random Forest algorithms for price prediction
- **Model Training & Evaluation** – Train models with customizable parameters and view performance metrics (R² score, MSE, MAE)
- **Real-time Price Prediction** – Input diamond attributes and get instant price predictions
- **Data Visualization** – Matplotlib and Seaborn-based charts for exploratory analysis
- **Performance Metrics** – Detailed model evaluation and accuracy reporting

## Technologies Used
- **Python** – Core programming language
- **CustomTkinter** – Modern GUI framework (dark theme)
- **Scikit-learn** – Machine learning algorithms and preprocessing
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization

## Working Principle

### Data Processing Pipeline
1. **Data Loading** – Import diamond dataset (CSV format) with attributes like carat, cut, color, clarity, depth, table, x, y, z dimensions, and price
2. **Feature Engineering** – Encode categorical features (cut, color, clarity) and normalize numerical attributes
3. **Train-Test Split** – Divide dataset (default 80-20 split) for model training and validation
4. **Feature Scaling** – Standardize features using StandardScaler for optimal model performance

### Machine Learning Models
- **Linear Regression** – Captures linear relationships between diamond attributes and price
- **Random Forest Regressor** – Ensemble method for capturing complex non-linear patterns

### Prediction Workflow
1. User inputs diamond attributes through GUI
2. Features are preprocessed (scaled/encoded) using trained transformers
3. Selected model generates price prediction
4. Results displayed with confidence metrics

## Hardware Components
- **Computer/Laptop** – Any modern system with Python 3.7+
- **Display Monitor** – For GUI interaction
- **Keyboard & Mouse** – User input devices

## Software Used
- **Python 3.7+** – Programming runtime
- **CustomTkinter** – UI Framework
- **Scikit-learn** – ML models and preprocessing tools
- **Pandas** – Data manipulation library
- **Matplotlib & Seaborn** – Visualization libraries
- **NumPy** – Numerical operations
- **Git** – Version control (optional)

## Project Structure
```
Diamond-Price-Prediction-System/
├── README.md
├── Diamond_Price_Prediction_GUI/
│   ├── readme.MD
│   ├── requirements.txt
│   ├── src/
│   │   └── app.py                 # Main GUI application
│   └── Data/
│       └── diamonds.csv           # Diamond dataset
```

## Installation & Setup

### Prerequisites
- Python 3.7 or higher installed on your system

### Steps
1. Clone or download the project repository
2. Navigate to the project directory:
   ```bash
   cd Diamond_Price_Prediction_GUI
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python src/app.py
   ```

## Results
The system successfully predicts diamond prices with high accuracy by:
- Analyzing multiple diamond attributes to identify price patterns
- Achieving strong R² scores on test datasets
- Providing instant predictions through an accessible GUI interface
- Enabling users to compare predictions from different ML models
- Visualizing data distributions and model performance metrics

## Future Improvements
- **Advanced Models** – Implement Gradient Boosting, XGBoost, or Neural Networks for enhanced predictions
- **Feature Engineering** – Add polynomial features and interaction terms for better pattern capture
- **Hyperparameter Optimization** – Implement grid search/random search for model tuning
- **Data Upload** – Enable users to upload custom CSV files
- **Export Functionality** – Save predictions and model reports to files
- **Cross-Validation** – Implement k-fold cross-validation for robust evaluation
- **API Integration** – Deploy as REST API for programmatic access
- **Model Persistence** – Save trained models for reuse without retraining
- **Enhanced Visualizations** – 3D plots, correlation heatmaps, feature importance analysis
- **Batch Predictions** – Process multiple diamond predictions simultaneously

## Author
Abdul Rafay

---

**License**: Open Source  
**Last Updated**: May 2026
