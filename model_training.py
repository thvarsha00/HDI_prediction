

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")  # Allows plot generation without a GUI window (needed on servers)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "hdi_dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "hdi_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PLOTS_DIR = os.path.join("static", "plots")

FEATURE_COLUMNS = ["life_expectancy", "education_index", "income_index", "gni_per_capita"]
TARGET_COLUMN = "hdi"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# EPIC 3: DATASET LOADING & UNDERSTANDING
# ---------------------------------------------------------------------------
def load_and_explore_data(path):
    print("\n" + "=" * 70)
    print("EPIC 3: DATASET LOADING AND UNDERSTANDING")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\n[1] First 5 rows of the dataset:")
    print(df.head())

    print("\n[2] Dataset shape (rows, columns):", df.shape)

    print("\n[3] Dataset info:")
    print(df.info())

    print("\n[4] Data types:")
    print(df.dtypes)

    print("\n[5] Statistical summary (numerical columns):")
    print(df.describe())

    print("\n[6] Missing values per column:")
    print(df.isnull().sum())

    print("\n[7] Duplicate rows found:", df.duplicated().sum())

    print("\n[8] Features identified:", FEATURE_COLUMNS)
    print("    Target variable identified:", TARGET_COLUMN)

    return df


def perform_eda(df):
    print("\nGenerating EDA visualizations in '%s' ..." % PLOTS_DIR)

    numeric_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    # 1. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap of HDI Features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()

    # 2. Distribution plots for each numeric feature
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS + [TARGET_COLUMN]):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {col}")
    for j in range(len(FEATURE_COLUMNS) + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "distribution_plots.png"))
    plt.close()

    # 3. Scatter plots: each feature vs HDI
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        sns.scatterplot(x=df[col], y=df[TARGET_COLUMN], ax=axes[i], color="darkorange")
        axes[i].set_title(f"{col} vs HDI")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scatter_plots.png"))
    plt.close()

    # 4. Pairplot for overall feature relationships
    pairplot_fig = sns.pairplot(numeric_df.dropna())
    pairplot_fig.savefig(os.path.join(PLOTS_DIR, "feature_relationships.png"))
    plt.close()

    print("EDA visualizations saved successfully.")


# ---------------------------------------------------------------------------
# EPIC 4: DATA PREPROCESSING AND LABEL ENCODING
# ---------------------------------------------------------------------------
def preprocess_data(df):
    print("\n" + "=" * 70)
    print("EPIC 4: DATA PREPROCESSING AND LABEL ENCODING")
    print("=" * 70)

    df = df.copy()

    # 1. Handle missing values -- fill numeric columns with column median
    print("\n[1] Missing values BEFORE handling:\n", df.isnull().sum())
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    print("\n[1] Missing values AFTER handling:\n", df.isnull().sum())

    # 2. Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"\n[2] Removed {before - after} duplicate rows. New shape: {df.shape}")

    # 3. Label Encoding for categorical column ('continent')
    if "continent" in df.columns:
        le = LabelEncoder()
        df["continent_encoded"] = le.fit_transform(df["continent"])
        print("\n[3] Label encoding applied to 'continent' column.")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

    # 4. Feature selection -- keep only relevant columns for modelling
    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    print("\n[4] Final feature set used for modelling:", FEATURE_COLUMNS)

    return df, model_df


# ---------------------------------------------------------------------------
# EPIC 5: TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
def split_data(model_df):
    print("\n" + "=" * 70)
    print("EPIC 5: TRAIN-TEST SPLIT")
    print("=" * 70)

    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining samples: {X_train.shape[0]} (80%)")
    print(f"Testing samples : {X_test.shape[0]} (20%)")

    # Scale features (helps Linear Regression converge to better coefficients)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# EPIC 6: MODEL TRAINING
# ---------------------------------------------------------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                               X_train_scaled, X_test_scaled):
    print("\n" + "=" * 70)
    print("EPIC 6: MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    models = {
        "Linear Regression": (LinearRegression(), True),
        "Random Forest Regression": (RandomForestRegressor(n_estimators=200, random_state=42), False),
        "Decision Tree Regression": (DecisionTreeRegressor(random_state=42, max_depth=6), False),
    }

    results = {}
    trained_models = {}

    for name, (model, use_scaled) in models.items():
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        trained_models[name] = model

        print(f"\n--- {name} ---")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

    results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
    print("\nModel Comparison Table (sorted by R2 score):")
    print(results_df)

    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    best_uses_scaled = models[best_model_name][1]

    print(f"\nBest performing model: {best_model_name}")

    return best_model, best_model_name, best_uses_scaled, results_df


# ---------------------------------------------------------------------------
# EPIC 7: SAVE THE MODEL
# ---------------------------------------------------------------------------
def save_model(model, scaler, uses_scaled, model_name):
    print("\n" + "=" * 70)
    print("EPIC 7: SAVING THE TRAINED MODEL")
    print("=" * 70)

    payload = {
        "model": model,
        "scaler": scaler,
        "uses_scaled_input": uses_scaled,
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nModel saved successfully at: {MODEL_PATH}")


def load_model_for_prediction():
    """
    Utility function showing how to load the saved model and use it for
    future predictions. This mirrors what app.py does inside the Flask app.
    """
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    scaler = payload["scaler"]
    uses_scaled_input = payload["uses_scaled_input"]

    sample_input = pd.DataFrame(
        [[70.0, 0.65, 0.60, 15000.0]], columns=FEATURE_COLUMNS
    )

    if uses_scaled_input:
        sample_input_transformed = scaler.transform(sample_input)
    else:
        sample_input_transformed = sample_input

    prediction = model.predict(sample_input_transformed)[0]
    print(f"\nSample prediction using saved model ({payload['model_name']}): {prediction:.3f}")
    return prediction


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = load_and_explore_data(DATASET_PATH)
    perform_eda(df_raw)

    df_clean, model_df = preprocess_data(df_raw)

    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = split_data(model_df)

    best_model, best_model_name, best_uses_scaled, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    save_model(best_model, scaler, best_uses_scaled, best_model_name)

    load_model_for_prediction()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE -- Model is ready for the Flask app (app.py)")
    print("=" * 70)
