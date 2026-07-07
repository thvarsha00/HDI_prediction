# A Comprehensive Measure of Well-Being
### Human Development Index (HDI) Prediction System

A beginner-friendly, end-to-end Machine Learning project that predicts a
country's **Human Development Index (HDI)** score from socio-economic
indicators — life expectancy, education index, income index, and GNI per
capita — served through a **Flask** web application.

Built for a final-year engineering project submission using **Python,
Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Flask**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Technology Stack](#technology-stack)
4. [Setup Instructions (VS Code)](#setup-instructions-vs-code)
5. [How to Run](#how-to-run)
6. [Machine Learning Workflow](#machine-learning-workflow)
7. [Model Performance](#model-performance)
8. [Flask App Features](#flask-app-features)
9. [Database Design (Optional)](#database-design-optional)
10. [Using a Real Kaggle Dataset](#using-a-real-kaggle-dataset)
11. [Future Enhancements](#future-enhancements)

---

## Project Overview

The **Human Development Index (HDI)** is a composite statistic published by
the UNDP that summarizes a country's average achievement in three key
dimensions: health, education, and standard of living. This project trains
a regression model to *predict* the HDI score from four raw indicators, and
exposes that model through a clean, responsive web interface.

**Objective:** Given `life_expectancy`, `education_index`, `income_index`,
and `gni_per_capita`, predict the `hdi` score (0–1) and classify it into one
of four UNDP bands: **Low, Medium, High, Very High**.

---

## Project Structure

```
HDI_Prediction_System/
│
├── app.py                     # Flask web application
├── model_training.py          # Full ML pipeline: EDA → preprocessing → training → saving
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── ER_DIAGRAM.md              # Database ER diagram + explanation
├── WORKFLOW.md                # Project & request-flow diagrams
│
├── dataset/
│   └── hdi_dataset.csv        # HDI dataset (see note below on real Kaggle data)
│
├── model/
│   └── hdi_model.pkl          # Serialized best-performing model (+ scaler)
│
├── notebooks/
│   └── EDA.ipynb              # Standalone exploratory data analysis notebook
│
├── templates/
│   ├── index.html             # Home page with the prediction form
│   └── result.html            # Prediction result page
│
└── static/
    ├── css/
    │   └── style.css          # Custom styling (Bootstrap 5 based)
    └── plots/                 # Auto-generated EDA charts (created by model_training.py)
```

---

## Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| IDE | VS Code |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Model Serialization | Pickle |
| Frontend | HTML5, CSS3, Bootstrap 5 |
| Database (optional) | SQLite / MySQL |

---

## Setup Instructions (VS Code)

1. **Open the project folder in VS Code**
   ```
   code HDI_Prediction_System
   ```

2. **Create a virtual environment** (recommended)
   ```
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Select the Python interpreter** in VS Code:
   `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the `venv` one.

---

## How to Run

### Step 1 — Train the model
Run the full ML pipeline (EDA + preprocessing + training + evaluation +
saving the model). This must be run at least once before starting the Flask
app:

```
python model_training.py
```

This will:
- Print dataset exploration details (head, info, describe, missing values)
- Generate EDA plots inside `static/plots/`
- Clean the data (missing values, duplicates, label encoding)
- Split data 80/20 and train **Linear Regression, Random Forest, and
  Decision Tree** models
- Print MAE, MSE, RMSE, and R² for each model
- Save the **best model** to `model/hdi_model.pkl`

### Step 2 — Start the Flask app

```
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

### Step 3 — Use the app
1. Fill in the four indicator fields on the home page.
2. Click **Predict HDI Score**.
3. View the predicted HDI value and its development category on the
   result page.

### (Optional) Explore the EDA notebook
```
jupyter notebook notebooks/EDA.ipynb
```

---

## Machine Learning Workflow

1. **Dataset Loading & Understanding** — shape, dtypes, statistics, missing
   values, duplicates.
2. **Exploratory Data Analysis** — correlation heatmap, distribution plots,
   scatter plots, pairwise feature relationships.
3. **Data Preprocessing** — missing value imputation (median), duplicate
   removal, label encoding of the `continent` column.
4. **Feature Selection** — `life_expectancy`, `education_index`,
   `income_index`, `gni_per_capita` → target `hdi`.
5. **Train-Test Split** — 80% training / 20% testing via
   `train_test_split(..., random_state=42)`.
6. **Model Training & Comparison**
   - Linear Regression *(baseline / interpretable model)*
   - Random Forest Regression *(ensemble, usually most accurate)*
   - Decision Tree Regression *(simple non-linear baseline)*
7. **Evaluation Metrics** — MAE, MSE, RMSE, R² Score for each model.
8. **Model Selection & Saving** — the model with the highest R² is
   serialized with **Pickle** to `model/hdi_model.pkl`, together with the
   fitted `StandardScaler` (used only if the winning model needs scaled
   input, e.g. Linear Regression).

See `WORKFLOW.md` for a visual flowchart and sequence diagram.

---

## Model Performance

Example results on the synthetic dataset included in this repo (your exact
numbers may vary slightly by dataset/run):

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Random Forest Regression | ~0.0105 | ~0.0002 | ~0.0131 | **~0.990** |
| Linear Regression | ~0.0122 | ~0.0002 | ~0.0156 | ~0.986 |
| Decision Tree Regression | ~0.0167 | ~0.0005 | ~0.0212 | ~0.973 |

The pipeline automatically selects and saves whichever model scores
highest on R² for the dataset you provide.

---

## Flask App Features

**Home Page (`/`)**
- Project title & short description
- Input form for the four HDI indicators
- Responsive Bootstrap 5 layout with a navigation bar, hero section, and
  informational cards

**Prediction (`POST /predict`)**
- Loads the saved `hdi_model.pkl`
- Validates and accepts user input
- Generates the HDI prediction using the trained model

**Result Page**
- Displays the predicted HDI score (rounded to 3 decimals)
- Displays the HDI category: **Low / Medium / High / Very High**
- Visual progress bar showing where the score falls on the 0–1 scale
- Summary of the inputs used and the model that generated the prediction

---

## Database Design (Optional)

A full ER diagram covering `USER`, `COUNTRY`, `HDI_INPUT_DATA`, `ML_MODEL`,
`HDI_PREDICTION`, `DATASET`, `VISUALIZATION_REPORT`, and `SESSION` entities
is provided in [`ER_DIAGRAM.md`](ER_DIAGRAM.md), including ready-to-run
SQLite `CREATE TABLE` statements. This layer is **optional** — the current
`app.py` runs statelessly without a database — but is included so the
project can be extended to track prediction history per user for a viva
demonstration.

---

## Using a Real Kaggle Dataset

This repo ships with a **synthetic but realistically generated** HDI
dataset (`dataset/hdi_dataset.csv`) so the project runs immediately offline.
To use real data instead:

1. Search Kaggle for **"Human Development Index dataset"** (e.g. UNDP HDI
   datasets covering 190+ countries).
2. Download the CSV and place it at `dataset/hdi_dataset.csv`.
3. Make sure the column names match (rename if needed):
   `country, continent, life_expectancy, education_index, income_index, gni_per_capita, hdi`
4. Re-run `python model_training.py` — everything downstream (EDA,
   preprocessing, training, Flask app) works unchanged.

---

## Future Enhancements

- Add user authentication and prediction history using the ER schema above
- Deploy to Render/Heroku/PythonAnywhere for public access
- Add a world map visualization (Plotly Choropleth) of predicted HDI by
  country
- Add SHAP/feature-importance explanations to the result page
- Add batch prediction via CSV upload
