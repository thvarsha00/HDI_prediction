# Project Workflow Diagram — HDI Prediction System

## End-to-End Pipeline

```mermaid
flowchart TD
    A[Kaggle HDI Dataset<br/>hdi_dataset.csv] --> B[Data Loading & Understanding<br/>pandas.read_csv, .info, .describe]
    B --> C[Exploratory Data Analysis<br/>Heatmap, Distributions, Scatter, Pairplot]
    C --> D[Data Preprocessing<br/>Missing values, Duplicates, Label Encoding]
    D --> E[Feature Selection<br/>life_expectancy, education_index,<br/>income_index, gni_per_capita]
    E --> F[Train-Test Split<br/>80% train / 20% test]
    F --> G1[Linear Regression]
    F --> G2[Random Forest Regression]
    F --> G3[Decision Tree Regression]
    G1 --> H[Model Evaluation<br/>MAE, MSE, RMSE, R2]
    G2 --> H
    G3 --> H
    H --> I[Select Best Model]
    I --> J[Save Model<br/>model/hdi_model.pkl via Pickle]
    J --> K[Flask Web Application<br/>app.py]
    K --> L[User Fills Input Form<br/>index.html]
    L --> M[Model Loaded & Prediction Generated]
    M --> N[Result Page<br/>result.html<br/>HDI Score + Category]
```

## Request Flow Inside the Flask App

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Flask as Flask App (app.py)
    participant Model as hdi_model.pkl

    User->>Browser: Opens homepage
    Browser->>Flask: GET /
    Flask->>Browser: Renders index.html (input form)
    User->>Browser: Fills indicators & clicks "Predict HDI Score"
    Browser->>Flask: POST /predict (form data)
    Flask->>Flask: Validate input ranges
    Flask->>Model: model.predict(input_features)
    Model-->>Flask: predicted_hdi
    Flask->>Flask: get_hdi_category(predicted_hdi)
    Flask->>Browser: Renders result.html (score + category)
    Browser->>User: Displays prediction result
```

> Render these diagrams by pasting the Mermaid code into
> [mermaid.live](https://mermaid.live) or viewing this file in VS Code with
> the "Markdown Preview Mermaid Support" extension.
