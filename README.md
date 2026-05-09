# Student Churn Prediction Model

This project predicts student churn (the likelihood of dropping out) by analyzing financial, academic, and attendance data. It utilizes a trained machine learning model to identify at-risk students before they leave.

---

## Folder Structure

```text
Student Churn Prediction Model/
│
├── Main_Notebook.ipynb       # Primary development notebook with sequential project steps
├── main.py                   # Driver code for running predictions
├── churn_model.pkl           # Trained ML model serialization
├── dependencies.txt          # Required Python packages
├── README.md                 # Project documentation
│
├── data/                     # Input CSV files for processing
│   ├── invoices.csv
│   ├── marks.csv
│   ├── attendance1.csv
│   ├── attendance2.csv
│   ├── attendance3.csv
│   ├── attendance4.csv
│   └── Ultimate1.csv         # Pre-transformed dataset used for training
│
└── output/                   # Generated results and visualizations
    ├── predictions.csv       # Final churn predictions
    ├── tree_visualization.png # Visualization of the model's decision pathway
    ├── feature_importance.png # Graph showing the impact of each variable
    └── Student_Churn_Dashboard.pbix  # PowerBI Dashboard for summary statistics

```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sami-shah-210504/My-Projects.git
cd "My-Projects/AI or ML Projects/Student Churn Prediction Model"

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r dependencies.txt

```

### 3. Prepare Input Files

Place your raw CSV data files into the `data/` folder. Ensure the filenames match those listed in the requirements below.

---

## Running the Project

### 1. Run Predictions

To process the data and generate results, execute:

```bash
python main.py

```

### 2. View Results

* **Data:** Predictions will be exported to `output/predictions.csv`.
* **Visuals:** Check `output/feature_importance.png` and `output/tree_visualization.png` for model insights.
* **Analytics:** Open the `.pbix` file in PowerBI for high-level reporting.

---

## Important Data Requirements

For the model to function correctly, the input CSVs must follow this strict schema:

### 1. Attendance Data

* **Files:** Must be provided as four separate files: `attendance1.csv`, `attendance2.csv`, `attendance3.csv`, and `attendance4.csv`.
* **Required Columns:** `ATT_STATUS`, `ATT_DATE`, `SYSTEM_ID`.

### 2. Invoice Data

* **File Name:** `invoices.csv`
* **Required Columns:** `PAID_DATE`, `DUE_DATE`, `SYSTEM_ID`.

### 3. Grades Data

* **File Name:** `marks.csv`
* **Required Columns:** `TERM_NAME`, `SUBJECT_NAME`, `OBTAINED_MARKS`, `TOTAL_MARKS`, `SYSTEM_ID`.
