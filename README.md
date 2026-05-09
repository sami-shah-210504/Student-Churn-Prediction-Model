# Student Churn Prediction Model

This project predicts student churn (the likelihood of dropping out) by analyzing financial, academic, and attendance data. The project uses a Random Forest and Gradient Boosting approach to identify at-risk students based on historical patterns.

---

## Folder Structure

```text
Student Churn Prediction Model/
│
├── Main_Notebook.ipynb       # Primary development and EDA notebook
├── main.py                   # Driver code for running predictions
├── dependencies.txt          # Required Python packages
├── README.md                 # Project documentation
├── .gitignore                # Configured to exclude .pkl and data/*.csv
│
├── data/                     # (Local only) Input CSV files for processing
│   └── .gitkeep              # Ensures the directory structure exists in Git
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

```bash
pip install -r dependencies.txt

```

### 3. Download Model Weights

Due to file size, the trained model is hosted on Hugging Face.

* **Download:** [`churn_model.pkl`](https://huggingface.co/samishah2004/Student-Churn-Prediction-Model/tree/main) 
* **Action:** Place the downloaded `.pkl` file directly in the **root directory** of this project.

### 4. Prepare Input Data

Place your raw data into the `data/` folder. For security and storage efficiency, these files are ignored by Git. Ensure filenames match the following:

* **Attendance:** `attendance1.csv` through `attendance4.csv`
* **Invoices:** `invoices.csv`
* **Grades:** `marks.csv`

---

## Running the Project

### Run Predictions

Once the model weights and data files are in place, execute the driver script:

```bash
python main.py

```

* **Results:** Churn probabilities will be saved to `output/predictions.csv`.
* **Insights:** Review `output/feature_importance.png` to see which factors (e.g., attendance vs. grades) most influenced the model.

---

## Data Schema Requirements

The input CSVs must contain the following columns for the preprocessing pipeline to work:

| File Type | Required Columns |
| --- | --- |
| **Attendance** | `ATT_STATUS`, `ATT_DATE`, `SYSTEM_ID` |
| **Invoices** | `PAID_DATE`, `DUE_DATE`, `SYSTEM_ID` |
| **Grades** | `TERM_NAME`, `SUBJECT_NAME`, `OBTAINED_MARKS`, `TOTAL_MARKS`, `SYSTEM_ID` |

---

## Note on Model Hosting

The model weights are tracked externally to keep the repository lightweight. If you update the model, ensure you upload the new version to Hugging Face and update the link in this README.
