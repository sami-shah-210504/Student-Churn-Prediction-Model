import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from xgboost import XGBClassifier
import joblib
# ...existing code...
#from transform import transform_all_data
# ...existing code...
import numpy as np
import pandas as pd


def transform_all_attendance_data(df1, df2, df3, df4):
    # Use the DataFrames passed as arguments
    df_t = pd.concat([df1, df2, df3], ignore_index=True)
    df_t['ATT_STATUS'] = df_t['ATT_STATUS'].replace({
        'O': 'P',  # Others => Present
    })
    df_t['ATT_DATE'] = pd.to_datetime(df_t['ATT_DATE'], format='%d-%b-%Y')
    start_date = df_t['ATT_DATE'].min()
    end_date = df_t['ATT_DATE'].max()
    acad_year_df = df_t[(df_t['ATT_DATE'] >= start_date) & (df_t['ATT_DATE'] <= end_date)]
    total_attendance = acad_year_df.groupby('SYSTEM_ID').size().reset_index(name='Total_Days_Recorded')
    present_df = acad_year_df[acad_year_df['ATT_STATUS'] == 'P']
    present_days = present_df.groupby('SYSTEM_ID').size().reset_index(name='Present_Days')
    full_att = pd.merge(total_attendance, present_days, on='SYSTEM_ID', how='left').copy()
    full_att['Present_Days'] = full_att['Present_Days'].fillna(0)
    full_att['Attendance_Percentage'] = full_att.apply(
        lambda row: (row['Present_Days'] / row['Total_Days_Recorded']) * 100 if row['Total_Days_Recorded'] > 0 else 0,
        axis=1
    )
    full_att['Attendance_Percentage'] = full_att['Attendance_Percentage'].round(2)
    return full_att

    # Optional: Round to 2 decimal places
    full_att['Attendance_Percentage'] = full_att['Attendance_Percentage'].round(2)
    return full_att
def transform_invoice_data(df):
    """
    Transform raw invoice data into aggregated invoice metrics by SYSTEM_ID.

    Parameters:
        df (pd.DataFrame): Raw invoice DataFrame.

    Returns:
        pd.DataFrame: Aggregated DataFrame with invoice features.
    """

    # Ensure date columns are parsed
    date_cols = ["PAID_DATE", "DUE_DATE"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Total invoices per SYSTEM_ID
    total_invoices = df.groupby("SYSTEM_ID").size().rename("TOTAL_INVOICES")

    # Unpaid invoices (no paid date)
    unpaid_invoices = df[df["PAID_DATE"].isna()].groupby("SYSTEM_ID").size().rename("UNPAID_INVOICES")

    # Late invoices (paid after due date)
    late_invoices = df[(df["PAID_DATE"].notna()) & (df["PAID_DATE"] > df["DUE_DATE"])].groupby("SYSTEM_ID").size().rename("LATE_INVOICES")

    # Merge into one DataFrame
    invoice_summary = pd.concat([total_invoices, unpaid_invoices, late_invoices], axis=1).fillna(0)

    # Percentages
    invoice_summary["PERCENTAGE_UNPAID_INVOICES"] = (invoice_summary["UNPAID_INVOICES"] / invoice_summary["TOTAL_INVOICES"]) * 100
    invoice_summary["PERCENTAGE_LATE_INVOICES"] = (invoice_summary["LATE_INVOICES"] / invoice_summary["TOTAL_INVOICES"]) * 100

    # Ensure integer counts
    invoice_summary[["TOTAL_INVOICES", "UNPAID_INVOICES", "LATE_INVOICES"]] = invoice_summary[["TOTAL_INVOICES", "UNPAID_INVOICES", "LATE_INVOICES"]].astype(int)

    invoice_summary.drop(columns=["TOTAL_INVOICES", "UNPAID_INVOICES", "LATE_INVOICES"], inplace=True)

    # Reset index
    return invoice_summary.reset_index()
def transform_marks_data(marks_df):

    # Remove non-academic subjects
    non_academic_subjects = [
        "Visual Arts", "Music", "Physical Education", "Translation of the Holy Quran (Muslim Students Only)",
        "Computer Practical", "Library", "Drawing"
    ]
    marks_df = marks_df[~marks_df["SUBJECT_NAME"].isin(non_academic_subjects)]

    # Filter only Mid Year and End of Year
    filtered = marks_df[marks_df["TERM_NAME"].isin(["Mid Year", "End of Year"])]

    # Group by SYSTEM_ID and TERM_NAME, summing obtained and total marks
    grouped = filtered.groupby(["SYSTEM_ID", "TERM_NAME"])[["OBTAINED_MARKS", "TOTAL_MARKS"]].sum().reset_index()

    # Calculate percentage
    grouped["PERCENTAGE"] = (grouped["OBTAINED_MARKS"] / grouped["TOTAL_MARKS"]) * 100

    # Pivot to wide format
    pivot_df = grouped.pivot(index="SYSTEM_ID", columns="TERM_NAME", values="PERCENTAGE").reset_index()

    # Rename columns
    pivot_df.rename(columns={
        "Mid Year": "MID_YEAR_PCT",
        "End of Year": "FINAL_YEAR_PCT"
    }, inplace=True)

    return pivot_df
def transform_all_data(inv_df, marks_df, df_a1, df_a2, df_a3, df_a4):
    # Clean SYSTEM_ID in marks and invoice data
    inv_df["SYSTEM_ID"] = pd.to_numeric(inv_df["SYSTEM_ID"].astype(str).str.replace(",", "", regex=False), errors="coerce").astype("Int64")
    marks_df["SYSTEM_ID"] = pd.to_numeric(marks_df["SYSTEM_ID"].astype(str).str.replace(",", "", regex=False), errors="coerce").astype("Int64")

    # Transform each dataset
    invoice_summary = transform_invoice_data(inv_df)
    marks_summary = transform_marks_data(marks_df)
    attendance_summary = transform_all_attendance_data(df_a1, df_a2, df_a3, df_a4)

    # Merge all three on SYSTEM_ID
    combined_df = pd.merge(invoice_summary, marks_summary, on="SYSTEM_ID", how="outer")
    combined_df = pd.merge(combined_df, attendance_summary, on="SYSTEM_ID", how="inner")

    # Fill NaN with 0 where numeric
    combined_df = combined_df.fillna(0)

    return combined_df



df_a1 = pd.read_csv(r"data\att1.csv")
df_a2 = pd.read_csv(r"data\att2.csv")
df_a3 = pd.read_csv(r"data\att3.csv")
df_a4 = pd.read_csv(r"data\att4.csv")
invoices=pd.read_csv(r"data\invoices.csv")
marks=pd.read_csv(r"data\marks.csv")
df = transform_all_data(invoices, marks, df_a1, df_a2, df_a3, df_a4)
df

merged_df=df[['SYSTEM_ID', 'PERCENTAGE_UNPAID_INVOICES', 'PERCENTAGE_LATE_INVOICES',
                'MID_YEAR_PCT', 'FINAL_YEAR_PCT', 'Attendance_Percentage']]

# Check missing values
print("Any missing values?:", merged_df.isnull().values.any())
print("Rows with missing values:\n", merged_df[merged_df.isnull().any(axis=1)])
print("% missing values per column:\n", merged_df.isnull().mean() * 100)

# Fill missing values with median
for col in ['FINAL_YEAR_PCT', 'MID_YEAR_PCT']:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

merged_df['FINAL_YEAR_PCT'] = merged_df['FINAL_YEAR_PCT'].fillna(merged_df['FINAL_YEAR_PCT'].median())
merged_df['MID_YEAR_PCT'] = merged_df['MID_YEAR_PCT'].fillna(merged_df['MID_YEAR_PCT'].median())

# Filter out students with 100% unpaid invoices
merged_df = merged_df[merged_df['PERCENTAGE_UNPAID_INVOICES'] != 100]

features = ['SYSTEM_ID', 'PERCENTAGE_UNPAID_INVOICES', 'PERCENTAGE_LATE_INVOICES',
            'MID_YEAR_PCT', 'FINAL_YEAR_PCT', 'Attendance_Percentage']

# ==== 3. Data cleaning (same as training) ====
# Keep only the relevant columns (if extra columns exist)
merged_df = merged_df[[col for col in features if col in merged_df.columns]]

# Fill missing numeric values with median from test data itself
for col in features:
    if col in merged_df.columns:
        merged_df.loc[:, col] = merged_df[col].fillna(merged_df[col].median())

# ==== 4. Select cleaned feature matrix ====
X_new = merged_df[features]

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

import os
# ==== 1. Load the trained model ====
model_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
model = joblib.load(model_path)

# ==== Ensure 'output' directory exists ====
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# ==== Path for the output file ====
output_file = os.path.join(output_dir, "predictions.csv")

# ==== 5. Make predictions ====
y_new_pred = model.predict(X_new_scaled)  # Predicted class (0 or 1)
y_new_proba = model.predict_proba(X_new_scaled)[:, 1]  # Probability of churn

# ==== 6. Add results to DataFrame ====
merged_df['Predicted_CHURN'] = y_new_pred
merged_df['Churn_Probability'] = y_new_proba

# ==== Save results to 'output' folder ====
merged_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

import os
import xgboost as xgb
import matplotlib.pyplot as plt

# ==== 1. Ensure 'output' directory exists ====
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# ==== 2. Save tree visualization ====
plt.figure(figsize=(20, 10))  # Bigger so nodes are readable
xgb.plot_tree(model, num_trees=0, rankdir='LR')  # Left-to-right tree layout
tree_path = os.path.join(output_dir, "tree_visualization.png")
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Tree visualization saved to {tree_path}")

# ==== 3. Save feature importance plot ====
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=10)
feat_imp_path = os.path.join(output_dir, "feature_importance.png")
plt.savefig(feat_imp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to {feat_imp_path}")


