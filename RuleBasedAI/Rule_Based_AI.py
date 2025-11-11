"""
**********************************************

        RULE BASED AI ASSESSMENT

**********************************************
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import numpy as np


#constant variables for file name, test size, and reproducibilit

FILE_NAME = "Adult_Income.csv"
TEST_SIZE = 0.3
RANDOM_STATE = 42




#RULE-BASED AI CLASSES WITH SCORES

class SimpleRuleBasedAdultAI:
    """
    A simple, highly interpretable model based on three core rules
    (Education, Hours, Age) to predict income (>50K).
    """
    def __init__(self, name="Simple Rule-Based AI"):
        self.name = name

    def predict(self, row):
        """Binary prediction: 1 if score >= 0.5, else 0."""
        return 1 if self.predict_score(row) >= 0.5 else 0

    def predict_score(self, row):
        """Calculates a pseudo-confidence score based on simple additive rules."""
        score = 0.0
        
        # Rule 1: High Education (Bachelor's degree or higher equivalent: education-num >= 13)
        if row.get('education-num', 0) >= 13:
            score += 0.5 # High weight assigned to education
            
        # Rule 2: Long Working Hours (More than standard 40 hours)
        if row.get('hours-per-week', 0) > 40:
            score += 0.4 # Medium weight assigned to working hard
            
        # Rule 3: Age/Experience (Older age adds a small boost to income probability)
        if row.get('age', 0) > 45:
            score += 0.2 # Small weight assigned to experience
            
        return min(score, 1.0) # Cap the score at 1.0

    def predict_dataframe(self, X):
        """Applies binary prediction row-wise to the entire test set DataFrame."""
        return X.apply(self.predict, axis=1)

    def predict_dataframe_score(self, X):
        """Applies score prediction row-wise to the entire test set DataFrame (for ROC/AUC)."""
        return X.apply(self.predict_score, axis=1)



class AdvancedRuleBasedAdultAI(SimpleRuleBasedAdultAI):
    """
    An advanced model using more specific, interactive rules, leveraging
    one-hot encoded features (like occupation and marital status).
    """
    def __init__(self, name="Advanced Rule-Based AI"):
        super().__init__(name) # Inherit basic methods and setup

    def predict_score(self, row):
        """Calculates a pseudo-confidence score based on complex, interactive rules."""
        score = 0.0
        
        # Rule 1: Elite Education (Masters, Prof-School, or Doctorate: education-num >= 14) 
        # Combined with marital stability and high hours.
        if row.get('education-num', 0) >= 14:
            score += 0.4
            if row.get('hours-per-week', 0) > 45:
                score += 0.2
            # Interaction: High education combined with married status
            if row.get('marital-status_Married-civ-spouse', 0) == 1:
                score += 0.2

        # Rule 2: Executive/Professional Occupation AND Capital Gains
        is_executive_or_prof = (
            row.get('occupation_Exec-managerial', 0) == 1 or 
            row.get('occupation_Prof-specialty', 0) == 1
        )
        if is_executive_or_prof:
            score += 0.5

            # Strong Interaction: High status job AND significant capital gains/losses
            if row.get('capital-gain', 0) > 5000:
                score += 0.3
            elif row.get('capital-loss', 0) > 1000:
                score += 0.1

        # Rule 3: Older, Experienced, and Still Working Full Time (Age/Hours)
        if row.get('age', 0) >= 50 and row.get('hours-per-week', 0) >= 40:
            score += 0.4
            
        # Rule 4: Self-Employed with Income (high-risk, high-reward) AND Capital Gains
        if row.get('workclass_Self-emp-inc', 0) == 1 and row.get('capital-gain', 0) > 0:
                score += 0.4
                
        return min(score, 1.0)


#DATA LOADING & PREPROCESSING FUNCTION

def load_and_preprocess_data(file_name, test_size, random_state):
    """Loads, cleans, and splits the Adult Income data."""
    try:
        # Load the CSV file
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Ensure it is in the same directory.")
        return None, None, None, None

    # Clean and encode target variable
    df.columns = df.columns.str.strip() # Remove leading/trailing spaces from column names
    df = df.replace("?", pd.NA).dropna() # Replace '?' with NaN and drop missing rows

    # Encode 'salary' column: 1 for >50K, 0 for <=50K
    df['salary'] = df['salary'].apply(lambda x: 1 if '>50K' in x else 0)

    # Split data into features (X) and target (y)
    X, y = df.drop('salary', axis=1), df['salary']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # One-hot encode categorical variables
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    
    # Align columns between train and test sets (crucial after one-hot encoding)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    return X_train, X_test, y_train, y_test



#EVALUATION UTILITY FUNCTIONS

def classification_summary(y_true, y_pred, y_score, model_name):
    """Calculates all key metrics including confusion matrix components and ROC-AUC."""
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Calculate Area Under the Receiver Operating Characteristic Curve
    roc_auc = roc_auc_score(y_true, y_score)
    
    return {
        "Model": model_name,
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "True Negatives (TN)": tn,
        "False Negatives (FN)": fn,
        "ROC-AUC Score": round(roc_auc, 4),
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2),
        "F1 Score (%)": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    }

def visualize_results(summary_df, y_test, y_pred_simple, y_pred_advanced, y_score_simple, y_score_advanced, simple_model_name, advanced_model_name):
    """Generates and displays all required visualizations."""
    
    #Performance Bar Chart
    plt.close('all') #Close previous plot figures
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
    x = np.arange(len(metrics))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    results_df = summary_df.drop(columns=['True Positives (TP)', 'False Positives (FP)', 'True Negatives (TN)', 'False Negatives (FN)'])
    simple_values = results_df.iloc[0][metrics].values
    advanced_values = results_df.iloc[1][metrics].values

    #Plotting the bars
    ax.bar(x - bar_width/2, simple_values, bar_width, label=simple_model_name, color="#70192F")
    ax.bar(x + bar_width/2, advanced_values, bar_width, label=advanced_model_name, color="#751675")
    
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Performance Comparison of Rule-Based AI Models")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show() #Display the bar chart

    #Confusion Matrices
    plt.close('all')
    cm_simple = confusion_matrix(y_test, y_pred_simple)
    cm_advanced = confusion_matrix(y_test, y_pred_advanced)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["<=50K (0)", ">50K (1)"]

    # Simple Model Confusion Matrix
    ConfusionMatrixDisplay(cm_simple, display_labels=labels).plot(ax=axes[0], cmap='Reds', colorbar=False)
    axes[0].set_title(f"Confusion Matrix: {simple_model_name}")
    
    # Advanced Model Confusion Matrix
    ConfusionMatrixDisplay(cm_advanced, display_labels=labels).plot(ax=axes[1], cmap='Purples', colorbar=False)
    axes[1].set_title(f"Confusion Matrix: {advanced_model_name}")

    plt.tight_layout()
    plt.show() # Display the confusion matrices

    # ROC Curve
    plt.close('all')
    roc_auc_simple = summary_df[summary_df['Model'] == simple_model_name]['ROC-AUC Score'].iloc[0]
    roc_auc_advanced = summary_df[summary_df['Model'] == advanced_model_name]['ROC-AUC Score'].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot Simple Model ROC
    RocCurveDisplay.from_predictions(y_test, y_score_simple, name=f"{simple_model_name} (AUC = {roc_auc_simple:.4f})", ax=ax, color="#70192F")
    # Plot Advanced Model ROC
    RocCurveDisplay.from_predictions(y_test, y_score_advanced, name=f"{advanced_model_name} (AUC = {roc_auc_advanced:.4f})", ax=ax, color="#751675")

    # Plot the baseline Random Guess line
    ax.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
    ax.set_title("ROC Curve Comparison - Rule-Based AI Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show() # Display the ROC curve
    
    # Note: On a local machine, plt.show() will open interactive windows.
    # In a virtual/headless environment, this must be replaced with plt.savefig().



# MAIN EXECUTION FUNCTION

def main():
    """Main execution flow for the Rule-Based AI experiment."""
    
    # 2. Data Loading & Preprocessing
    # Load data and get the test set (X_test, y_test)
    _, X_test, _, y_test = load_and_preprocess_data(FILE_NAME, TEST_SIZE, RANDOM_STATE)
    
    if X_test is None:
        return # Exit if data loading failed

    # Model Instantiation and Execution
    simple_model = SimpleRuleBasedAdultAI()
    advanced_model = AdvancedRuleBasedAdultAI()

    # Predictions
    y_pred_simple = simple_model.predict_dataframe(X_test)
    y_pred_advanced = advanced_model.predict_dataframe(X_test)
    y_score_simple = simple_model.predict_dataframe_score(X_test) # Scores for ROC/AUC
    y_score_advanced = advanced_model.predict_dataframe_score(X_test)

    # Compile Summary Table
    summary_data = [
        classification_summary(y_test, y_pred_simple, y_score_simple, simple_model.name),
        classification_summary(y_test, y_pred_advanced, y_score_advanced, advanced_model.name)
    ]
    summary_df = pd.DataFrame(summary_data)

    # Print the summary table to the console
    print("\n*****************************************************************************************************************************************************************************************")
    print("                                                                  Detailed Classification Summary (Results Section) ")
    print("*****************************************************************************************************************************************************************************************")
    print(summary_df.to_string(index=False))
    print("*****************************************************************************************************************************************************************************************\n")

    # Saves summary to CSV file (as requested by the user's provided code structure)
    summary_df.to_csv("classification_summary.csv", index=False)
    print("Classification summary saved to classification_summary.csv")
    print("\n\n\n")

    #Visualization
    visualize_results(
        summary_df, y_test, y_pred_simple, y_pred_advanced, 
        y_score_simple, y_score_advanced, simple_model.name, advanced_model.name
    )

if __name__ == "__main__":
    main() # Execute the main function