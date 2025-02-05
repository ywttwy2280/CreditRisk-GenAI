import ollama  # Make sure this is at the top
import matplotlib.pyplot as plt
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO
import textwrap 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import ollama
import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter 
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, fbeta_score, roc_auc_score,
    classification_report, RocCurveDisplay)
def AI_report(response_text,topic,withplot=False,plot_data=None):
    # Create figure with subplots
    fig = plt.figure(figsize=(8, 11))  # Letter size
    if withplot:
        # Split layout for text and plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_text = fig.add_subplot(gs[0])
        ax_plot = fig.add_subplot(gs[1])
    else:
        # Single subplot for text only
        gs = fig.add_gridspec(1, 1)
        ax_text = fig.add_subplot(gs[0])

    # Add wrapped text
    ax_text.axis('off')
    wrapped_text = textwrap.fill(response_text, width=100)  # Adjust width as needed
    ax_text.text(-0.1, 0.7, response_text, wrap=True)

    
    if withplot:
        # Use provided data or default values
        x = plot_data.get('x', np.linspace(0, 10, 100)) if plot_data else np.linspace(0, 10, 100)
        y = plot_data.get('y', np.sin(x)) if plot_data else np.sin(x)
        plot_type = plot_data.get('plot_type', 'line') if plot_data else 'line'
        color = plot_data.get('color', 'blue') if plot_data else 'blue'
        
        # Create plot based on type
        if plot_type == 'bar':
            ax_plot.bar(x, y, color=color)
        else:  # Default to line plot
            ax_plot.plot(x, y, color=color, linewidth=2)
        
        # Add labels and title
        ax_plot.set_title(plot_data.get('title', f"{topic} Analysis"))
        ax_plot.set_xlabel(plot_data.get('xlabel', 'X Axis'))
        ax_plot.set_ylabel(plot_data.get('ylabel', 'Y Axis'))
        ax_plot.grid(True)
    
    # Save to PDF
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Write to file
    with open(f"{topic}.pdf", "wb") as f:
        f.write(buffer.getvalue())

# Example usage

def GenAI_Analyst(prompt):
# Generate a response using the DeepSeek model
    full_response = ollama.generate(
    model="deepseek-r1:1.5b",  # Ensure this matches the model name in Ollama
    prompt=prompt)
    if "</think>" in full_response["response"]:
# Extract text after </think>
        final_answer = full_response["response"].split("</think>", 1)[-1].strip()
    else:
        final_answer = full_response["response"]  # Fallback if tags are missing
    
    return final_answer

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Results for {model_name}:")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n")
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Recall": recall,
        "F2": f2,
        "AUC": auc
    }