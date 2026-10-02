"""
Este documento `model_utils.py` contém funções auxiliares 
que ajudam a simplificar e organizar várias tarefas relacionadas com o training, 
avaliação e uso de modelos (ML e DL). 

Achei por bem meter em ficheiros separados, mpara o código ser modular. 

"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

def calculate_metrics(y_true, y_pred):
    """
    Calculate and return key evaluation metrics: accuracy, precision, recall, and F1-score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    tuple: Accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the Confusion Matrix.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot the ROC Curve and calculate the AUC.

    Parameters:
    y_true (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def save_model(model, filename):
    """
    Save the trained model to disk.

    Parameters:
    model: Trained model to be saved.
    filename (str): Name of the file to save the model.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load the saved model from disk.

    Parameters:
    filename (str): Name of the file from which to load the model.

    Returns:
    model: Loaded model.
    """
    return joblib.load(filename)