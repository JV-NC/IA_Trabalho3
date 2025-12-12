from typing import Optional, Dict, List, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os
import pickle
import matplotlib.pyplot as plt


RESET = '\033[0m'
BOLD = '\033[1m'

# Cores
COMPONENT = '\033[96m' #CYAN
TITLE = '\033[95m' #MAGENTA
PRICE = '\033[92m' #GREEN
BORDER = '\033[94m' #BLUE
CHOICE = '\033[97m' #WHITE

def color(text: str, c: str) -> str:
    return f'{c}{text}{RESET}'

def print_build(build) -> None:
    """
    Show final build table with colors.
    """

    title_line = '=' * 29
    print(
        '\n'
        + color(title_line, BORDER)
        + ' '
        + color('FINAL BUILD', TITLE + BOLD)
        + ' '
        + color(title_line, BORDER)
    )

    #Header
    header = (
        f'{color('Component', TITLE):20} | '
        f'{color('Choice', TITLE):35} | '
        f'{color('Price (R$)', TITLE)}'
    )
    print(header)

    print(color('-' * 71, BORDER))

    # Table body
    total = 0
    for comp, (name, price) in build.items():
        total += price
        print(
            f'{color(comp, COMPONENT):20} | '
            f'{color(name, CHOICE):35} | '
            f'{color(f'{price:>8}', PRICE)}'
        )

    print(color('-' * 71, BORDER))

    # Total
    print(
        f'{color('TOTAL', TITLE + BOLD):20} | '
        f'{'':35} | '
        f'{color(f'{total:>8}', PRICE + BOLD)}'
    )

    print(color('=' * 71, BORDER) + '\n')

def load_dataset(
        csv_path: str,
        target_column: str,
        n_splits: int=5,
        normalize: Literal['std','minmax','abs'] = 'std',
        apply_pca: bool=False,
        pca_components: int=2,
        ignore_columns: Optional[List[str]]=None,
        encoding: Literal['onehot','label'] = 'onehot',
        imputer_strategy: Literal['mean','median','most_frequent','constant'] = 'mean'
        ) ->List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a CSV dataset and apply preprocessing:
        feature e label split
        normalization (optional)
        PCA (optional)
        KFold
        Encoding categorical columns (optional)
    Returns a list of (X_train, X_test, y_train, y_test) for each fold
    """
    df = pd.read_csv(csv_path)

    #Drop ignored columns (optional)
    if ignore_columns:
        cols_to_drop = [col for col in ignore_columns if col !=target_column]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    #slit in feature and label
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    #apply Label encoder if target is categorical
    if df[target_column].dtype =='object' or df[target_column].dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    #apply encoding on categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    if encoding == 'onehot' and categorical_columns:
        enc = OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
        encoded = enc.fit_transform(X[categorical_columns])
        encoded_df = pd.DataFrame(encoded,columns=enc.get_feature_names_out(categorical_columns))
        X = pd.concat([X.drop(columns=categorical_columns),encoded_df], axis=1)
    elif encoding == 'label' and categorical_columns:
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col])

    #handle missing values (imputation)
    imputer_strategy = imputer_strategy if imputer_strategy in ['mean','median','most_frequent','constant'] else 'mean'
    if imputer_strategy == 'constant':
        imputer = SimpleImputer(strategy=imputer_strategy, fill_value=0.0)
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)
    X = imputer.fit_transform(X)

    #normalize (optional)
    if normalize == 'std':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'abs':
        scaler = MaxAbsScaler()
        X= scaler.fit_transform(X)
    
    #apply pca with pca_components (optional)
    if apply_pca:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    
    #KFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for train_idx, test_idx in skf.split(X,y):
        folds.append((X[train_idx], X[test_idx], y[train_idx], y[test_idx]))
    
    return folds

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   metrics: Optional[List[str]] = None,
                   average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'macro'
                   )-> Dict[str, float]:
    """
    Evaluate a model using selected metrics.
    Args:
        y_true: array with true labels;
        y_pred: array with model predictions;
        metrics: list of desired metrics;
        average: type of score analysis ['macro', 'micro', 'weighted', 'binary']
    

    Returns: dict with metrics name as key and value as value. 
    """
    #if no metric is provided, calculate all of them:
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    performance = {}

    if 'accuracy' in metrics:
        performance['accuracy'] = accuracy_score(y_true,y_pred)

    if 'precision' in metrics:
        performance['precision'] = precision_score(y_true,y_pred,average=average,zero_division=0) #needs to change to multiclass

    if 'recall' in metrics:
        performance['recall'] = recall_score(y_true,y_pred,average=average,zero_division=0)

    if 'f1' in metrics:
        performance['f1'] = f1_score(y_true,y_pred,average=average,zero_division=0)
    
    if 'roc_auc' in metrics:
        # For multiclass, uses OVR automatically.
        try:
            performance['roc_auc'] = roc_auc_score(
                y_true, 
                y_pred, 
                multi_class='ovr',
                average=average
            )
        except:
            performance['roc_auc'] = np.nan  # if its not possible to calculate
    
    return performance

def save_metrics_csv(
        results: List[Dict[str,float]],
        fold_times: List[float],
        metrics_path: str,
        filename: str = 'metrics.csv'
)->None:
    """
    Saves the metrics of a fold into a CSV file.
    Args:
        results: List of dicts (each fold's metrics)
        fold_times: List of float (seconds spent per fold)
        metrics_path: Directory where the CSV will be saved
        filename: Name of the CSV file (default = 'metrics.csv')
    """

    #Ensure that metrics dir exist
    os.makedirs(metrics_path, exist_ok=True)

    csv_file = os.path.join(metrics_path, filename)

    df = pd.DataFrame(results)

    df['time_sec'] = fold_times

    #Add mean and std at the end
    mean_row = df.mean(numeric_only=True)
    mean_row.name = 'mean'

    std_row = df.std(numeric_only=True)
    std_row.name = 'std'

    df_final = pd.concat([df, pd.DataFrame([mean_row, std_row])], axis=0)

    df_final.to_csv(csv_file, index=True)

def save_model(model, model_path: str, filename: str)->None:
    """
    Saves a trained model using pickle.
    Args:
        model: sklearn trained model.
        model_path: directory where the model will be saved.
        filename: file name for the model.
    """

    os.makedirs(model_path,exist_ok=True)

    full_path = os.path.join(model_path, filename)

    with open(full_path,'wb') as f:
        pickle.dump(model,f)

def save_plot(plot_path: str, filename: str = "plot.png"):
    """
    Saves the current Matplotlib figure to the specified directory.

    Args:
        plot_path: directory where the plot will be saved.
        filename: name of the output image file.
    """
    os.makedirs(plot_path, exist_ok=True)
    full_path = os.path.join(plot_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve_binary(y_true, y_score, plot_path: str, filename: str = 'roc_curve.png')->None:
    """
    Plots and saves a ROC curve for binary classification.
    Args:
        y_true: Ground truth labels
        y_score: Scores/probabilities for class 1 (predict_proba)
        plot_path: Directory to save plot
        filename: Output filename
    """
    fpr, tpr, _ = roc_curve(y_true,y_score)
    roc_auc = auc(fpr,tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary)")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_plot(plot_path, filename)

def plot_confusion_matrix(y_true, y_pred, classes: list, plot_path: str, filename: str = "confusion_matrix.png", normalize: bool = True):
    """
    Saves a confusion matrix plot to disk.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes (list): List of class names in order
        plot_path (str): Directory to save plot
        filename (str): Filename of saved image
        normalize (bool): If True, normalizes confusion matrix percentage
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    # remove redundant figure opening from sklearn
    plt.tight_layout()

    save_plot(plot_path, filename)