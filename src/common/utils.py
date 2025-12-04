from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RESET = "\033[0m"
BOLD = "\033[1m"

# Cores
COMPONENT = "\033[96m" #CYAN
TITLE = "\033[95m" #MAGENTA
PRICE = "\033[92m" #GREEN
BORDER = "\033[94m" #BLUE
CHOICE = "\033[97m" #WHITE

def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"

def print_build(build) -> None:
    """
    Show final build table with colors.
    """

    title_line = "=" * 29
    print(
        "\n"
        + color(title_line, BORDER)
        + " "
        + color("FINAL BUILD", TITLE + BOLD)
        + " "
        + color(title_line, BORDER)
    )

    #Header
    header = (
        f"{color('Component', TITLE):20} | "
        f"{color('Choice', TITLE):35} | "
        f"{color('Price (R$)', TITLE)}"
    )
    print(header)

    print(color("-" * 71, BORDER))

    # Table body
    total = 0
    for comp, (name, price) in build.items():
        total += price
        print(
            f"{color(comp, COMPONENT):20} | "
            f"{color(name, CHOICE):35} | "
            f"{color(f'{price:>8}', PRICE)}"
        )

    print(color("-" * 71, BORDER))

    # Total
    print(
        f"{color('TOTAL', TITLE + BOLD):20} | "
        f"{'':35} | "
        f"{color(f'{total:>8}', PRICE + BOLD)}"
    )

    print(color("=" * 71, BORDER) + "\n")

#TODO: apply one-hot encoding or label encoding for 'object' columns
def load_dataset(
        csv_path: str,
        target_column: str,
        n_splits: int=5,
        normalize: bool=True,
        apply_pca: bool=False,
        pca_components: int=2,
        ignore_columns: Optional[List[str]]=None,
        encoding: str='onehot'
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
            X[col] = le.fit_transform[col]

    #handle missing values (imputation)
    imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or a constant value instead
    X = imputer.fit_transform(X)

    #normalize (optional)
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
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
                   average: str = 'macro'
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