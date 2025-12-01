from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def load_dataset(
        csv_path: str,
        target_column: str,
        n_splits: int=5,
        normalize: bool=True,
        apply_pca: bool=False,
        pca_components: int=2,
        ignore_columns: Optional[List[str]]=None
        ) ->List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a CSV dataset and apply preprocessing:
        feature e label split
        normalization (optional)
        PCA (optional)
        KFold
    Returns a list of (X_train, X_test, y_train, y_test) for each fold
    """
    df = pd.read_csv(csv_path)

    #Drop ignored columns (optional)
    if ignore_columns:
        cols_to_drop = [col for col in ignore_columns if col !=target_column]
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    #slit in feature and label
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

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