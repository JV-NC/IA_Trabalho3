from typing import Optional, Dict, List, Tuple, Literal, Self
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random


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
        f'{color("Component", TITLE):20} | '
        f'{color("Choice", TITLE):35} | '
        f'{color("Price (R$)", TITLE)}'
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
            f'{color(f"{price:>8}", PRICE)}'
        )

    print(color('-' * 71, BORDER))

    # Total
    print(
        f'{color("TOTAL", TITLE + BOLD):20} | '
        f'{"":35} | '
        f'{color(f"{total:>8}", PRICE + BOLD)}'
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

#part 3 and 4
class Item:
    def __init__(self, w: int, h: int, d: int, x: int=0, y: int=0, z: int=0)->None:
        self.w = w
        self.h = h
        self.d = d
        self.x = x
        self.y = y
        self.z = z
    
    def volume(self)->int:
        return self.w * self.h * self.d
    
    def rotated(self, r: Literal[0,1,2,3,4,5] = 0)->Self:
        """
        Returns a new Item instance with r rotation applied.
        r = [0,1,2,3,4,5]
        """

        w, h, d = self.w, self.h, self.d

        if r == 0:
            nw, nh, nd = w, h, d
        elif r == 1:
            nw, nh, nd = w, d, h
        elif r == 2:
            nw, nh, nd = h, w, d
        elif r == 3:
            nw, nh, nd = h, d, w
        elif r == 4:
            nw, nh, nd = d, w, h
        elif r == 5:
            nw, nh, nd = d, h, w
        else:
            raise ValueError("Invalid Rotation")
    
        return Item(nw, nh, nd, 0, 0, 0)

    def intersects(self, other: Self)->bool:
        """check if itself intersects with another Item"""
        return not (self.x + self.w <= other.x or other.x + other.w <= self.x or self.y + self.d <= other.y or other.y + other.d <= self.y or self.z + self.h <= other.z or other.z + other.h <= self.z)
    
    def fits_in_bin(self, bin_w: int, bin_h: int, bin_d: int)->bool:
        """check if the Item fits in bin"""
        return (self.x + self.w <= bin_w and self.y + self.d <= bin_d and self.z + self.h <= bin_h)
    
    def copy(self) ->Self:
        return Item(self.w, self.h, self.d)

class Bin:
    def __init__(self, w: int, h: int, d: int)->None:
        self.w = w
        self.h = h
        self.d = d

        self.items: List[Item] = []
        self.candidate_points: List[Tuple[int, int, int]] = [(0, 0, 0)]
    
    def can_place(self, item: Item)->bool:
        """Verify if a certain Item can be placed in the bin"""
        if not item.fits_in_bin(self.w, self.h, self.d):
            return False
        
        for placed in self.items:
            if item.intersects(placed):
                return False
        
        return True
    
    def place_item(self, item: Item) -> bool:
        """Try place item with candidate points on BLF"""
        for (x, y, z) in self.candidate_points:
            item.x, item.y, item.z = x, y, z

            if self.can_place(item):
                self.items.append(item)

                # new candidate points on BLF (Bottom Left Front)
                self.candidate_points.extend([
                    (x + item.w, y, z),
                    (x, y + item.d, z),
                    (x, y, z + item.h)
                ])

                return True

        return False
    
    def try_place_item_with_rotation(self, item: Item, rotation: int) -> bool:
        """Apply rotation on item and try placing it in the bin """
        rotated_item = item.rotated(rotation)
        return self.place_item(rotated_item)

    def used_volume(self) -> int:
        return sum(item.volume() for item in self.items)

    def fill_ratio(self) -> float:
        return self.used_volume() / (self.w * self.h * self.d)

    def max_height_used(self) -> int:
        if not self.items:
            return 0
        return max(item.z + item.h for item in self.items)
    
def generate_random_items(
        n: int,
        min_size: int,
        max_size: int,
        seed: Optional[int] = None,
)-> List[Item]:
    """
    Generate a List of random Item with size n, and dimensions of Item limited with min_size and max_size.
    """
    if seed is not None:
        random.seed(seed)

    items: List[Item] = []

    for _ in range(n):
        shape = random.choice(['cube', 'bar', 'plate', 'random'])

        match shape:
            case 'cube':
                a = random.randint(min_size, max_size)
                w, h, d = a, a, a
            case 'bar':
                a = random.randint(min_size, max_size)
                b = random.randint(min_size, max_size // 2)
                w, h, d = a, b, b
            case 'plate':
                a = random.randint(min_size, max_size)
                b = random.randint(min_size, max_size // 3)
                w, h, d = a, a, b
            case _:
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                d = random.randint(min_size, max_size)
            
        items.append(Item(w,h,d))
    
    return items

def evaluate_individual(
        individual: List[Tuple[int, int]],
        items: List[Item],
        bin: Bin,
        fitness_type: Literal['fill_ratio','item_rejected','volume_rejected','height_penalized'] = 'fill_ratio',
        *,
        rejection_penalty: float = 0.01,
        volume_penalty: float = 0.001,
        height_penalty: float = 0.01,
)->float:
    """
    Evaluate a GA individual for the 3D Bin Packing Problem using a single bin and different fitness_type.

    -'fill_ratio': Maximizes the bin volume utilization.
    -'item_rejected': Maximizes fill ratio while penalizing rejected items.
    -'volume_rejected': Maximizes fill ratio while penalizing rejected volume.
    -'height_penalized': Maximizes fill ratio while penalizing the used bin height.
    """

    rejected_items = 0
    rejected_volume = 0

    for item_id, rotation in individual:
        item = items[item_id].copy()

        if not bin.try_place_item_with_rotation(item, rotation):
            rejected_items += 1
            rejected_volume += item.volume()

    # main metric
    fill = bin.fill_ratio()
    used_height = bin.max_height_used()
    bin_height = bin.h

    if fitness_type == 'fill_ratio':
        fitness = fill
    elif fitness_type == 'item_rejected':
        fitness = fill - rejection_penalty * rejected_items
    elif fitness_type == 'volume_rejected':
        fitness = fill - volume_penalty * rejected_volume
    else:
        height_ratio = used_height / bin_height if bin_height > 0 else 0
        fitness = fill - height_penalty * height_ratio

    return max(0.0, fitness)

def build_bin_from_individual(
        individual: List[Tuple[int, int]],
        items: List[Item],
        bin_dims: Tuple[int, int, int]
)->Bin:
    """Build a Bin class using a individual tuple"""
    bin = Bin(*bin_dims)

    for item_id, rotation in individual:
        item = items[item_id].copy()
        bin.try_place_item_with_rotation(item, rotation)

    return bin

def create_individual(num_items: int) -> List[Tuple[int, int]]:
    """Create individual with a certain number of items and rotations"""
    ids = list(range(num_items))
    random.shuffle(ids)

    individual = []
    for i in ids:
        rotation = random.randint(0, 5)
        individual.append((i, rotation))

    return individual

def fitness(individual: list[tuple[int, int]], items: List[Item], bin_dims: Tuple[float, float, float], type: Literal['fill_ratio','item_rejected','volume_rejected','height_penalized'] = 'fill_ratio')->float:
    """Instantiate a bin and evaluate individual using it"""
    bin = Bin(*bin_dims)
    return evaluate_individual(
        individual,
        items,
        bin,
        type
    )

def ox_crossover(p1, p2):
    """Crossover parents genes, creating two oposite children"""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    def make_child(p1, p2):
        child = [None] * size
        child[a:b] = p1[a:b]

        used = {gene[0] for gene in child if gene is not None}
        pos = b

        for gene in p2:
            if gene[0] not in used:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1

        return child

    return make_child(p1, p2), make_child(p2, p1)

def mutate_swap(individual: List[Tuple[int, int]])->List[Tuple[int, int]]:
    """Mutate swaping individual's genes"""
    ind = individual[:]
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def mutate_rotation(individual: List[Tuple[int, int]])->List[Tuple[int, int]]:
    """Mutate individual changing the rotation"""
    ind = individual[:]
    i = random.randrange(len(ind))
    item_id, _ = ind[i]
    ind[i] = (item_id, random.randint(0, 5))
    return ind

def mutate(individual: List[Tuple[int, int]])->List[Tuple[int, int]]:
    """Choose between swap or rotation"""
    if random.random() < 0.5:
        return mutate_swap(individual)
    else:
        return mutate_rotation(individual)

def plot_history(
        history_best: list[float],
        history_avg: list[float],
        plot_path: str,
        filename: str,
        title: str,
        xlabel: str = 'Iteration',
        ylabel: str = 'Fitness'
)->None:
    """Plot iteration history"""
    plt.figure()
    plt.plot(history_best, label='Best fitness')
    plt.plot(history_avg, label='Average fitness')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_plot(plot_path, filename)

def plot_sensitivity(
    df: pd.DataFrame,
    param_name: str,
    fixed_params: dict,
    plot_path: str,
    filename: str,
    title_prefix: str = 'Sensitivity'
)->None:
    """Plot sensitivity graph changing one hyperparameter and fixating the others """
    subset = df.copy()

    for k, v in fixed_params.items():
        if k != param_name:
            subset = subset[subset[k] == v]

    subset = subset.sort_values(param_name)

    x = subset[param_name]
    y = subset['best_fit']

    plt.figure()
    plt.plot(x, y, '-o')
    plt.xlabel(param_name)
    plt.ylabel('Best fitness')
    plt.title(f'{title_prefix} – {param_name}')
    plt.grid(True)
    save_plot(plot_path, filename)

def draw_cuboid(ax, x, y, z, w, h, d, color, alpha=0.6)->None:
    """Draw cuboid for items on bin"""
    vertices = [
        [(x, y, z), (x+w, y, z), (x+w, y+d, z), (x, y+d, z)],
        [(x, y, z+h), (x+w, y, z+h), (x+w, y+d, z+h), (x, y+d, z+h)],
        [(x, y, z), (x+w, y, z), (x+w, y, z+h), (x, y, z+h)],
        [(x, y+d, z), (x+w, y+d, z), (x+w, y+d, z+h), (x, y+d, z+h)],
        [(x, y, z), (x, y+d, z), (x, y+d, z+h), (x, y, z+h)],
        [(x+w, y, z), (x+w, y+d, z), (x+w, y+d, z+h), (x+w, y, z+h)],
    ]

    ax.add_collection3d(
        Poly3DCollection(vertices, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha)
    )

def draw_bin_wireframe(ax, w, h, d)->None:
    """Draw bin wireframe"""
    edges = [
        [(0,0,0), (w,0,0)], [(0,d,0), (w,d,0)],
        [(0,0,h), (w,0,h)], [(0,d,h), (w,d,h)],

        [(0,0,0), (0,d,0)], [(w,0,0), (w,d,0)],
        [(0,0,h), (0,d,h)], [(w,0,h), (w,d,h)],

        [(0,0,0), (0,0,h)], [(w,0,0), (w,0,h)],
        [(0,d,0), (0,d,h)], [(w,d,0), (w,d,h)],
    ]

    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, color='black', linewidth=1)

def plot_bin_3d(bin: Bin, plot_path: str, filename: str='bin_3d.png')->None:
    """Use matplotlib and mpl_toolkits for a 3d plot of bin with items"""
    bin_w = bin.w
    bin_h = bin.h
    bin_d = bin.d

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw items
    for item in bin.items:
        color = (
            random.random(),
            random.random(),
            random.random()
        )
        draw_cuboid(
            ax,
            item.x, item.y, item.z,
            item.w, item.h, item.d,
            color=color,
            alpha=0.6
        )

    # Draw bin wireframe
    draw_bin_wireframe(ax, bin_w, bin_h, bin_d)

    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, bin_d)
    ax.set_zlim(0, bin_h)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('3D Bin Packing – Final Solution')

    save_plot(plot_path, filename)

def assert_no_collisions(bin: Bin):
    for i in range(len(bin.items)):
        for j in range(i+1, len(bin.items)):
            if bin.items[i].intersects(bin.items[j]):
                raise RuntimeError("COLISION DETECTED!")

def save_dataframe_csv(
        results: pd.DataFrame,
        metrics_path: str,
        filename: str = 'metrics.csv'
)->None:
    #Ensure that metrics dir exist
    os.makedirs(metrics_path, exist_ok=True)

    csv_file = os.path.join(metrics_path, filename)

    results.to_csv(csv_file, index=False)

#TODO: implement other heuristics
def item_heuristic(item: Item, bin: Bin, type: Literal['volume_rate','item_density']='volume_rate')->float:
    if type == 'volume_rate':
        return item.volume() / (bin.w * bin.h * bin.d)

