import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model, save_metrics_csv, save_model

#TODO: test all scalers and imputers, test weights of KNN
#TODO: test diferent values of n_splits and pca_components
#TODO: implement output files or matplotlib models evaluation

csv_path = 'data/kaggle_dataset/FlightSatisfaction.csv'
target_column = 'satisfaction'
n_splits = 5
normalize = 'minmax'
pca = True
pca_components = 5
ignore_columns = []
encoder = 'onehot'
imputer_strategy = 'constant'

#Dir for saving models
model_path = 'output/models/knn'
plot_path = 'output/plots/knn'
metrics_path = 'output/metrics/knn'

def train_one_fold(i, X_train, X_test, y_train, y_test, best_k):
    """Parallel function executed for each fold, measure time and save model."""
    start = time.perf_counter()

    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    metrics = evaluate_model(
        y_test,
        y_pred,
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        average='macro'
    )

    elapsed = time.perf_counter() - start

    # --- Save model ---
    save_model(knn,model_path,f'knn_fold_{i+1}.pkl')

    return i, metrics, elapsed

def main():
    #Ensure output dirs
    os.makedirs(model_path,exist_ok=True)
    os.makedirs(plot_path,exist_ok=True)
    os.makedirs(metrics_path,exist_ok=True)

    start_total = time.perf_counter()

    folds = load_dataset(csv_path,target_column,n_splits,normalize,pca,pca_components,ignore_columns,encoder,imputer_strategy)

    #Pick best K using only first fold
    X_train, X_test, y_train, y_test = folds[0]

    k_values = list(range(1,31,2))
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for k in k_values:
        pipe = make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=k))
        score = cross_val_score(pipe, X_train, y_train, cv=cv).mean()
        scores.append(score)

    #elbow curve
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, scores)
    plt.xlabel("K")
    plt.ylabel("Accuracy (CV)")
    plt.title("Elbow method - pick best K")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/elbow_method.png", dpi=300)
    plt.close()

    best_k = k_values[int(np.argmax(scores))]
    #best_k = 10
    print(f"\nBest K found: {best_k}")

    #Train and evaluate model
    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(train_one_fold)(i, X_train, X_test, y_train, y_test, best_k)
        for i, (X_train, X_test, y_train, y_test) in enumerate(folds)
    )

    #ensure that outputs is sorted by fold id
    raw_results.sort(key=lambda x: x[0])

    results = []
    fold_times = []

    for i, metrics, elapsed in raw_results:
        print(f'\n===== FOLD {i+1} =====')
        for m, v in metrics.items():
            print(f'{m}: {v:.4f}')
        #print(f'Fold {i+1} time: {elapsed:.3f} s')

        results.append(metrics)
        fold_times.append(elapsed)
    
    df_results = pd.DataFrame(results)

    elapsed_total = time.perf_counter() - start_total

    save_metrics_csv(results,fold_times,metrics_path)

    print(f'\n\n===== Final Metrics (Mean on {n_splits} folds) =====')
    print(df_results.mean())

    print('\n===== Times =====')
    for i, t in enumerate(fold_times):
        print(f'Fold {i+1}: {t:.3f} s')

    print(f'\nTotal execution time: {elapsed_total:.3f} s\n')
    print(f'Models saved on: {model_path}')

if __name__ == '__main__':
    main()