import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed, dump
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model, save_metrics_csv, save_model, save_plot, plot_roc_curve_binary, plot_confusion_matrix

csv_path = 'data/kaggle_dataset/FlightSatisfaction.csv'
target_column = 'satisfaction'
n_splits = 5
normalize = 'std'
pca = True
pca_components = 5
ignore_columns = []
encoder = 'onehot'
imputer_strategy = 'constant'

#Dir for saving outputs
model_path = 'output/models/dt'
plot_path = 'output/plots/dt'
metrics_path = 'output/metrics/dt'

def train_one_fold(i, X_train, X_test, y_train, y_test):
    """Parallel function executed for each fold, measure time and save model."""
    start = time.perf_counter()

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    metrics = evaluate_model(
        y_test,
        y_pred,
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        average='macro'
    )

    elapsed = time.perf_counter() - start

    #plot metrics
    y_score = dt.predict_proba(X_test)[:, 1]
    plot_roc_curve_binary(y_test, y_score, plot_path, f'roc_fold{i+1}.png')
    plot_confusion_matrix(y_test,y_pred,['dissatisfied', 'satisfied'],plot_path,f'cm_fold{i+1}.png')


    #Saving model
    save_model(dt,model_path,f'dt_fold_{i+1}.pkl')

    return i, metrics, elapsed #return index for sort

def main():
    start_total = time.perf_counter()

    folds = load_dataset(csv_path,target_column,n_splits,normalize,pca,pca_components,ignore_columns,encoder,imputer_strategy)

    #Train and evaluate model
    raw_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(train_one_fold)(i, X_train, X_test, y_train, y_test)
        for i, (X_train, X_test, y_train, y_test) in enumerate(folds)
    )

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
