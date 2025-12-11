import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from joblib import Parallel, delayed, dump
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model

#TODO: training slow, verify optimization
#TODO: SVCLinear with pca_components = 5 is as good as SVC 'rbf' with 3 pca_components

csv_path = 'data/kaggle_dataset/FlightSatisfaction.csv'
target_column = 'satisfaction'
n_splits = 5
normalize = 'std'
pca = True
pca_components = 5
ignore_columns = []
encoder = 'onehot'
imputer_strategy = 'constant'

#Dir for saving models
model_path = 'output/models/svm'

def train_one_fold(i, X_train, X_test, y_train, y_test):
    """Parallel function executed for each fold and measure time."""
    start = time.perf_counter()

    svm = SVC(kernel='rbf')
    #svm = LinearSVC()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    metrics = evaluate_model(
        y_test,
        y_pred,
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        average='macro'
    )

    elapsed = time.perf_counter() - start

    #Saving model
    model_filename = os.path.join(model_path,f'svm_fold_{i+1}.pkl')

    dump(svm,model_filename)

    return i, metrics, elapsed #return index for sort

def main():
    os.makedirs(model_path,exist_ok=True)

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

    print(f'\n\n===== Final Metrics (Mean on {n_splits} folds) =====')
    print(df_results.mean())

    print('\n===== Times =====')
    for i, t in enumerate(fold_times):
        print(f'Fold {i+1}: {t:.3f} s')

    print(f'\nTotal execution time: {elapsed_total:.3f} s\n')

if __name__ == '__main__':
    main()
