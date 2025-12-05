import pandas as pd
import numpy as np
from sklearn.svm import SVC
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model

#TODO: maybe save model?
#TODO: training slow, verify optimization
#TODO: print after load dataset

csv_path = 'data/kaggle_dataset/FlightSatisfaction.csv'
target_column = 'satisfaction'
n_splits = 5
normalize = 'std'
pca = True
pca_components = 3
ignore_columns = []
encoder = 'onehot'
imputer_strategy = 'constant'

def main():
    folds = load_dataset(csv_path,target_column,n_splits,normalize,pca,pca_components,ignore_columns,encoder,imputer_strategy)

    #Train and evaluate model
    results = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)

        metrics = evaluate_model(
            y_test,
            y_pred,
            metrics=['accuracy', 'precision', 'recall', 'f1','roc_auc'],
            average='macro'
        )

        print(f"\n===== FOLD {i+1} =====")
        for m, v in metrics.items():
            print(f"{m}: {v:.4f}")

        results.append(metrics)

        df_results = pd.DataFrame(results)

    print(f'\n\n===== Final Metrics (Mean on {n_splits} folds) =====')
    print(df_results.mean())

if __name__ == '__main__':
    main()
