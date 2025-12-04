import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model

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

def main():
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
    plt.ylabel("Acurácia (CV)")
    plt.title("Curva do Cotovelo - escolha do melhor K")
    plt.grid()
    plt.show()

    best_k = k_values[int(np.argmax(scores))]
    #best_k = 10
    print(f"\nBest K found: {best_k}")

    #Train and evaluate model
    results = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        knn = KNeighborsClassifier(n_neighbors=best_k,weights='distance')
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

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