import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
data = load_iris()
X = data.data
y = data.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3
                                                    , random_state=42)

# Padronizar os dados (muito importante para PCA e SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=2)  # Reduzindo para 2 componentes principais
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Treinar o modelo SVM
svm = SVC(kernel='linear')  # Você pode escolher outros kernels como 'rbf'
svm.fit(X_train_pca, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm.predict(X_test_pca)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

import pickle
# Save the model to a file
with open('svm.model', 'wb') as file:
  pickle.dump(svm, file)

# Load the model from the file
with open('svm.model', 'rb') as file:
  svm = pickle.load(file)
  
  
# treinar o model com cross-validation
from sklearn.model_selection import cross_val_score
svm_cv = SVC(kernel='linear')
scores = cross_val_score(svm_cv, X_train_pca, y_train, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores):.2f}')
# fazer as predicoes com o modelo treinado com cross-validation
y_pred_cv = svm_cv.fit(X_train_pca, y_train).predict(X_test_pca)
accuracy_cv = accuracy_score(y_test, y_pred_cv)
print(f'Acurácia com cross-validation: {accuracy_cv:.2f}')
