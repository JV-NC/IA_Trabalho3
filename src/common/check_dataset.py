from typing import Optional
import pandas as pd
import numpy as np

csv_path = 'data/kaggle_dataset/WineQT.csv'

def analisar_dataset(csv_path: str, target: Optional[str]=None, max_classes_target: int=20):
    print('='*80)
    print(f'Carregando dataset: {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'Dataset carregado! Formato: {df.shape[0]} linhas × {df.shape[1]} colunas')
    print('='*80)

    # 1. ANÁLISE GERAL
    print('\nVISÃO GERAL')
    print(df.info())

    print('\nAmostra do dataset:')
    print(df.head())

    print('\nValores ausentes por coluna:')
    print(df.isna().sum())

    # 2. IDENTIFICAÇÃO AUTOMÁTICA DO TARGET
    if target is None:
        print('\nTentando identificar automaticamente uma possível coluna target...')
        # coluna categórica com poucas classes
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() <= max_classes_target]

        if len(cat_cols) == 0:
            print('Não encontrei nenhuma coluna adequada como target.')
            return

        print('Candidatas a target:', cat_cols)
        target = cat_cols[-1]  # escolhe a última apenas como sugestão
        print(f'Usando "{target}" como target sugerido.')
    else:
        print(f'Target definido manualmente: {target}')

    # 3. ANÁLISE DO TARGET
    print('\nANÁLISE DO TARGET')
    print(f'- Tipo da coluna target: {df[target].dtype}')
    print(f'- Número de classes: {df[target].nunique()}')
    print(f'- Classes: {df[target].unique()}')

    print('\nQuantidade de itens por classe:')
    print(df[target].value_counts())

    print('\nPercentual por classe:')
    print(df[target].value_counts(normalize=True) * 100)

    # 4. AVALIAÇÃO DE BALANCEAMENTO
    print('\nAvaliação de balanceamento:')
    class_counts = df[target].value_counts(normalize=True)
    imbalance = class_counts.max() - class_counts.min()
    print(f'- Grau de desbalanceamento: {imbalance:.3f}')

    if imbalance > 0.5:
        print('Dataset MUITO desbalanceado! Técnicas recomendadas: SMOTE, undersampling, etc.')
    elif imbalance > 0.2:
        print('Moderadamente desbalanceado. Pode exigir cuidado no treino.')
    else:
        print('Dataset bem balanceado.')

    # 5. TIPOS DE FEATURES
    print('\nTIPOS DE FEATURES')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    other_cols = [c for c in df.columns if c not in num_cols + cat_cols]

    print(f'- Variáveis numéricas ({len(num_cols)}): {num_cols}')
    print(f'- Variáveis categóricas ({len(cat_cols)}): {cat_cols}')
    print(f'- Outros tipos ({len(other_cols)}): {other_cols}')

    # Cardinalidade de categóricas
    print('\nCardinalidade das variáveis categóricas:')
    for col in cat_cols:
        print(f'{col}: {df[col].nunique()} categorias')

    # 6. CORRELAÇÃO ENTRE FEATURES NUMÉRICAS
    if len(num_cols) > 1:
        print('\nCorrelação entre variáveis numéricas:')
        corr_matrix = df[num_cols].corr()
        print(corr_matrix)
    else:
        print('\nNão há variáveis numéricas suficientes para correlação.')

    # 7. ALERTAS PARA MODELOS
    print('\nALERTAS PARA ALGORITMOS DE CLASSIFICAÇÃO')

    # KNN / SVM precisam de numéricos
    if len(cat_cols) > 0:
        print('Existem variáveis categóricas → será necessário OneHotEncoder/LabelEncoder.')

    # decision tree não precisa de normalização
    if len(num_cols) > 0:
        print('Para SVM/KNN, normalize os dados (StandardScaler ou MinMaxScaler).')

    print('\nAnálise concluída!\n')

def main():
    analisar_dataset(csv_path, target='quality')

if __name__=='__main__':
    main()
