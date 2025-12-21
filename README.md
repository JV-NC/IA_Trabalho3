# Trabalho 3 de Inteligência Artificial

Este Repositório contém todos os 4 trabalhos da etapa final da disciplina de Inteligência Artificial, que aborda Algoritmos de Aprendizagem. 

## Estrutura Geral

O projeto segue a seguinte estrutura de pastas:

```
/IA-Trabalho3-main/
│
├── data/
  ├── part1_JSON/
    └── parts.json
  └── kaggle_dataset/
    ├── train.csv
    ├── test.csv 
    └── FlightSatisfaction.csv
├── src/
  └── common/
    ├── check_dataset.py
    ├── merge_csv.py
    └── utils.py
├── part1_decision_tree/
  ├── DecisionTree.py
  └── tree_diagram.md
├── part2_ml/
  ├── dt.py
  ├── knn.py
  └── svm.py
├── part3_ga/
  └── GeneticAlgorithm.py
├── part4_swarm_immune/
  ├── ACO.py
  ├── CLONALG_immune.py
  └── PSO_swarm.py
```

Sendo que a pasta `data/kaggle_dataset` consiste no dataset [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) que deve ser adicionado e combinado conforme o **Trabalho 3.2**, e a pasta`src/common` contém scripts que podem ser utilizados por diversas partes do trabalho. Em especial `utils.py`, que possui diversas funções de entrada e saída de dados que serão importadas conforme necessário.

## Especificações dos Testes
Os testes foram executados utilizando as seguintes especificações de máquina:

| Componente | Modelo / Especificação |
|-------------|------------------------|
| **Processador** | AMD Ryzen 3 4350G 3.8GHz|
| **Placa-mãe** | ASRock B450 Steel Legend |
| **Memória RAM** | 32 GB Crucial Ballistix 3800 MT/s CL16 |
| **Sistema Operacional** | Windows 11 24H2 |
| **Versão do Python** | 3.12.10 |

# Trabalho 3.1 - Árvore de Decisão Manual
Este trabalho tem como objetivo implementar manualmente uma Árvore de Decisão de forma a representar o processo decisório de um especialista em uma loja de computadores, gerando, ao final das perguntas, uma recomendação de PC, e seu custo.

---

### Estrutura do Projeto

#### Mermaid Chart

A representação inicial do problema foi feita a partir de um [Mermaid Chart](src/part1_decision_tree/tree_diagram.md) localizado em `src/part1_decision_tree/tree_diagram.md`.

Foram construídas **5 funções em Python**:

- **load_parts**  
  A partir da biblioteca `json` e do caminho para o arquivo de peças definido globalmente (por padrão em `data/part_JSON/parts.json`), cria um dicionário que relaciona as peças ao seu preço de mercado encontrado no json.

- **pick_part**  
  A partir do dicionário gerado anteriormente e do nome da categoria gerada pela Árvore de Decisão, atualiza o dicionário da configuração com as informações da peça respectiva.

- **make\_branch**  
  Utilizando `Callable` e a partir de uma pergunta, do resultado da resposta afirmativa de cada ramo e - opcionalmente - o conjunto de categorias resultantes das respostas, para perguntas não binárias, gera uma função que imprime a pergunta no terminal e atualiza a peça armazenada - chamando `pick_part`.

- **main**  
  É a função principal do programa, contendo todas as perguntas a serem feitas e sua estruturação de condicionais que formas a estrutura de árvore. Desta função todas as outras são chamadas, incluindo as geradas pela `make_branch`

- **print\_build**  
  Uma função localizada em `utils.py` que, a partir da configuração final gerada por pick\_part gera uma tabela contendo a categoria do componente, a escolha feita pela Árvore de Decisão, e o seu preço, com a soma total no final.

---

### Utilização

#### 1. Pré-requisitos
- Certifique-se de ter o **Python 3.12** ou superior instalado em seu sistema.  
- O **PATH** deve estar configurado corretamente.  
- Certifique-se que o arquivo `parts.json`, contendo as informações dos componentes, esteja no diretório: `/data/part1_JSON/`

---

#### 2. Execução
Abra um terminal e navegue até o diretório raiz do projeto.  
Execute o comando:

```bash
python /src/part1_decision_tree/DecisionTree.py
```
#### 3. Saída

Em cada etapa da Árvore de Decisão o usuário será requisitado a responder a pergunta atual impressa na interface de comando, sendo também informado do conjunto de respostas válidas. Ao final de todas as perguntas, a tabela final de componentes será impressa seguindo a seguinte estrutura de exemplo:

```
============================= FINAL BUILD =============================
Component            | Choice                              | Price (R$)
-----------------------------------------------------------------------
Keyboard             | Kaihl Brown Keyboard                |   350.00
Controller           | No controller                       |     0.00
Camera               | No camera                           |     0.00
CPU                  | Ryzen 7600                          |  1200.00
Motherboard          | B650 (AM5)                          |  1100.00
RAM                  | 32GB DDR5                           |   850.00
GPU                  | RTX 4070                            |  4200.00
PSU                  | G800W                               |   550.00
Audio                | None                                |     0.00
Monitor              | 24" 1440p 144Hz                     |  1800.00
Mouse                | Deathadder V2 Mini                  |   250.00
Wifi                 | Bluetooth Dongle                    |    50.00
-----------------------------------------------------------------------
TOTAL                |                                     | 10350.00
=======================================================================
```

---

# Trabalho 3.2 - Algoritmos de Aprendizado Supervisionado

Este trabalho tem como objetivo implementar algoritmos de **Aprendizado Supervisionado** de forma a classificar o dataset [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) do Kaggle. Mais especificamente, foram implementados os algoritmos **KNN**, **SVM**, e **Árvore de Decisão**.

---

### Implementação

Muitos dos métodos utilizados para implementação vêm da biblioteca `scikit-learn` que é utilizada em todos os script para funções diversas relacionadas a manipulação de bases de dados.

As métricas utilizadas para aferir a efetividade dos algoritmos são:

**`accuracy`:** Taxa de acerto média das classificações 

**`precision`:** Proporção de predições positivas corretas com relação ao conjunto total de predições positivas.

**`recall_score`:** Proporção de casos positivos corretamente preditos com relação ao conjunto de casos positivos reais.

**`f1_score`:** Baseado nos dois anteriores gera um valor único que representa o equilíbrio entre `precision` e `recall`.

**`roc_auc`:** desbalanceamento da área positiva e negativa do gráfico de classificação independente da frequência relativa de cada classe.

#### Funções compartilhadas
No arquivo `utils.py` estão localizadas diversas funções que são utilizadas por todos os algoritmos implementados de forma a evitar redundância. Estas funções são:

- **load_dataset**  
Script de preparação do dataset, rotulando colunas ou eliminando as irrelevantes para o aprendizado, preenchendo dados inválidos, normalizando ou convertendo o conjunto de dados para valores numéricos ou colunas binárias, reduzindo a dimensionalidade do conjunto de dados, e por fim, dividindo o conjunto de dados para **Validação Cruzada** (K-Fold) e Processamento Paralelo para treinar múltiplos modelos simultaneamente.

- **evaluate_model**  
Gera, usando `sklearn.metrics` as métricas de efetividade para um dado modelo e o retorna na forma de um dicionário.

- **save_metrics_csv**
Recebe um dicionário contendo as métricas de um modelo e o armazena em um arquivo `.csv`.

- **save_plot**
Salva uma imagem gerada pelo `matplotlib` como um arquivo `.png`no caminho especificado.

- **plot_roc_curve_binary**
Usando `matplotlib` e com base em uma classificação binária e sua predição, gera um gráfico da curva ROC e chama a função `save_plot` para salvá-la como uma imagem.

- **plot_confusion_matrix**
Usando `matplotlib` e com base em uma classificação binária e sua predição, gera um gráfico mostrando o número de acertos positivos e negativos e o número de perdições erradas positivas e negativas. Depois, chama a função `save_plot` para salvá-la como uma imagem.

#### Outras especificações
**`Folds`:** São criados usando Stratified KFold, garantindo a mesma divisão de amostras para cada classe em todos os Folds. São executados em paralelo através do backend `loky` acelerando substancialmente o tempo de treinamento.


**`KNN`:** O valor ideal de K é gerado a partir do primeiro Fold usando o Método do Cotovelo, selecionando o melhor valor antes da execução completa. Os valores são normalizados utilizando `MinMaxScaler`

**`SVM`:** O código por padrão utiliza `RBF` para separar as classes em espaços de maior dimensionalidade, aumentando substancialmente o tempo de execução. Modificando-se a linha para especificar `LinearSVC` cria um grande ganho de performance, mas gera resultados insatisfatórios pois a base de dados não é linearmente separável.
Dessa forma, para manter uma performance ainda aceitável, deve-se limitar o número de componentes PCA para 5.
Os dados são Z-Normalizados de forma a contrapor os efeitos de variância de características em dimensões mais altas.

---

### Utilização

#### 1. Pré-requisitos
- Certifique-se de ter o **Python 3.12** ou superior instalado em seu sistema.  
- O **PATH** deve estar configurado corretamente.  

#### Preparação do Dataset
- Mova a pasta do [dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) `/kaggle_dataset` para a pasta `/data/`.
- Em seguida execute o script `src/common/merge_csv` para unir a bases de dados de treinamento e teste usando o seguinte comando:

```bash
python src/common/merge_csv.py
```

#### Dependências
Para instalar as dependências, abra um terminal e execute o seguinte comando
```bash
pip install pandas numpy scikit-learn joblib matplotlib
```
---

#### 2. Execução
Abra um terminal e navegue até o diretório raiz do projeto.  
Execute o comando:

```bash
python /src/part2_ml/ALGORITMO.py
```

Substituindo ALGORITMO pela sigla do método que deseja executar: `dt` para Árvore de Decisão, `knn` para K-Vizinhos Próximos e `svm` para Máquina de Vetores de Suporte.

#### 3. Saída

Para cada algoritmo de aprendizado executado serão gerados os seguintes arquivos na pasta `/output/`:

- `metrics/ALGORITMO/metrics.csv`: Arquivo `csv` contendo as metricas usadas para aferir o desempenho dos algoritmos além da média destes valores entre cada `fold`.
- `models/ALGORITMO/ALGORITMO_fold_X`: Arquivo binário `pkl` gerado usando a biblioteca `pickle` para armazenar o modelo de predição de cada `fold`.
- `plots/ALGORITMO/cm_fold_X`: Imagem `png` gerada usando a biblioteca `matplotlib` para representar a matriz de confusão do modelo de predição de cada `fold`.
- `plots/ALGORITMO/roc_fold_X`: Imagem `png` gerada usando a biblioteca `matplotlib` para representar a matriz de confusão do modelo de predição de cada `fold`.

# Autores

<table style="margin: 0 auto; text-align: center;">
  <tr>
    <td colspan="5"><strong>Alunos</strong></td>
  </tr>
  <tr>
      <td>
      <img src="https://avatars.githubusercontent.com/u/83346676?v=4" alt="Avatar de Arthur Santana" style="border-radius:50%; border:4px solid #4ECDC4; box-shadow:0 0 10px #4ECDC4; width:100px;"><br>
      <strong>Arthur Santana</strong><br>
      <a href="https://github.com/Rutrama">
        <img src="https://img.shields.io/github/followers/Rutrama?label=Seguidores&style=social&logo=github" alt="GitHub - Arthur Santana">
      </a>
    </td>
        <td>
      <img src="https://avatars.githubusercontent.com/u/114318721?v=4" alt="Avatar de João Vitor" style="border-radius:50%; border:4px solid #4ECDC4; box-shadow:0 0 10px #4ECDC4; width:100px;"><br>
      <strong>João Vitor</strong><br>
      <a href="https://github.com/JV-NC">
        <img src="https://img.shields.io/github/followers/JV-NC?label=Seguidores&style=social&logo=github" alt="GitHub - João Vitor">
      </a>
    </td>
