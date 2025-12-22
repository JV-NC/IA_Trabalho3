# Trabalho 3 de Inteligência Artificial

Este Repositório contém todos os 4 trabalhos da etapa final da disciplina de Inteligência Artificial, que aborda Algoritmos de Aprendizagem. 

## Utilização Global

Apesar de cada Parte do Trabalho poder ser executada separadamente. O repositório como um todo é feito para ser executado automaticamente através de um script, conforme as instruções a seguir:

### 1. Pré-requisitos
- Certifique-se de estar no Linux
- Ter o **Python 3.12** ou superior instalado em seu sistema. 
- Ter o venv (Virtual Environment) disponível.
- O **PATH** deve estar configurado corretamente.

---

### Peparação do dataset
- Mova os dois arquivos na pasta do [dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) para a pasta `/data/kaggle_dataset`.

### 2. Execução
Abra um terminal e navegue até o diretório raiz do projeto.  
Execute o comando:

```bash
./run.sh
```

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

### Saída

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

### 3. Saída

Para cada algoritmo de aprendizado executado serão gerados os seguintes arquivos na pasta `/output/`:

- `metrics/ALGORITMO/metrics.csv`: Arquivo `csv` contendo as metricas usadas para aferir o desempenho dos algoritmos além da média destes valores entre cada `fold`.
- `models/ALGORITMO/ALGORITMO_fold_X`: Arquivo binário `pkl` gerado usando a biblioteca `pickle` para armazenar o modelo de predição de cada `fold`.
- `plots/ALGORITMO/cm_fold_X`: Imagem `png` gerada usando a biblioteca `matplotlib` para representar a matriz de confusão do modelo de predição de cada `fold`.
- `plots/ALGORITMO/roc_fold_X`: Imagem `png` gerada usando a biblioteca `matplotlib` para representar a matriz de confusão do modelo de predição de cada `fold`.

# Trabalho 3.3 - Bin Packing com Algoritmo Genético

Este trabalho tem como objetivo implementar o algoritmo genético de aprendizado de máquina de forma a eficientemente posicionar objetos 3D no mínimo de contâiners possível, consequentemente ocupando o máximo de volume possível por contâiner.

---

### Implementação

O projeto utiliza uma abordagem evolutiva para organizar itens em um espaço tridimensional. Buscou-se a melhor sequência de inserção dos itens para minimizar o espaço vazio dentro de um volume delimitado.

As métricas e componentes principais utilizados são:

**`fill_ratio`:** Porcentagem do volume total do contêiner que foi efetivamente preenchida por itens.

**`fitness`:** Valor de aptidão calculado com base na eficiência do empacotamento, utilizado pelo GA para selecionar os melhores indivíduos.

**`time_sec`:** Tempo de execução necessário para o algoritmo completar o número total de iterações em cada configuração.

#### Componentes do Algoritmo Genético
A classe `GA` no script principal gerencia o ciclo de vida evolutivo:



#### Mecanismos do Algoritmo
A implementação segue o ciclo de vida clássico de um Algoritmo Genético, adaptado para problemas de permutação:

- **Representação (Cromossomo):** Cada indivíduo é representado por uma lista de índices (ex: `[5, 2, 19, 0, ...]`). Esta lista define a **ordem de prioridade** em que os itens serão processados pelo algoritmo de posicionamento 3D.
  
- **Seleção (`select`):** Implementada via **Seleção por Torneio**. O código sorteia k indivíduos (padrão $k=2$) e seleciona aquele com o maior valor de `fitness`. Isso garante pressão seletiva enquanto mantém a diversidade da população.

- **Crossover (`ox_crossover`):** Como o problema exige que cada item seja único no empacotamento, utiliza-se o **Crossover de Ordem (OX)**. Este operador copia um segmento de um pai e preenche o restante com os genes do outro pai na ordem em que aparecem, evitando duplicatas.


- **Mutação (`mutate`):** Utiliza a **Mutação por Troca (Swap)**. Com uma probabilidade definida por `mut_rate`, dois índices do cromossomo são trocados de posição. Isso altera a sequência de empacotamento, permitindo que o algoritmo explore novas organizações espaciais.

- **Loop Evolutivo (`step` & `run`):** - A função `step` cria uma nova geração completa através de seleção, cruzamento e mutação.
    - A função `run` gerencia as iterações e mantém um registro histórico (`history_best` e `history_avg`) para análise de convergência, além de garantir o elitismo ao rastrear o melhor indivíduo global.

#### Paralelismo e Grid Search
Para encontrar a configuração ideal de hiperparâmetros, o script utiliza:

- **Busca em Grade (Grid Search):** O `product` do `itertools` gera todas as combinações possíveis de `POP_SIZES`, `CX_RATES`, `MUT_RATES` e `MAX_ITERS`.
- **Joblib (`Parallel`):** O backend `loky` é utilizado para distribuir cada configuração da grade para um núcleo diferente do processador (`n_jobs=-1`), acelerando o processamento.

#### Funções importadas
No arquivo `utils.py` estão localizadas as funções auxiliares para manipulação geométrica e visualização:

- **generate_random_items:** Gera um conjunto de itens com dimensões (W, H, D) aleatórias dentro de intervalos pré-definidos.
- **build_bin_from_individual:** Construtor que recebe uma sequência de itens e tenta posicioná-los no espaço 3D seguindo uma lógica de empacotamento.
- **assert_no_collisions:** Função de validação que garante que nenhum item sobrepõe o outro no espaço tridimensional.
- **plot_bin_3d:** Gera uma visualização 3D interativa ou estática do contêiner finalizado.
- **plot_sensitivity:** Gera gráficos comparativos para analisar o impacto de cada hiperparâmetro no desempenho final.

---

### 3. Saída
Após a execução do algoritmo uma saída similar ao exemplo a seguir deve ser gerada, com as métricas de desempenho variando conforme o hardware utilizado.

```
Total GA grid search time: 424.07s
total item volume = 51129
bin volume = 31250
item/bin rate = 1.6361

===== BEST CONFIGURATION =====
pop_size: 50
cx_rate: 0.6
mut_rate: 0.4
max_iters: 150
fitness = 0.8296
fill ratio = 82.96%
time (sec) = 17.25
```

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
