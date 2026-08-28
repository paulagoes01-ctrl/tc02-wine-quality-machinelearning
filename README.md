# Wine Quality Classification - Tech Challenge 2

## 📌 Sobre o projeto

Este projeto foi desenvolvido como parte do Tech Challenge - Fase 2, com o objetivo de construir uma pipeline de análise e Machine Learning capaz de classificar vinhos de acordo com sua qualidade a partir de características físico-químicas.

A variável original `quality` foi transformada em uma classificação binária:

* **0 — Baixa/Média Qualidade:** `quality < 7`
* **1 — Alta Qualidade:** `quality >= 7`

A partir da análise exploratória e do pré-processamento dos dados, foram desenvolvidos e comparados dois modelos de classificação: **K-Nearest Neighbors (KNN)** e **Regressão Logística**.

---

## Objetivo

O objetivo do projeto é analisar as características físico-químicas dos vinhos, identificar sua relação com a qualidade e desenvolver modelos capazes de classificar os vinhos entre **Baixa/Média Qualidade** e **Alta Qualidade**.

O projeto contempla:

* compreensão e preparação dos dados;
* análise exploratória (EDA);
* análise de correlações e outliers;
* avaliação do balanceamento das classes;
* padronização das variáveis;
* desenvolvimento de modelos de classificação;
* comparação de métricas;
* validação cruzada;
* interpretação dos resultados.

---

## Dataset

O dataset utilizado contém informações sobre diferentes características físico-químicas dos vinhos, além da avaliação de qualidade.

Entre as variáveis disponíveis estão:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol
* Quality

A base utilizada está disponível no **Kaggle — Wine Quality Dataset**.

---

## Análise Exploratória dos Dados

Durante a EDA foram analisados:

* distribuição das variáveis;
* presença de valores ausentes;
* identificação de outliers;
* distribuição da variável `quality`;
* balanceamento da classificação binária;
* correlação entre as características físico-químicas e a qualidade;
* comportamento das principais variáveis entre as classes.

A classe de **Alta Qualidade representa aproximadamente 13,91% da base**, indicando um desbalanceamento entre as classes. Por esse motivo, a avaliação dos modelos não foi baseada apenas em Accuracy, sendo consideradas também métricas como Precision, Recall, F1-score e ROC-AUC.

### Principais correlações

Na análise de correlação, o **teor alcoólico** apresentou a associação positiva mais evidente com a Alta Qualidade (**0,40**), seguido pelo **ácido cítrico (0,25)** e pelos **sulfatos (0,21)**.

A **densidade (-0,15)** apresentou associação negativa, enquanto o **pH (-0,07)** apresentou uma relação mais fraca.

Esses resultados indicam que valores mais elevados de algumas características tendem a estar associados à classificação de Alta Qualidade, embora essas relações representem **associação e não necessariamente causalidade**.

---

## Pré-processamento

Para garantir uma comparação adequada entre os modelos, foi utilizado o mesmo processo de preparação dos dados.

As cinco variáveis utilizadas nos dois modelos foram:

* `alcohol`
* `citric acid`
* `pH`
* `density`
* `sulphates`

Os modelos utilizam exatamente as mesmas features.

Também foram utilizados:

* divisão de **80% para treino e 20% para teste**;
* `random_state = 42`;
* divisão estratificada com `stratify`;
* padronização utilizando `StandardScaler`;
* ajuste do scaler somente nos dados de treino para evitar **data leakage**.

---

## Modelos

### K-Nearest Neighbors — KNN

O primeiro algoritmo utilizado foi o **K-Nearest Neighbors**, configurado com:

```python
KNeighborsClassifier(n_neighbors=6)
```

O KNN classifica uma observação considerando as classes das observações mais próximas no espaço das variáveis.

Como o algoritmo utiliza distâncias, a padronização das variáveis com `StandardScaler` é especialmente importante.

### Regressão Logística

O segundo modelo utilizado foi a **Regressão Logística**, configurada com balanceamento das classes:

```python
LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
```

O `class_weight="balanced"` auxilia o modelo a lidar com o desbalanceamento existente entre as classes.

---

##  Avaliação dos Modelos

Os modelos foram avaliados utilizando:

* **Accuracy:** proporção total de classificações corretas;
* **Precision:** precisão das classificações realizadas como Alta Qualidade;
* **Recall:** capacidade de identificar os vinhos que realmente são de Alta Qualidade;
* **F1-score:** equilíbrio entre Precision e Recall;
* **ROC-AUC:** capacidade de separação entre as classes;
* **Matriz de Confusão:** análise dos acertos e tipos de erro;
* **Cross-Validation:** avaliação da estabilidade dos modelos em diferentes divisões da base.

### Resultados

| Métrica   |        KNN | Regressão Logística |
| --------- | ---------: | ------------------: |
| Accuracy  | **88,21%** |              78,17% |
| Precision | **63,16%** |              35,48% |
| Recall    |     37,50% |          **68,75%** |
| F1-score  | **47,06%** |              46,81% |
| ROC-AUC   | **80,91%** |              80,31% |

O KNN apresentou maior Accuracy e Precision, enquanto a Regressão Logística apresentou Recall significativamente superior. F1-score e ROC-AUC ficaram próximos entre os dois modelos.

Na análise dos erros, o **KNN apresentou 7 falsos positivos**, contra **40 da Regressão Logística**. Em contrapartida, a Regressão apresentou apenas **10 falsos negativos**, contra **20 do KNN**.

---

##  Validação Cruzada

Além da avaliação no conjunto de teste, foi realizada **Cross-Validation** para analisar a estabilidade dos modelos em diferentes divisões dos dados.

Os resultados encontrados foram:

| Modelo              |   F1 médio | Desvio padrão |
| ------------------- | ---------: | ------------: |
| KNN                 |     0,4256 |        0,1041 |
| Regressão Logística | **0,5052** |    **0,0269** |

A Regressão Logística apresentou maior F1 médio e menor variação entre os folds, indicando maior estabilidade entre diferentes amostras da base.

---

## Interpretação dos Resultados

Os resultados mostram comportamentos diferentes entre os modelos.

O **KNN apresentou melhor desempenho geral no conjunto de teste**, com maior Accuracy e Precision, além de resultados ligeiramente superiores em F1-score e ROC-AUC. O modelo também apresentou menor ocorrência de falsos positivos.

A **Regressão Logística**, por outro lado, apresentou Recall superior, demonstrando maior capacidade de identificar os vinhos que realmente pertencem à classe de Alta Qualidade. Além disso, apresentou maior estabilidade na validação cruzada.

Dessa forma, os dois modelos podem ser utilizados dependendo do objetivo da análise:

* **KNN:** indicado quando se busca maior precisão nas classificações e menor ocorrência de falsos positivos.
* **Regressão Logística:** indicada quando o objetivo é identificar uma maior proporção dos vinhos de Alta Qualidade e obter maior estabilidade entre diferentes amostras.

---

## Implicações para o processo produtivo

A análise das variáveis indica que características como **teor alcoólico, ácido cítrico, sulfatos, densidade e pH** apresentam diferentes níveis de associação com a classificação da qualidade.

Essas informações podem auxiliar no monitoramento das características físico-químicas durante o processo produtivo e na identificação de padrões associados aos vinhos classificados como de Alta Qualidade.

Entretanto, as relações encontradas representam **associações estatísticas e não relações de causalidade**. Portanto, não é possível afirmar que alterar isoladamente uma determinada característica resultará necessariamente no aumento da qualidade do vinho.

---

## Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## Estrutura do projeto

```text
wine-quality-classification/
│
├── data/
│   └── WineQT.csv
│
├── notebooks/
│   ├── EDA_Wine_Quality.ipynb
│   ├── KNN.ipynb
│   ├── Regressao_Logistica.ipynb
│   └── Comparacao_Modelos.ipynb
│
├── results/
│   └── gráficos e resultados dos modelos
│
├── requirements.txt
│
└── README.md
```

---

## Conclusão

O projeto permitiu desenvolver uma pipeline completa de análise e classificação da qualidade dos vinhos, passando pela exploração dos dados, preparação das variáveis, desenvolvimento dos modelos e comparação de seus resultados.

O **KNN apresentou resultados superiores na maior parte das métricas no conjunto de teste**, destacando-se principalmente em Accuracy e Precision. Já a **Regressão Logística apresentou maior Recall e maior estabilidade na validação cruzada**.

No geral, ambos os modelos se mostraram adequados para o problema de classificação e podem ser indicados de acordo com o objetivo da análise. O KNN se destaca quando se busca maior precisão e menor ocorrência de falsos positivos, enquanto a Regressão Logística é mais indicada quando se busca identificar uma maior proporção dos vinhos de Alta Qualidade.

---

## Equipe
Ingrid Xavier 
Patricia Ferreira 
Paula Goés
Suellen Moraes

## Referência do Dataset

HASSAN, Yasser. **Wine Quality Dataset**. Kaggle, 2022. Disponível em: Kaggle — Wine Quality Dataset. Acesso em: 28 ago. 2026.
