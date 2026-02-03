# Clasificación de variedades de frijoles a partir de características morfológicas

## Índice

- [Descripción general](#descripción-general)
- [Objetivos](#objetivos)
  - [Objetivo general](#objetivo-general)
  - [Objetivos específicos](#objetivos-específicos)
- [Dataset](#dataset)
  - [Variables principales](#variables-principales)
- [Metodología y modelos implementados](#metodología-y-modelos-implementados)
  - [Análisis Exploratorio de Datos (EDA)](#1-análisis-exploratorio-de-datos-eda)
  - [Preprocesamiento](#2-preprocesamiento)
  - [Agrupamiento Difuso (Fuzzy C-Means)](#3-agrupamiento-difuso-fuzzy-c-means)
  - [Clasificación Supervisada](#4-clasificación-supervisada)
  - [Importancia de características](#5-importancia-de-características)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
  - [Instalación de dependencias](#instalación-de-dependencias)
- [Instrucciones de uso](#instrucciones-de-uso)
- [Autor](#autor)


## Descripción general

Este proyecto aborda el problema de la **clasificación automática de variedades de frijoles secos** utilizando características morfológicas extraídas de imágenes y técnicas de **aprendizaje automático**.  
El objetivo es automatizar una tarea tradicionalmente realizada por inspección visual humana, la cual es costosa, subjetiva y propensa a errores, especialmente en contextos de **control de calidad agrícola** y **agricultura de precisión**.

El análisis se realiza sobre el **Dry Bean Dataset** del *UCI Machine Learning Repository*, que contiene **13,611 muestras** correspondientes a **siete variedades de frijoles**, cada una caracterizada por **16 atributos morfológicos**.

El problema se aborda desde dos enfoques complementarios:

- **Agrupamiento no supervisado**, mediante **Fuzzy C-Means (FCM)**, para analizar la estructura natural de los datos.
- **Clasificación supervisada**, utilizando **SVM, Random Forest y KNN**, para identificar la variedad de frijol con alta precisión.

---

## Objetivos

### Objetivo general

Desarrollar y evaluar un modelo de clasificación de variedades de frijoles secos basado en características morfológicas, combinando técnicas de **agrupamiento difuso** y **aprendizaje supervisado**, con el fin de analizar la separabilidad entre variedades y determinar el clasificador con mejor desempeño.

### Objetivos específicos

- Aplicar **Fuzzy C-Means** para identificar agrupamientos naturales en el conjunto de datos.
- Evaluar la calidad del agrupamiento mediante índices de validación (FPC y Silhouette).
- Entrenar y comparar clasificadores supervisados (**SVM, Random Forest y KNN**).
- Analizar el impacto del desbalance de clases en el desempeño de los modelos.
- Identificar las **características morfológicas más relevantes** para la clasificación.

---

## Dataset

El conjunto de datos utilizado es el **Dry Bean Dataset** del *UCI Machine Learning Repository*.

- **Observaciones:** 13,611  
- **Variedades:** 7  
- **Características:** 16 variables morfológicas + clase  
- **Clases:** Dermason, Sira, Seker, Horoz, Cali, Barbunya, Bombay  

### Variables principales

- **Medidas de tamaño y forma:**
  - Area
  - Perimeter
  - MajorAxisLength / MinorAxisLength
  - AspectRatio
  - Eccentricity
  - ConvexArea
  - EquivDiameter
- **Medidas de compacidad y redondez:**
  - Extent
  - Solidity
  - Roundness
  - Compactness
- **Factores geométricos:**
  - ShapeFactor1 – ShapeFactor4

**Fuente:**  
Dry Bean Dataset – UCI Machine Learning Repository  
https://doi.org/10.24432/C50S4B

---

## Metodología y modelos implementados

### 1. Análisis Exploratorio de Datos (EDA)

- Estadísticas descriptivas de las 16 características morfológicas.
- Análisis de correlación para detectar redundancia entre variables.
- Identificación de desbalance de clases.
- Visualización de distribuciones por variedad.

---

### 2. Preprocesamiento

- Normalización de variables mediante **StandardScaler** (media = 0, desviación = 1).
- Reducción dimensional con **PCA** para visualización y análisis exploratorio.
- División estratificada del dataset:
  - 70% entrenamiento
  - 30% prueba

---

### 3. Agrupamiento Difuso (Fuzzy C-Means)

- Implementación de **FCM** con exponente difuso *m = 2*.
- Evaluación de valores de *k* desde 2 hasta 11 clusters.
- Métricas de validación:
  - **Coeficiente de Partición Difusa (FPC)**
  - **Índice de Silhouette**
- Selección de **k = 7** por correspondencia con las variedades reales.

---

### 4. Clasificación Supervisada

Se entrenaron y evaluaron los siguientes modelos:

- **Support Vector Machine (SVM)** con kernel RBF  
- **Random Forest** (100 árboles)  
- **K-Nearest Neighbors (KNN)** con *k = 5*

#### Métricas de evaluación

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión

**Mejor desempeño:**  
- **SVM:** 92.58% de exactitud  
- Random Forest: 92.04%  
- KNN: 91.75%

---

### 5. Importancia de características

El análisis mediante **Random Forest** identificó como variables más relevantes:

1. ShapeFactor3  
2. Perimeter  
3. Compactness  
4. ShapeFactor1  
5. ConvexArea  

Estas características, relacionadas con la **forma y compacidad del grano**, resultaron clave para la correcta discriminación entre variedades.

---

## Estructura del repositorio

```text
├── data/
│   └── Dry_Bean_Dataset.csv
├── notebooks/
│   └── analisis_y_modelado.ipynb
├── figures/
│   └── resultados_y_graficas/
├── README.md
```
###Requisitos

Para ejecutar el proyecto se requiere:

- Python 3.8 o superior

Librerías:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scikit-fuzzy

###Instalación de dependencias
```text
pip install numpy pandas matplotlib seaborn scikit-learn scikit-fuzzy
```
###Instrucciones de uso

1. Clonar el repositorio:

git clone https://github.com/cast06/Clasificacion-de-Frijoles.git

2. Descargar el dataset desde UCI y colocarlo en la carpeta data/.

3. Ejecutar el notebook principal para reproducir el análisis y los resultados.

###Autor

Sharon Xolocotzi Castillo
Ciencia de Datos
Escuela Superior de Cómputo (ESCOM – IPN)
