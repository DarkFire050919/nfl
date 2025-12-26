# NFL Project Documentation

> Esta documentación describe el flujo de trabajo, las funciones clave, los modelos y los resultados generados por el script **`nfl_project.py`**.

## Índice
1. [Resumen](#resumen)
2. [Estructura del Código](#estructura-del-código)
3. [Preprocesamiento de Datos](#preprocesamiento-de-datos)
   - 3.1 Bank Marketing
   - 3.2 Forest Covertype (clasificación 2 vs 4)
   - 3.3 Appliances Energy (regresión)
4. [Evaluación de Modelos](#evaluación-de-modelos)
   - 4.1 Clasificación
   - 4.2 Regresión
5. [Comparación de Resultados](#comparación-de-resultados)
6. [Visualizaciones](#visualizaciones)
7. [Cómo Ejecutar](#cómo-ejecutar)
8. [Dependencias](#dependencias)
9. [Agradecimientos](#agradecimientos)

---

## 1. Resumen
El script realiza una serie de análisis de modelos para tres conjuntos de datos distintos:
- **Bank Marketing** – clasificación binaria (`yes`/`no`).
- **Forest Cover Type** – clasificación binaria entre “Type 2” y “Type 4”.
- **Appliances Energy Data** – regresión para predecir el consumo de electricidad.

En cada caso se prueban tres algoritmos (Árbol de decisión, Naive Bayes, y KNN), se calcula la métrica de rendimiento y se presenta un comparador.

## 2. Estructura del Código
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
# ...

# 1. PREPROCESAMIENTO DE DATOS
# 2. EVALUACIÓN DE MODELOS
# 3. ANÁLISIS COMPARATIVO
# 4. VISUALIZACIONES
```
El código está organizado en bloques puramente funcionales y utilitarios.

## 3. Preprocesamiento de Datos
### 3.1 Bank Marketing
- Se lee `bank.csv`.
- Se convierte todas las columnas categóricas a one‑hot.
- Se crea `X` y `y` (map de `yes`/`no`).
- División 80/20.

### 3.2 Forest Covertype
- Se filtran solo las clases 2 y 4.
- Se remapean a 0 y 1.
- Se escala con `StandardScaler`.
- División 80/20.

### 3.3 Appliances Energy
- Se lee `energydata_complete.csv`.
- Se extraen `hour` y `day_of_week` de la columna `date`.
- Se escala y se divide 80/20.

## 4. Evaluación de Modelos
Se utilizan dos funciones utilitarias:
- `evaluate_classification` (para clasificación).
- `evaluate_regression` (para regresión).
### 4.1 Clasificación
Para cada dataset se construyen:
| Algoritmo | CV Mean | CV Std | Test Accuracy |
|-----------|---------|--------|---------------|
### 4.2 Regresión
Para el dataset de energía:
| Algoritmo | CV Mean MSE | CV Std | Test MSE |

## 5. Comparación de Resultados
Los resultados se guardan en un `DataFrame` y se imprime:
```	Dataset	Algoritmo	Metrica	Test	CV Mean	CV Std
```
Esto facilita la comparación simultánea de todas las métricas.

## 6. Visualizaciones
- Un **gráfico de barras** comparando el accuracy de los tres modelos en ambos datasets.
- Un gráfico de barras de **MSE** para la regresión.
- Se guardan las gráficas como PNG (`accuracy_comparison.png`).

## 7. Cómo Ejecutar
```bash
python nfl_project.py
```
Asegúrate de que los CSV (`bank.csv`, `covtype.csv`, `energydata_complete.csv`) estén en el mismo directorio.

## 8. Dependencias
- pandas >= 1.5
- numpy >= 1.21
- scikit‑learn >= 1.0
- matplotlib >= 3.4
- seaborn >= 0.11

## 9. Agradecimientos
- A los autores de los datasets.
- A los creadores de Scikit-Learn.

---

> Si quieres ver el flujo de procesamiento visual, revisa el diagrama creado con Mermaid en la carpeta `generated_docs/diagrams`.
