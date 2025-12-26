import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('classic')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

## ----------------------------
## 1. PREPROCESAMIENTO DE DATOS
## ----------------------------

# Cargar datasets
print("Cargando datasets...")
bank = pd.read_csv('bank.csv')
forest = pd.read_csv('covtype.csv')
energy = pd.read_csv('energydata_complete.csv')

# -------------------------------------------------
# 1.1 Bank Marketing Dataset (Clasificación discreta)
# -------------------------------------------------
print("\nPreprocesando Bank Marketing...")
# Codificar variables categóricas
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
bank = pd.get_dummies(bank, columns=categorical_cols)

# Variable objetivo
y_bank = bank['y'].map({'yes': 1, 'no': 0})
X_bank = bank.drop('y', axis=1)

# Dividir datos
X_train_bank, X_test_bank, y_train_bank, y_test_bank = train_test_split(
    X_bank, y_bank, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 1.2 Forest Covertype Dataset (Clasificación continua/mixta) - VERSIÓN 2 CLASES
# -------------------------------------------------
print("Preprocesando Forest Covertype...")
# Filtrar para usar solo 2 clases (2 y 4 como en tu ejemplo original)
forest_filtered = forest[forest['Cover_Type'].isin([2, 4])].copy()

# Recodificar a 0 y 1 para mayor claridad
forest_filtered['Cover_Type'] = forest_filtered['Cover_Type'].replace({2: 0, 4: 1})

y_forest = forest_filtered['Cover_Type']
X_forest = forest_filtered.drop('Cover_Type', axis=1)

# Estandarizar
scaler = StandardScaler()
X_forest_scaled = scaler.fit_transform(X_forest)

# Dividir datos
X_train_forest, X_test_forest, y_train_forest, y_test_forest = train_test_split(
    X_forest_scaled, y_forest, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 1.3 Appliances Energy Prediction (Regresión)
# -------------------------------------------------
print("Preprocesando Energy Data...")
y_energy = energy['Appliances']
X_energy = energy.drop('Appliances', axis=1)

# Procesamiento de fecha
X_energy['date'] = pd.to_datetime(X_energy['date'])
X_energy['hour'] = X_energy['date'].dt.hour
X_energy['day_of_week'] = X_energy['date'].dt.dayofweek
X_energy = X_energy.drop('date', axis=1)

# Estandarizar
X_energy_scaled = scaler.fit_transform(X_energy)

# Dividir datos
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(
    X_energy_scaled, y_energy, test_size=0.2, random_state=42
)

## ----------------------------
## 2. EVALUACIÓN DE MODELOS
## ----------------------------

def evaluate_classification(model, model_name, X_train, X_test, y_train, y_test, cv=10):
    """Función para evaluar modelos de clasificación"""
    # Entrenamiento y predicción
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas de test simple
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Validación cruzada (solo con datos de entrenamiento)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Reporte completo
    print(f"\n=== Evaluación de {model_name} ===")
    print(f"Accuracy (test): {accuracy:.4f}")
    print(f"Validación Cruzada (accuracy): {cv_mean:.4f} ± {cv_std:.4f}")
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy_test': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm
    }

def evaluate_regression(model, model_name, X_train, X_test, y_train, y_test, cv=10):
    """Función para evaluar modelos de regresión"""
    # Entrenamiento y predicción
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas de test simple
    mse = mean_squared_error(y_test, y_pred)
    
    # Validación cruzada (solo con datos de entrenamiento)
    cv_scores = -cross_val_score(model, X_train, y_train, cv=cv, 
                                scoring='neg_mean_squared_error')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Reporte completo
    print(f"\n=== Evaluación de {model_name} ===")
    print(f"MSE (test): {mse:.4f}")
    print(f"Validación Cruzada (MSE): {cv_mean:.4f} ± {cv_std:.4f}")
    
    return {
        'model_name': model_name,
        'mse_test': mse,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }

# -------------------------------------------------
# 2.1 Evaluación en Bank Marketing
# -------------------------------------------------
print("\n" + "="*50)
print("EVALUACIÓN EN BANK MARKETING DATASET")
print("="*50)

results_bank = []

# Modelos
dt_bank = DecisionTreeClassifier(max_depth=5)
nb_bank = GaussianNB()
knn_bank = KNeighborsClassifier(n_neighbors=5)

# Evaluación
results_bank.append(evaluate_classification(dt_bank, "Árbol de Decisión", 
                                          X_train_bank, X_test_bank, 
                                          y_train_bank, y_test_bank))

results_bank.append(evaluate_classification(nb_bank, "Naive Bayes", 
                                          X_train_bank, X_test_bank, 
                                          y_train_bank, y_test_bank))

results_bank.append(evaluate_classification(knn_bank, "KNN", 
                                          X_train_bank, X_test_bank, 
                                          y_train_bank, y_test_bank))

# -------------------------------------------------
# 2.2 Evaluación en Forest Covertype (2 clases)
# -------------------------------------------------
print("\n" + "="*50)
print("EVALUACIÓN EN FOREST COVERTYPE DATASET (Clases 2 vs 4)")
print("="*50)

results_forest = []

# Modelos
dt_forest = DecisionTreeClassifier(max_depth=10)
nb_forest = GaussianNB()
knn_forest = KNeighborsClassifier(n_neighbors=5)

# Evaluación
results_forest.append(evaluate_classification(dt_forest, "Árbol de Decisión", 
                                            X_train_forest, X_test_forest, 
                                            y_train_forest, y_test_forest))

results_forest.append(evaluate_classification(nb_forest, "Naive Bayes", 
                                            X_train_forest, X_test_forest, 
                                            y_train_forest, y_test_forest))

results_forest.append(evaluate_classification(knn_forest, "KNN", 
                                            X_train_forest, X_test_forest, 
                                            y_train_forest, y_test_forest))

# -------------------------------------------------
# 2.3 Evaluación en Appliances Energy
# -------------------------------------------------
print("\n" + "="*50)
print("EVALUACIÓN EN APPLIANCES ENERGY DATASET")
print("="*50)

results_energy = []

# Modelos
dt_energy = DecisionTreeRegressor(max_depth=5)
knn_energy = KNeighborsRegressor(n_neighbors=5)

# Evaluación
results_energy.append(evaluate_regression(dt_energy, "Árbol de Regresión", 
                                        X_train_energy, X_test_energy, 
                                        y_train_energy, y_test_energy))

results_energy.append(evaluate_regression(knn_energy, "KNN Regresión", 
                                        X_train_energy, X_test_energy, 
                                        y_train_energy, y_test_energy))

## ----------------------------
## 3. ANÁLISIS COMPARATIVO
## ----------------------------

# Crear DataFrame con todos los resultados
print("\n" + "="*50)
print("RESUMEN COMPARATIVO")
print("="*50)

# Preparar datos para el DataFrame
data = []
for res in results_bank:
    data.append({
        'Dataset': 'Bank Marketing',
        'Algoritmo': res['model_name'],
        'Metrica': 'Accuracy',
        'Test': res['accuracy_test'],
        'CV Mean': res['cv_mean'],
        'CV Std': res['cv_std']
    })

for res in results_forest:
    data.append({
        'Dataset': 'Forest Covertype',
        'Algoritmo': res['model_name'],
        'Metrica': 'Accuracy',
        'Test': res['accuracy_test'],
        'CV Mean': res['cv_mean'],
        'CV Std': res['cv_std']
    })

for res in results_energy:
    data.append({
        'Dataset': 'Appliances Energy',
        'Algoritmo': res['model_name'],
        'Metrica': 'MSE',
        'Test': res['mse_test'],
        'CV Mean': res['cv_mean'],
        'CV Std': res['cv_std']
    })

df_results = pd.DataFrame(data)
print("\nTabla Resumen de Resultados:")
print(df_results.to_string(index=False))

## ----------------------------
## 4. VISUALIZACIONES
## ----------------------------

# Gráfico de Accuracy para clasificación
plt.figure(figsize=(12, 6))
sns.barplot(data=df_results[df_results['Metrica'] == 'Accuracy'],
            x='Algoritmo', y='Test', hue='Dataset')
plt.title("Comparación de Accuracy en Problemas de Clasificación\n(Resultados en conjunto de prueba)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico de MSE para regresión
plt.figure(figsize=(8, 5))
sns.barplot(data=df_results[df_results['Metrica'] == 'MSE'],
            x='Algoritmo', y='Test')
plt.title("Comparación de MSE en Problema de Regresión\n(Resultados en conjunto de prueba)")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig('mse_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

