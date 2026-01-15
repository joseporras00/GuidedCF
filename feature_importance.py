import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from utils import create_sequences3  # Assuming perturbate_series function is in utils.py

# Añadir las librerías necesarias para SHAP, LIME y Permutation Feature Importance
import shap
import lime
from lime import lime_tabular
from sklearn.inspection import permutation_importance

# Semillas para las diferentes particiones
SEEDS = [42, 123, 456, 789, 101112]

# Definir los métodos XAI a evaluar.
methods = ['IG', 'SG', 'GRAD']
#methods = ['random']

# Evaluar perturbaciones
perturbation_methods = ['zero', 'inverse', 'noise', 'swap', 'mean']

# Cargar datos.
X, y = create_sequences3('combined_preprocessed_data')

# Cargar los nombres de las variables
feature_names = pd.read_csv('data/combined_preprocessed_data.csv').columns[:-3]

sz = 90
special_value = -10.0
X = pad_sequences(X, maxlen=sz, dtype='float', padding='post', truncating='post', value=special_value)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Función para aplicar SHAP
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Función para calcular la métrica de rendimiento
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred)
    #return accuracy, f1, auc
    return auc

# Función para evaluar la importancia de las características mediante permutaciones
def permutation_importance(model, X_test, y_test, n_repeats=10):
    # Métricas originales sin permutar
    #y_pred = np.argmax(model.predict(X_test), axis=1)
    original_probs = model.predict(X_test)
    y_pred = (original_probs > 0.5).astype(int)
    original_metrics = calculate_metrics(y_test, y_pred)

    feature_importance = np.zeros(X_test.shape[2])  # Array para almacenar la importancia de cada característica

    for feature_idx in range(X_test.shape[2]):
        permuted_metrics = np.zeros((n_repeats, len(original_metrics)))  # Guardar las métricas permutadas

        for i in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])  # Permutar la característica en todos los timesteps

            original_probs = model.predict(X_test)
            y_pred_permuted = (original_probs > 0.5).astype(int)
            permuted_metrics[i] = calculate_metrics(y_test, y_pred_permuted)

        # Calcular la disminución media en las métricas
        metric_differences = original_metrics - np.mean(permuted_metrics, axis=0)
        feature_importance[feature_idx] = np.mean(metric_differences)  # Media de la disminución en las métricas

    return feature_importance

def permutation_importance_multi(model, X_test, y_test, n_repeats=10):
    # Métricas originales sin permutar
    y_pred = np.argmax(model.predict(X_test), axis=1)
    #original_probs = model.predict(X_test)
    #y_pred = (original_probs > 0.5).astype(int)
    original_metrics = calculate_metrics(y_test, y_pred)

    feature_importance = np.zeros(X_test.shape[2])  # Array para almacenar la importancia de cada característica

    for feature_idx in range(X_test.shape[2]):
        #permuted_metrics = np.zeros((n_repeats, len(original_metrics)))  # Guardar las métricas permutadas
        permuted_metrics = np.zeros((n_repeats, 1))

        for i in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])  # Permutar la característica en todos los timesteps

            y_pred_permuted = np.argmax(model.predict(X_permuted), axis=1)
            permuted_metrics[i] = calculate_metrics(y_test, y_pred_permuted)

        # Calcular la disminución media en las métricas
        metric_differences = original_metrics - np.mean(permuted_metrics, axis=0)
        feature_importance[feature_idx] = np.mean(metric_differences)  # Media de la disminución en las métricas

    return feature_importance

# Inicializar un dataframe para almacenar las importancias
importance_df = pd.DataFrame(columns=feature_names)

# Iterar sobre las diferentes semillas
for seed in SEEDS:
    print(f'Evaluating seed {seed}')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Dividir los datos nuevamente con la misma semilla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # Cargar el modelo
    #model = tf.keras.models.load_model(f'modelos_fusionados/model_seed_{seed}.h5')
    model = tf.keras.models.load_model(f'modelos_4/model_seed_{seed}.h5')

    # Aplicar el método de importancia por permutación
    #feature_importances = permutation_importance(model, X_test, y_test)
    feature_importances = permutation_importance_multi(model, X_test, y_test)

    n = 5
    top_perm_features = np.argsort(feature_importances)[:n]
    print(f"Top {n} Permutation features for seed {seed}: {top_perm_features}")

    # Almacenar los resultados en el dataframe
    importance_df[f'Seed_{seed}'] = feature_importances

# Crear un gráfico de barras agrupadas para mostrar las importancias de las características por semilla
fig, ax = plt.subplots(figsize=(14, 8))

# Configurar el número de características y semillas
n_features = len(feature_names)
n_seeds = len(SEEDS)

# Índices de las barras para cada característica
bar_width = 0.15
indices = np.arange(n_features)

# Dibujar las barras para cada semilla
for i, seed in enumerate(SEEDS):
    ax.bar(indices + i * bar_width, importance_df[f'Seed_{seed}'], bar_width, label=f'Seed {i}')

# Configurar etiquetas y título
ax.set_xlabel('Features')
ax.set_ylabel('Feature Importance')
ax.set_xticks(indices + bar_width * (n_seeds / 2))
ax.set_xticklabels(feature_names, rotation=90)
ax.legend(title='Seed')

plt.tight_layout()
#plt.savefig('XXXX_features_final.png')
