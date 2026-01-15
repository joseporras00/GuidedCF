import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from utils import create_sequences3  # Assuming perturbate_series function is in utils.py

# Semillas para las diferentes particiones
SEEDS = [42, 123, 456, 789, 101112]

# Definir los métodos XAI a evaluar.
methods = ['IG', 'SG', 'GRAD', 'random']
#methods = ['random']

# Evaluar perturbaciones
perturbation_methods = ['zero','inverse', 'noise', 'swap', 'mean']

# Cargar datos.
X, y = create_sequences3('combined_preprocessed_data')
#X, y = create_sequences2('scaled_df_final_AAA_2013J')

sz = 90
special_value = -10.0
X = pad_sequences(X, maxlen=sz, dtype='float', padding='post', truncating='post', value=special_value)

important_features = [0, 1, 2, 3, 4, 5, 13]
X = X[:, :, important_features]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Función para perturbar las series temporales
def perturbate_series(series, relevance, method='zero', threshold=0.6):
    perturbed_series = series.copy()
    num_timesteps, num_variables = series.shape

    relevance_threshold = np.percentile(relevance, threshold * 100)
    relevant_indices = np.where(relevance > relevance_threshold)


    for var_idx in range(num_variables):
        var_series = series[:, var_idx]
        var_relevant_indices = relevant_indices[0][relevant_indices[1] == var_idx]

        if method == 'zero':
            var_series[var_relevant_indices] = 0
        elif method == 'inverse':
            var_series[var_relevant_indices] = var_series.max() - var_series[var_relevant_indices]
        elif method == 'swap':
            swap_indices = np.random.permutation(var_relevant_indices)
            var_series[var_relevant_indices] = var_series[swap_indices]
        elif method == 'noise':
            var_series[var_relevant_indices] += np.random.normal(0, 0.5, size=var_series[var_relevant_indices].shape)
        elif method == 'mean':
            mean_value = np.mean(var_series[var_relevant_indices])
            var_series[var_relevant_indices] = mean_value

        perturbed_series[:, var_idx] = var_series
    return perturbed_series

def perturbate_series_random(series, method='zero', threshold=0.9):
    perturbed_series = series.copy()
    num_timesteps, num_variables = series.shape

    relevance = np.random.rand(*series.shape)

    relevance_threshold = np.percentile(relevance, threshold * 100)
    relevant_indices = np.where(relevance > relevance_threshold)

    for var_idx in range(num_variables):
        var_series = series[:, var_idx]
        var_relevant_indices = relevant_indices[0][relevant_indices[1] == var_idx]

        if method == 'zero':
            var_series[var_relevant_indices] = 0
        elif method == 'inverse':
            var_series[var_relevant_indices] = var_series.max() - var_series[var_relevant_indices]
        elif method == 'swap':
            swap_indices = np.random.permutation(var_relevant_indices)
            var_series[var_relevant_indices] = var_series[swap_indices]
        elif method == 'noise':
            var_series[var_relevant_indices] += np.random.normal(0, 0.5, size=var_series[var_relevant_indices].shape)
        elif method == 'mean':
            mean_value = np.mean(var_series[var_relevant_indices])
            var_series[var_relevant_indices] = mean_value

        perturbed_series[:, var_idx] = var_series
    return perturbed_series


def evaluate_perturbations(X_test, y_test, model, method, int_mod=None, perturb_method='zero'):
    original_probs = model.predict(X_test)
    original_preds = (original_probs > 0.5).astype(int)
    original_acc = accuracy_score(y_test, original_preds)
    original_f1 = f1_score(y_test, original_preds)

    if len(np.unique(original_preds)) > 1:
        original_auc = roc_auc_score(y_test, original_probs)
    else:
        original_auc = np.nan  # No se puede calcular AUC con una sola clase predicha

    y_pert = []

    for idx in range(len(y_test)):
        item = np.array([X_test[idx]])
        label = int(np.argmax(y_test[idx]))

        if method=='random':
            perturbed_item = perturbate_series_random(item[0], method=perturb_method)
        else:
            exp = int_mod.explain(item, labels=label, TSR=False)
            perturbed_item = perturbate_series(item[0], exp, method=perturb_method)

        perturbed_pred = (model.predict(np.array([perturbed_item])) > 0.5).astype(int)

        y_pert.append(perturbed_pred[0])

    y_pert = np.array(y_pert).flatten()
    y_test_flat = np.array(y_test).flatten()

    perturbed_acc = accuracy_score(y_test_flat, y_pert)
    perturbed_f1 = f1_score(y_test_flat, y_pert)

    if len(np.unique(y_pert)) > 1:
        perturbed_auc = roc_auc_score(y_test_flat, y_pert)
    else:
        perturbed_auc = np.nan  # No se puede calcular AUC con una sola clase predicha

    return original_acc, original_f1, original_auc, perturbed_acc, perturbed_f1, perturbed_auc


# Inicializar dataframe para almacenar los resultados
results_df = pd.DataFrame(columns=['Seed', 'XAI_Method', 'Perturb_Method', 'Original_Acc', 'Perturbed_Acc', 'Random_Acc', 'Perturbed_F1', 'Random_F1', 'Perturbed_AUC', 'Random_AUC'])

# Iterar sobre las diferentes semillas
for seed in SEEDS:
    print(f'Evaluating seed {seed}')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Dividir los datos nuevamente con la misma semilla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # Cargar el modelo
    model = tf.keras.models.load_model(f'modelos_fusionados_7variables/model_seed_{seed}.h5')

    # Evaluar con métodos XAI
    for method in methods:

        print(f'XAI Method: {method}')

        for perturb_method in perturbation_methods:
            print(f'Evaluating perturbations with method {method} and perturbation method {perturb_method}')
            try:
                if method=='random':
                    original_acc, original_f1, original_auc, perturbed_acc, perturbed_f1, perturbed_auc=evaluate_perturbations(X_test, y_test, model, method, perturb_method=perturb_method)
                else:
                    int_mod = TSR(model, 90, X_test.shape[2], method=method, mode='time')
                    #original_acc, perturbed_acc = evaluate_perturbations(X_test, y_test, model, method, int_mod, perturb_method=perturb_method)
                    original_acc, original_f1, original_auc, perturbed_acc, perturbed_f1, perturbed_auc=evaluate_perturbations(X_test, y_test, model, method, int_mod, perturb_method=perturb_method)
                new_row = {
                    'Seed': seed,
                    'XAI_Method': method,
                    'Perturb_Method': perturb_method,
                    'Original_Acc': original_acc,
                    'Original_F1': original_f1,
                    'Original_AUC': original_auc,
                    'Perturbed_Acc': perturbed_acc,
                    'Perturbed_F1': perturbed_f1,
                    'Perturbed_AUC': perturbed_auc
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                print(f'Error evaluating perturbations with method {method} and perturbation method {perturb_method}: {e}')


# Guardar resultados en un CSV
results_df.to_csv('modelos_fusionados_7variables/results_final/results.csv', index=False)

# Calcular y mostrar medias y desviaciones estándar
summary_df = results_df.groupby(['XAI_Method', 'Perturb_Method']).agg(
    Mean_Original_Acc=('Original_Acc', 'mean'),
    Std_Original_Acc=('Original_Acc', 'std'),
    Mean_Original_F1=('Original_F1', 'mean'),
    Std_Original_F1=('Original_F1', 'std'),
    Mean_Original_AUC=('Original_AUC', 'mean'),
    Std_Original_AUC=('Original_AUC', 'std'),
    Mean_Perturbed_Acc=('Perturbed_Acc', 'mean'),
    Std_Perturbed_Acc=('Perturbed_Acc', 'std'),
    Mean_Perturbed_F1=('Perturbed_F1', 'mean'),
    Std_Perturbed_F1=('Perturbed_F1', 'std'),
    Mean_Perturbed_AUC=('Perturbed_AUC', 'mean'),
    Std_Perturbed_AUC=('Perturbed_AUC', 'std')
).reset_index()

# Guardar el resumen en un CSV
summary_df.to_csv('modelos_fusionados_7variables/results_final/summary_results.csv', index=False)
