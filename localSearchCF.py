import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class LocalSearchCounterfactuals:
    def __init__(self, model, target_class_idx, relevances, num_samples=1000, value_range=(0.0, 1.0), tol=0.3, percentile=99):
        self.model = model
        self.target_class_idx = target_class_idx
        self.relevances = relevances
        self.num_samples = num_samples
        self.value_range = value_range
        self.tol = tol
        self.percentile = percentile
        self.threshold = np.percentile(self.relevances, self.percentile)
        
    def clear_memory(self):
        """Limpia todas las variables de la clase."""
        self.model = None
        self.target_class_idx = None
        self.relevances = None
        self.num_samples = None
        self.value_range = None
        self.tol = None
        self.percentile = None
        self.threshold = None
    
    def generate_random_instances(self, instance):
        random_instances = []
        for _ in range(self.num_samples):
            random_instance = instance[0].copy()
            for t in range(random_instance.shape[0]):
                for v in range(random_instance.shape[1]):
                    if self.relevances[t, v] > self.threshold:
                    #if np.random.rand() < self.relevances[t, v]:
                        #random_instance[t, v] += np.random.uniform(-self.tol, self.tol)
                        random_instance[t, v] += np.random.uniform(0, self.tol)
                        random_instance[t, v] = np.clip(random_instance[t, v], self.value_range[0], self.value_range[1])
            random_instances.append(random_instance)
        return np.array(random_instances)

    def filter_by_class_change(self, instances):
        filtered_instances = []
        for instance in instances:
            output = self.model.predict(instance[np.newaxis, :, :])
            #predicted_class = (output >= 0.5).astype(int)[0][0]
            predicted_class = int(np.argmax(output))
            if predicted_class == self.target_class_idx:
                filtered_instances.append(instance)
        return np.array(filtered_instances)
    
    def filter_by_class_change2(self, instances):
        # Realizar la predicción en todas las instancias de una sola vez
        outputs = self.model.predict(instances)
        #print("prediccion realizada...")

        # Para clasificación multiclase: encontrar la clase con mayor probabilidad
        predicted_classes = np.argmax(outputs, axis=1)
        #print(f'Prediccion: {predicted_classes}')

        # Filtrar las instancias donde la clase predicha es la clase objetivo
        filter_mask = (predicted_classes == self.target_class_idx)

        # Devolver solo las instancias que cambiaron a la clase objetivo
        filtered_instances = instances[filter_mask]

        return filtered_instances

    def select_best_instance(self, filtered_instances, original_instance):
        # Calculate DTW distances to the original instance
        #print("Calculating DTW distances...")
        dtw_distances = [dtw.distance_fast(instance.flatten(), original_instance[0].flatten()) for instance in filtered_instances]
        #print("Sorting instances by DTW distances...")
        sorted_indices = np.argsort(dtw_distances)
        
        # Select the top 20% instances with the smallest DTW distances
        top_percentile = int(0.2 * len(filtered_instances))
        top_filtered_instances = filtered_instances[sorted_indices[:top_percentile]]
        
        # Sort the top instances by the number of changes
        """best_instances = []
        for instance in top_filtered_instances:
            changes = np.abs(instance - original_instance[0])
            num_changes = np.sum(changes > self.tol)
            best_instances.append((instance, num_changes))
        best_instances = sorted(best_instances, key=lambda x: x[1])
        
        # Return the best instance
        return best_instances[0][0] if best_instances else np.full(original_instance.shape, np.nan)"""
        
        if len(top_filtered_instances) > 0:
                
            #changes = np.abs(top_filtered_instances - original_instance[0][np.newaxis, :, :])
            #num_changes = np.sum(changes > 0)  # Sum across all timesteps and features
            changes = np.sum(top_filtered_instances != original_instance[0][np.newaxis, :, :], axis=(1, 2))
            
            # Find the instance with the fewest changes
            best_instance_idx = np.argmin(changes)
            
            # Return the best instance
            return top_filtered_instances[best_instance_idx]
        else:
            return np.full(original_instance.shape, np.nan)
    
    def select_best_instance2(self, filtered_instances, original_instance):
        # Calculate DTW distances to the original instance using fastdtw
        dtw_distances = []
        for instance in filtered_instances:
            print(instance.flatten().shape)
            print(original_instance[0].flatten().shape)
            distance, _ = fastdtw(instance.flatten(), original_instance[0].flatten(), dist=euclidean)
            dtw_distances.append(distance)
        dtw_distances = np.array(dtw_distances)
        
        # Select the top 20% instances with the smallest DTW distances
        top_percentile = int(0.2 * len(filtered_instances))
        top_indices = np.argpartition(dtw_distances, top_percentile)[:top_percentile]
        top_filtered_instances = filtered_instances[top_indices]
        
        # Vectorize calculation of the number of changes
        changes = np.abs(top_filtered_instances - original_instance[0][np.newaxis, :, :])
        num_changes = np.sum(changes > self.tol, axis=(1, 2))  # Sum across all time steps and features
        
        # Find the instance with the fewest changes
        best_instance_idx = np.argmin(num_changes)
        
        # Return the best instance
        return top_filtered_instances[best_instance_idx] if len(top_filtered_instances) > 0 else np.full(original_instance.shape, np.nan)

    def generate_counterfactuals(self, instance):        
        # Generate random instances
        random_instances = self.generate_random_instances(instance)
        #print("creating counterfactuals...")
        # Filter instances by class change
        filtered_instances = self.filter_by_class_change(random_instances)
        #print("filtered instances:", filtered_instances)
        
        if len(filtered_instances) == 0:
            #print("No valid counterfactuals found.")
            return np.full(instance.shape, np.nan)
        
        # Select the best instance based on DTW distance and number of changes
        best_instance = self.select_best_instance(filtered_instances, instance)
        
        # Borrar toda la memoria de la clase
        self.clear_memory()
        
        return best_instance
    
"""# Ejemplo de uso
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import *
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_TF import Saliency_TF
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


cols=[0, 1, 2, 3, 4, 5, 13]
feature_names = pd.read_csv('data/combined_preprocessed_data.csv').columns[cols]
print(feature_names)
X, y = create_sequences3('combined_preprocessed_data')

sz = 90
special_value = -10.0
X = pad_sequences(X, maxlen=sz, dtype='float', padding='post', truncating='post', value=special_value)

important_features = [0, 1, 2, 3, 4, 5, 13]
X = X[:, :, important_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Transformar etiquetas a categorías (One-Hot Encoding)
y_train_onehot = to_categorical(y_train, num_classes=2)
y_test_onehot = to_categorical(y_test, num_classes=2)

model=tf.keras.models.load_model('modeloCF_4.h5')
y_pred = np.argmax(model.predict(X_test), axis=1)

# Interpretabilidad con Saliency Maps
int_mod = Saliency_TF(model, X_train.shape[1], X_train.shape[2], method='IG', mode='time')

class_1_correct_indices = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1]

num_instances = 30  # Número de instancias para procesar
indices = np.random.choice(class_1_correct_indices, num_instances, replace=False)  # Selecciona índices aleatorios de clase 1 correctamente predichos

target_class_idx = 0  # Índice de la clase objetivo
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
for idx in [3257]:
    item = np.array([X_test[idx]])
    label = int(np.argmax(y_test_onehot[idx]))
    print(f'Processing instance {idx} with label {label}')

    relevances = int_mod.explain(item, labels=0, TSR=True)
    print('Relevances creadas')
    cf_generator = LocalSearchCounterfactuals(model, 0, relevances)
    counterfactual = cf_generator.generate_counterfactuals(item)
    if not np.isnan(counterfactual).all():
        #print(counterfactual)
        plotear_cf2(item_orig=item, item_cf=np.array([counterfactual]), exp=relevances, feature_names=feature_names, save=f'imgFO/LocalCF_v3_{idx}_3.png')
    else:
        print("No valid counterfactual found.")
        break

    exp_model = COMTECF(model, (X_train, y_train), mode='time', backend='TF', method='brute', max_attempts=500, max_iter=500)
    exp = exp_model.explain(item, orig_class=1, target=0)
    if not np.isnan(exp[0][0]).any() and exp[1]==0:
        plotear_cf2(item_orig=item, item_cf=np.array([exp[0][0]]), exp=relevances, feature_names=feature_names, save=f'imgFO/COMTE_v3_{idx}_4.png')
     
        print(f'creada comte')
    else:
        print('comte no ha encontrado cf')
        break

    exp_model_tsevo = TSEvo(model, (X_train, y_train_onehot), mode='time', backend='TF')
    exp = exp_model_tsevo.explain(item, 1, target_y=0)
    print(f"TSEVO: {exp[0].shape}")
    if not np.isnan(exp[0]).any() and int(np.argmax(exp[1]))==0:
        exp=np.swapaxes(exp[0], 0, 1)
        #print(exp_aux.shape)
        plotear_cf2(item_orig=item, item_cf=np.array([exp]), exp=relevances, feature_names=feature_names, save=f'imgFO/TSevo_v3_{idx}_4.png')
        print(f'creada TSEvo')
    else:
        print('TSEvo no ha encontrado cf')
        break"""