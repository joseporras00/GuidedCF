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
                        random_instance[t, v] += np.random.uniform(0, self.tol)
                        random_instance[t, v] = np.clip(random_instance[t, v], self.value_range[0], self.value_range[1])
            random_instances.append(random_instance)
        return np.array(random_instances)

    def filter_by_class_change(self, instances):
        filtered_instances = []
        for instance in instances:
            output = self.model.predict(instance[np.newaxis, :, :])
            predicted_class = int(np.argmax(output))
            if predicted_class == self.target_class_idx:
                filtered_instances.append(instance)
        return np.array(filtered_instances)
    
    def filter_by_class_change2(self, instances):
        # Realizar la predicción en todas las instancias de una sola vez
        outputs = self.model.predict(instances)

        # Para clasificación multiclase: encontrar la clase con mayor probabilidad
        predicted_classes = np.argmax(outputs, axis=1)

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
    
