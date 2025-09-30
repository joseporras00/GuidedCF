import random
import warnings

import numpy as np
import pandas as pd
from deap import creator, tools
from pyts.utils import windowed_view
from scipy.fft import irfft, rfft

warnings.filterwarnings("ignore", category=DeprecationWarning)


def eval(x, mop, return_values_of):
    """
    Help Function.
    Args:
            x (np.array): instance to evaluate.
            mop (pymop.Problem): instance of Multiobjective Problem.
            return_values_of (np.array): Specify array to return.

    Returns:
            [np.array]: fitnessvalues
    """
    # print('eval')
    return mop.evaluate([x], return_values_of)  # , mop.prediction


def evaluate_pop(pop, toolbox):
    for ind in pop:
        out = toolbox.evaluate(ind)
        # print(out)
        if type(out) == tuple:
            ind.fitness.values = out
        else:
            ind.fitness.values = tuple(out[0])
        # print('IND Fitness',ind.fitness.values)
    return pop

max_change = 0.25

import numpy as np

def apply_limit(original_value, new_value, max_change):
    # Si original_value y new_value son arrays, aplica el límite a cada elemento
    if isinstance(original_value, np.ndarray) and isinstance(new_value, np.ndarray):
        # Asegurarse de que ambas matrices tengan la misma forma
        assert original_value.shape == new_value.shape, "Las formas no coinciden para aplicar el límite"
        # Aplica la función elemento por elemento
        limited_value = np.where(
            np.abs(new_value - original_value) > max_change,
            original_value + np.sign(new_value - original_value) * max_change,
            new_value
        )
        return limited_value
    else:
        # Caso escalar: aplica el límite directamente
        difference = new_value - original_value
        if abs(difference) > max_change:
            new_value = original_value + np.sign(difference) * max_change
        return new_value



def recombine(ind1, ind2, ig_values, percentile):
    """Crossover guiado por IG"""

    window_size1 = ind1.window
    window_size2 = ind2.window
    shape = np.array(ind1).shape[1]
    num_channels = len(ind1.channels)
    channel1 = ind1.channels
    mutation = ind1.mutation
    
    percentile_95 = np.percentile(ig_values, percentile)
    
    # Si el tamaño de ventana es 1, aplicar crossover uniforme usando IG
    if window_size1 == 1:
        ind1_array = np.array(ind1).reshape(num_channels, shape)
        ind2_array = np.array(ind2).reshape(num_channels, shape)
        
        for t in range(shape):
            for v in range(num_channels):
                # Realiza el intercambio solo si `ig_values` es alto en esa posición
                if ig_values[t, v] > percentile_95:  # Ajusta el umbral según sea necesario
                    ind1_array[v, t], ind2_array[v, t] = ind2_array[v, t], ind1_array[v, t]
        ind1 = ind1_array
        ind2 = ind2_array

    else:
        # Si el tamaño de ventana es mayor que 1, utilizar ventanas en el crossover
        if (shape / window_size1).is_integer():
            ind1_windows = windowed_view(
                np.array(ind1).reshape(num_channels, shape),
                window_size1,
                window_step=window_size1,
            )
            ind2_windows = windowed_view(
                np.array(ind2).reshape(num_channels, shape),
                window_size1,
                window_step=window_size1,
            )
        else:
            # Agregar padding si es necesario para ajustar el tamaño de la ventana
            shape_new = window_size1 * (int(shape / window_size1) + 1)
            padded1 = np.zeros((num_channels, shape_new))
            padded2 = np.zeros((num_channels, shape_new))
            padded1[:, :shape] = np.array(ind1).reshape(num_channels, shape)
            padded2[:, :shape] = np.array(ind2).reshape(num_channels, shape)

            ind1_windows = windowed_view(
                padded1,
                window_size1,
                window_step=window_size1,
            )
            ind2_windows = windowed_view(
                padded2,
                window_size1,
                window_step=window_size1,
            )

        # Crossover en canales específicos con IG
        if num_channels == 1:
            for i in range(ind1_windows.shape[1]):
                if ig_values[i * window_size1, 0] > percentile_95:
                    ind1_windows[0, i], ind2_windows[0, i] = ind2_windows[0, i], ind1_windows[0, i]

        else:
            items = np.where(channel1 == 1)
            if len(items[0]) != 0:
                for item in items[0]:
                    for i in range(ind1_windows.shape[1]):
                        if ig_values[i * window_size1, item] > percentile_95:
                            ind1_windows[item, i], ind2_windows[item, i] = ind2_windows[item, i], ind1_windows[item, i]

    # Reconstruir los individuos
    ind1 = np.array(ind1).reshape(num_channels, -1).tolist()
    ind2 = np.array(ind2).reshape(num_channels, -1).tolist()
    ind1 = creator.Individual(ind1)
    ind2 = creator.Individual(ind2)

    # Restaurar atributos adicionales
    ind1.window = window_size1
    ind2.window = window_size2
    ind1.mutation = mutation
    ind2.mutation = mutation
    ind1.channels = channel1
    ind2.channels = channel1

    return ind1, ind2


def mutate(individual, means, sigmas, indpb, uopb, ig_values, top_fraction=0.05):
    """Gaussian Mutation guiada por IG"""

    window = individual.window
    channels = individual.channels
    items = np.where(channels == 1)

    if len(items[0]) != 0:
        channel = random.choice(items[0])
        means = means[channel]
        sigmas = sigmas[channel]

        # Obtener los índices de los valores de `ig_values` para este canal, ordenados de mayor a menor
        sorted_indices = np.argsort(ig_values[:, channel])[::-1]
        
        # Seleccionar solo los índices con los `ig_values` más altos
        num_top_indices = max(1, int(len(sorted_indices) * top_fraction))
        top_indices = sorted_indices[:num_top_indices]

        # Aplicar mutación solo en los índices seleccionados
        for i in top_indices:
            m = means[i]
            s = sigmas[i]
            if random.random() < indpb:
                #individual[channel][i] = random.gauss(m, s)
                individual[channel][i] = apply_limit(individual[channel][i], random.gauss(m, s), 0.25)

    # Mutar parámetros adicionales si es necesario
    window, channels = mutate_hyperperameter(
        individual, window, channels, len(channels)
    )
    
    # Reconstruir el individuo con los nuevos valores mutados
    ind = creator.Individual(individual)
    ind.window = window
    ind.channel = channels
    ind.mutation = "mean"
    return (ind,)


def create_mstats():
    """Logging the Stats"""
    stats_y_distance = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_x_distance = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_changed_features = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(
        stats_y_distance=stats_y_distance,
        stats_x_distance=stats_x_distance,
        stats_changed_features=stats_changed_features,
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def create_logbook():
    logbook = tools.Logbook()
    logbook.header = (
        "gen",
        "pop",
        "evals",
        "stats_y_distance",
        "stats_x_distance",
        "stats_changed_features",
    )
    # logbook.chapters["fitness"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_y_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_x_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_changed_features"].header = "std", "min", "avg", "max"
    return logbook


def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
    Parameters
    ----------
    ind1: DEAP individual from the GP population
        First individual to compare
    ind2: DEAP individual from the GP population
        Second individual to compare
    Returns
    ----------
    individuals_equal: bool
        Boolean indicating whether the two individuals are equal on
        the Pareto front
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)


def authentic_opposing_information(ind1, reference_set, ig_values, top_fraction=0.05):
    window = ind1.window
    num_channels = len(ind1.channels)
    channels = ind1.channels
    shape = np.array(ind1).shape[-1]
    sample_series = random.choice(reference_set)

    # Ajustar `ind1` y `sample_series` según el tamaño de ventana
    if (shape / window).is_integer():
        ind1 = windowed_view(
            np.array(ind1).reshape(num_channels, shape), window, window_step=window
        )
        sample_series = windowed_view(
            sample_series.reshape(num_channels, shape), window, window_step=window
        )
    else:
        shape_new = window * (int(shape / window) + 1)
        padded = np.zeros((num_channels, shape_new))
        sample_padded = np.zeros((num_channels, shape_new))
        padded[:, :shape] = np.array(ind1).reshape(num_channels, shape)
        sample_padded[:, :shape] = sample_series.reshape(num_channels, shape)
        ind1 = windowed_view(
            np.array(padded).reshape(num_channels, shape_new), window, window_step=window
        )
        sample_series = windowed_view(sample_padded, window, window_step=window)

    # Asegurar que los índices sean válidos dentro de las dimensiones actuales
    items = np.where(channels == 1)
    if len(items[0]) != 0:
        channel = random.choice(items[0])
        
        # Obtener índices de mayor IG en el canal seleccionado, ajustados a `ind1` actual
        sorted_indices = np.argsort(ig_values[:, channel])[::-1]
        num_top_indices = max(1, int(len(sorted_indices) * top_fraction))
        top_indices = sorted_indices[:num_top_indices]

        # Ajustar `top_indices` al tamaño actual de `ind1[channel]`
        max_index = ind1.shape[1] - 1  # Límite máximo de índice en la dimensión actual
        adjusted_indices = [idx for idx in top_indices if idx <= max_index]

        for index in adjusted_indices:
            #ind1[channel, index] = sample_series[channel, index]
            ind1[channel,index] = apply_limit(ind1[channel,index], sample_series[channel, index], 0.25)

    # Restaurar la forma original de `ind1`
    new_shape = ind1.reshape(num_channels, -1).shape[1]
    if new_shape > shape:
        diff = new_shape - shape
        ind1 = np.array(ind1).reshape(num_channels, -1)[:, :shape]

    ind1 = ind1.reshape(num_channels, -1)
    ind1 = creator.Individual(np.array(ind1).reshape(num_channels, -1).tolist())

    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.window = window
    ind1.channels = channels
    ind1.mutation = "auth"
    return (ind1,)



def frequency_band_mapping(ind1, reference_set, ig_values, top_fraction=0.05):
    num_channels = len(ind1.channels)
    channels = ind1.channels
    window = ind1.window
    ind1 = np.array(ind1).reshape(1, -1, reference_set.shape[-1])
    shape = ind1.shape
    fourier_timeseries = rfft(ind1)  # Fourier transformation of timeseries
    fourier_reference_set = rfft(np.array(reference_set))  # Fourier transformation of reference set
    len_fourier = fourier_timeseries.shape[-1]

    # Define variables
    length = 1
    num_slices = 1

    # Create slices based on Fourier frequency bands
    slices_start_end_value = pd.DataFrame(columns=["Slice_number", "Start", "End"])
    new_row = pd.DataFrame([[0, 0, 1]], columns=["Slice_number", "Start", "End"])
    slices_start_end_value = pd.concat([slices_start_end_value, new_row], ignore_index=True)

    start_idx = length
    end_idx = length
    while length < len_fourier:
        start_idx = length
        end_idx = start_idx + num_slices**2
        end_idx = min(end_idx, len_fourier)

        new_row = pd.DataFrame([[num_slices, start_idx, end_idx]], columns=["Slice_number", "Start", "End"])
        slices_start_end_value = pd.concat([slices_start_end_value, new_row], ignore_index=True)

        length = length + end_idx - start_idx
        num_slices += 1

    feature = np.where(channels == 1)
    if len(feature[0]) != 0:
        # Seleccionar el canal a modificar
        num_feature = random.choice(feature[0])
        
        # Seleccionar las bandas de frecuencia más importantes usando ig_values
        sorted_indices = np.argsort(ig_values[:, num_feature])[::-1]
        num_top_indices = max(1, int(len(sorted_indices) * top_fraction))
        top_indices = sorted_indices[:num_top_indices]

        # Copia temporal del series de Fourier para realizar la perturbación
        tmp_fourier_series = np.array(fourier_timeseries.copy())
        max_row_idx = fourier_reference_set.shape[0]

        # Modificar solo las bandas de frecuencia seleccionadas
        for idx in top_indices:
            start_idx = slices_start_end_value["Start"][idx]
            end_idx = slices_start_end_value["End"][idx]
            rand_idx = np.random.randint(0, max_row_idx)
            
            # Aplicar la mutación en el rango seleccionado de Fourier
            tmp_fourier_series[0, num_feature, start_idx:end_idx] = fourier_reference_set[
                rand_idx, num_feature, start_idx:end_idx
            ].copy()

        # Transformar de vuelta a la representación de tiempo
        perturbed_fourier_retransform = irfft(tmp_fourier_series, n=shape[2])
        ind1 = creator.Individual(
            np.array(perturbed_fourier_retransform).reshape(shape[1], shape[2]).tolist()
        )
    else:
        ind1 = creator.Individual(np.array(ind1).reshape(shape[1], shape[2]).tolist())

    # Mutar los hiperparámetros y reconstruir el individuo con los nuevos valores
    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.channels = channels
    ind1.window = window
    ind1.mutation = "freq"
    return (ind1,)


def mutate_mean(ind1, reference_set, ig_values, top_fraction=0.05):
    window = ind1.window
    num_channels = len(ind1.channels)
    channels = ind1.channels

    # Calcular medias y desviaciones del conjunto de referencia
    means = reference_set.mean(axis=0)
    sigmas = reference_set.std(axis=0)

    # Seleccionar los índices con mayor importancia de IG para cada canal
    items = np.where(channels == 1)
    if len(items[0]) != 0:
        channel = random.choice(items[0])  # Selecciona un canal de los activos
        sorted_indices = np.argsort(ig_values[:, channel])[::-1]  # Índices ordenados por IG en ese canal
        num_top_indices = max(1, int(len(sorted_indices) * top_fraction))  # Número de índices top según `top_fraction`
        top_indices = sorted_indices[:num_top_indices]  # Índices más relevantes según IG

        # Aplicar la mutación solo en los índices seleccionados
        for idx in top_indices:
            # Solo realizar la mutación en los genes relevantes con probabilidad `indpb`
            if random.random() < 0.5:  # Probabilidad de mutación `indpb`
                ind1[channel][idx] = random.gauss(means[channel][idx], sigmas[channel][idx])

    # Mutar los hiperparámetros y reconstruir el individuo con los nuevos valores
    window, channels = mutate_hyperperameter(ind1, window, channels, num_channels)
    ind1.channels = channels
    ind1.window = window
    ind1.mutation = "mean"
    return (ind1,)



def mutate_both(ind1, reference_set, ig_values, top_fraction):
    """Still TODO"""
    if ind1.mutation == "auth":
        (ind1,) = authentic_opposing_information(ind1, reference_set, ig_values=ig_values, top_fraction=top_fraction)
    elif ind1.mutation == "freq":
        (ind1,) = frequency_band_mapping(ind1, reference_set, ig_values=ig_values, top_fraction=top_fraction)
    if ind1.mutation == "mean":
        means = reference_set.mean(axis=0)  #
        sigmas = reference_set.std(axis=0)
        (ind1,) = mutate(ind1, means=means, sigmas=sigmas, indpb=0.56, uopb=0.32, ig_values=ig_values, top_fraction=top_fraction)
    return (ind1,)


def mutate_hyperperameter(ind1, window, channels, num_channels):
    window = window
    channels = channels
    if random.random() < 0.5:
        window = random.randint(1, np.floor(0.5 * np.array(ind1).shape[-1]))
    return window, channels