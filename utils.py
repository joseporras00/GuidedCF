import time
from dtaidistance import dtw
from tslearn.neighbors import KNeighborsTimeSeries
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones auxiliares
def is_valid(original_pred, cf_pred, target_class):
    return original_pred != cf_pred and cf_pred == target_class

def calculate_sparsity(original_instance, counterfactual_instance):
    return np.sum(original_instance != counterfactual_instance)

def ensure_correct_shape_and_type(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if array.dtype != np.float64:
        array = array.astype(np.float64)
    return array

def calculate_similarity(original_instance, counterfactual_instance):
    original_instance = ensure_correct_shape_and_type(original_instance)
    counterfactual_instance = ensure_correct_shape_and_type(counterfactual_instance)
    return dtw.distance_fast(original_instance.flatten(), counterfactual_instance.flatten())

def is_plausible(counterfactual_instance, reference_population, k=5):
    knn = KNeighborsTimeSeries(n_neighbors=k, metric="dtw")
    knn.fit(reference_population)
    distances, _ = knn.kneighbors(counterfactual_instance)
    return np.mean(distances)

def count_changes(original_instance, counterfactual_instance):
    changes = np.sum(original_instance != counterfactual_instance, axis=(1, 2))
    total_changes = np.sum(changes)
    return total_changes

def calculate_diversity(counterfactuals):
    distances = [dtw.distance_fast(cf1.flatten(), cf2.flatten()) for i, cf1 in enumerate(counterfactuals) for j, cf2 in enumerate(counterfactuals) if i < j]
    return np.mean(distances)

def plotear(item, feature_names, mode='time', figsize=(20, 10), heatmap=False, save=None):
    """
    Plots explanation on the explained Sample.

    Arguments:
        item np.array: instance to be explained, if `mode = time`->`(1,time,feat)` or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation, if `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`.
        feature_names list: names of features.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    plt.style.use("classic")
    i = 0
    if mode == "time":
        item = np.swapaxes(item, -1, -2)
    else:
        print("NOT Time mode")

    # Definir parámetros de tamaño de fuente y grosor de línea
    label_fontsize = 14
    title_fontsize = 16
    line_width = 2.5

    # Ajustar tamaño de la figura
    #if len(item[0]) > 1:
    #    figsize = (figsize[0], figsize[1] * len(item[0]))  # Escalar el tamaño vertical de la figura

    if heatmap:
        fig, ax011 = plt.subplots(1, 1, figsize=figsize)
        ax011.tick_params(axis='y', labelsize=label_fontsize)
        ax011.set_title('Heatmap', fontsize=title_fontsize, fontweight="bold")
    elif len(item[0]) == 1:
        fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=False, figsize=figsize)
        axn012 = axn.twinx()
        sns.lineplot(x=range(0, len(item[0][0].reshape(-1))), y=item[0][0].flatten(), ax=axn012, color="black", linewidth=line_width)
        axn012.tick_params(axis='y', labelsize=label_fontsize)
        axn.set_xticks(range(0, len(item[0][0].reshape(-1)), 7))  # Mostrar ticks cada 5 unidades
        axn.set_xticklabels(range(0, len(item[0][0].reshape(-1)), 7), rotation=45, ha='right')  # Rotar etiquetas
    else:
        fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=False, figsize=figsize)
        cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # Reducir la anchura de la barra de color
        for channel in item[0]:
            axn012 = axn[i].twinx()
            sns.lineplot(x=range(0, len(channel.reshape(-1))), y=channel.flatten(), ax=axn012, color="black", linewidth=line_width)
            axn012.set_ylim(0, 1)
            axn[i].tick_params(axis='y', right=False, left=True, labelsize=label_fontsize)
            axn012.set_yticks([0.5])  # Ubicar la etiqueta en el centro del eje Y
            axn012.set_yticklabels([feature_names[i]], fontsize=label_fontsize, rotation=90)  
            axn012.yaxis.set_label_position("right")
            
            #axn012.set_title(f"Feature: {feature_names[i]}", fontsize=title_fontsize, fontweight="bold")
            axn[i].set_xticks(range(0, len(channel.reshape(-1)), 7))  # Mostrar ticks cada 5 unidades
            axn[i].set_xticklabels(range(0, len(channel.reshape(-1)), 7), rotation=45, ha='right')  # Rotar etiquetas
            plt.xlabel("Time", fontweight="bold", fontsize=label_fontsize)
            i += 1
        
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Ajustar automáticamente los márgenes
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def plotear2(item, exp, feature_names, mode='time', figsize=(20, 10), heatmap=False, save=None):
    """
    Plots explanation on the explained Sample.

    Arguments:
        item np.array: instance to be explained, if `mode = time`->`(1,time,feat)` or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation, if `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`.
        feature_names list: names of features.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    plt.style.use("classic")
    i = 0
    if mode == "time":
        item = np.swapaxes(item, -1, -2)
        exp = np.swapaxes(exp, -1, -2)
    else:
        print("NOT Time mode")

    # Definir parámetros de tamaño de fuente y grosor de línea
    label_fontsize = 14
    title_fontsize = 16
    line_width = 2.5

    # Ajustar tamaño de la figura
    #if len(item[0]) > 1:
    #    figsize = (figsize[0], figsize[1] * len(item[0]))  # Escalar el tamaño vertical de la figura

    if heatmap:
        fig, ax011 = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(exp, fmt="g", cmap="viridis", cbar=True, ax=ax011, yticklabels=feature_names)
        ax011.tick_params(axis='y', labelsize=label_fontsize)
        ax011.set_title('Heatmap', fontsize=title_fontsize, fontweight="bold")
    elif len(item[0]) == 1:
        fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=False, figsize=figsize)
        axn012 = axn.twinx()
        sns.heatmap(exp.reshape(1, -1), fmt="g", cmap="viridis", ax=axn, yticklabels=feature_names)
        sns.lineplot(x=range(0, len(item[0][0].reshape(-1))), y=item[0][0].flatten(), ax=axn012, color="white", linewidth=line_width)
        axn012.tick_params(axis='y', labelsize=label_fontsize)
        axn.set_xticks(range(0, len(item[0][0].reshape(-1)), 7))  # Mostrar ticks cada 5 unidades
        axn.set_xticklabels(range(0, len(item[0][0].reshape(-1)), 7), rotation=45, ha='right')  # Rotar etiquetas
    else:
        fig, axn = plt.subplots(len(item[0]), 1, sharex=True, sharey=False, figsize=figsize)
        cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # Reducir la anchura de la barra de color
        for channel in item[0]:
            axn012 = axn[i].twinx()
            sns.heatmap(exp[i].reshape(1, -1), fmt="g", cmap="viridis", cbar=i == 0, cbar_ax=None if i else cbar_ax, ax=axn[i], yticklabels=[feature_names[i]])
            sns.lineplot(x=range(0, len(channel.reshape(-1))), y=channel.flatten(), ax=axn012, color="white", linewidth=line_width)
            axn012.set_ylim(0, 1)
            axn[i].tick_params(axis='y', labelsize=label_fontsize)
            axn012.tick_params(axis='y', labelsize=label_fontsize)
            #axn012.set_title(f"Feature: {feature_names[i]}", fontsize=title_fontsize, fontweight="bold")
            axn[i].set_xticks(range(0, len(channel.reshape(-1)), 7))  # Mostrar ticks cada 5 unidades
            axn[i].set_xticklabels(range(0, len(channel.reshape(-1)), 7), rotation=45, ha='right')  # Rotar etiquetas
            plt.xlabel("Time", fontweight="bold", fontsize=label_fontsize)
            i += 1
        
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Ajustar automáticamente los márgenes
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        
        
def plotear_cf2(item_orig, item_cf, exp, feature_names, mode='time', figsize=(20, 8), heatmap=False, save=None):
    """
    Plots explanation on the explained Sample and its counterfactual.

    Arguments:
        item_orig np.array: original instance to be explained, if `mode = time`->`(1,time,feat)` or `mode = feat`->`(1,feat,time)`.
        item_cf np.array: counterfactual instance, if `mode = time`->`(1,time,feat)` or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation, if `mode = time`->`(time,feat)` or `mode = feat`->`(feat,time)`.
        feature_names list: names of features.
        mode str: 'time' or 'feat' to indicate the mode of plotting.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    plt.style.use("classic")
    i = 0
    if mode == "time":
        item_orig = np.swapaxes(item_orig, -1, -2)
        item_cf = np.swapaxes(item_cf, -1, -2)
        exp = np.swapaxes(exp, -1, -2)
    else:
        print("NOT Time mode")

    # Definir parámetros de tamaño de fuente y grosor de línea
    label_fontsize = 14
    title_fontsize = 16
    line_width = 1.5

    # Ajustar tamaño de la figura
    #if len(item_orig[0]) > 1:
    #    figsize = (figsize[0], figsize[1] * len(item_orig[0]))  # Escalar el tamaño vertical de la figura

    if heatmap:
        fig, ax011 = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(exp, fmt="g", cmap="viridis", cbar=True, ax=ax011, yticklabels=feature_names)
        ax011.tick_params(axis='y', labelsize=label_fontsize)
        ax011.set_title('Heatmap', fontsize=title_fontsize, fontweight="bold")
    elif len(item_orig[0]) == 1:
        fig, axn = plt.subplots(len(item_orig[0]), 1, sharex=True, sharey=False, figsize=figsize)
        axn012 = axn.twinx()
        sns.heatmap(exp.reshape(1, -1), fmt="g", cmap="viridis", ax=axn, yticklabels=feature_names)
        sns.lineplot(x=range(0, len(item_orig[0][0].reshape(-1))), y=item_orig[0][0].flatten(), ax=axn012, color="blue", label='Original', linewidth=line_width)
        sns.lineplot(x=range(0, len(item_cf[0][0].reshape(-1))), y=item_cf[0][0].flatten(), ax=axn012, color="red", label='Counterfactual', linewidth=line_width)
        axn012.legend(fontsize=label_fontsize)
        axn.set_xticks(range(0, len(item_orig[0][0].reshape(-1)), 5))  # Mostrar ticks cada 5 unidades
        axn.set_xticklabels(range(0, len(item_orig[0][0].reshape(-1)), 5), rotation=45, ha='right')  # Rotar etiquetas
    else:
        fig, axn = plt.subplots(len(item_orig[0]), 1, sharex=True, sharey=False, figsize=figsize)
        cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # Reducir la anchura de la barra de color
        for i, channel in enumerate(item_orig[0]):
            axn012 = axn[i].twinx()
            sns.heatmap(exp[i].reshape(1, -1), fmt="g", cmap="viridis", cbar=i == 0, cbar_ax=None if i else cbar_ax, ax=axn[i], yticklabels=[feature_names[i]])
            sns.lineplot(x=range(0, len(item_cf[0][i].reshape(-1))), y=item_cf[0][i].flatten(), ax=axn012, color="red", linewidth=line_width)
            sns.lineplot(x=range(0, len(channel.reshape(-1))), y=channel.flatten(), ax=axn012, color="white", linewidth=line_width)
            axn012.set_ylim(0, 1)
            axn[i].tick_params(axis='y', labelsize=label_fontsize)
            axn012.tick_params(axis='y', labelsize=label_fontsize)
            #axn012.set_title(f"Feature: {feature_names[i]}", fontsize=title_fontsize, fontweight="bold")
            axn[i].set_xticks(range(0, len(channel.reshape(-1)), 7))  # Mostrar ticks cada 5 unidades
            axn[i].set_xticklabels(range(0, len(channel.reshape(-1)), 7), rotation=45, ha='right')  # Rotar etiquetas
            plt.xlabel("Time", fontweight="bold", fontsize=label_fontsize)
            i += 1
        
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Ajustar automáticamente los márgenes
    if save is None:
        plt.show()
    else:
        plt.savefig(save)



