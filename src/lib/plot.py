import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, bins=10, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Plot a histogram for the given data.

    Parameters:
    data (np.ndarray): Input data for the histogram.
    bins (int or sequence): Number of bins or a sequence defining the bin edges.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.

    Returns:
    None
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data should be a NumPy array")
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_learning_curve(hist, title_fontsize=18, fontsize=14):
    for key in [k for k in hist.keys() if not k.startswith('val_')]:
        plt.figure()  # 新しい図を作成
        # トレーニングデータのプロット
        plt.plot(hist[key], label=f'Training {key}')
        # バリデーションデータのプロット
        val_key = f'val_{key}'
        plt.plot(hist[val_key], label=f'Validation {key}')
        plt.title(f'Training and Validation {key.capitalize()}', fontsize=title_fontsize)
        plt.xlabel('Epochs',  fontsize=title_fontsize)
        plt.ylabel(key.capitalize(),  fontsize=title_fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 
        plt.legend(fontsize=fontsize)
        plt.show()