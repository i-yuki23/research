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

def plot_scatter(x, y, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis", color='blue', cmap=None, save_path=None):
    """
    Draws a scatter plot.

    Parameters:
    - x: List or array-like, x-axis data points
    - y: List or array-like, y-axis data points
    - title: str, title of the plot (default: "Scatter Plot")
    - xlabel: str, label for the x-axis (default: "X-axis")
    - ylabel: str, label for the y-axis (default: "Y-axis")
    - color: str or array-like, color of the points (default: "blue")
    - cmap: str or Colormap, Colormap for the points (default: None)
    - save_path: str, path to save the plot as an image (default: None)

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=color, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if cmap:
        plt.colorbar(scatter)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_hist2d(x, y, bins=30, title="2D Histogram", xlabel="X-axis", ylabel="Y-axis", cmap='viridis', save_path=None):
    """
    Draws a 2D histogram.

    Parameters:
    - x: List or array-like, x-axis data points
    - y: List or array-like, y-axis data points
    - bins: int or [int, int] or array-like or [array, array], number of bins or bin edges (default: 30)
    - title: str, title of the plot (default: "2D Histogram")
    - xlabel: str, label for the x-axis (default: "X-axis")
    - ylabel: str, label for the y-axis (default: "Y-axis")
    - cmap: str or Colormap, Colormap for the histogram (default: 'viridis')
    - save_path: str, path to save the plot as an image (default: None)

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    hist = plt.hist2d(x, y, bins=bins, cmap=cmap)
    plt.colorbar(hist[3])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_3d_hist2d(x, y, bins=30, title="3D Histogram", xlabel="X-axis", ylabel="Y-axis", zlabel="Frequency", cmap='viridis'):
    """
    Draws a 3D histogram based on 2D data, with interactive rotation.

    Parameters:
    - x: List or array-like, x-axis data points
    - y: List or array-like, y-axis data points
    - bins: int or [int, int] or array-like or [array, array], number of bins or bin edges (default: 30)
    - title: str, title of the plot (default: "3D Histogram")
    - xlabel: str, label for the x-axis (default: "X-axis")
    - ylabel: str, label for the y-axis (default: "Y-axis")
    - zlabel: str, label for the z-axis (default: "Frequency")
    - cmap: str or Colormap, Colormap for the histogram (default: 'viridis')

    Returns:
    - None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    # Construct arrays with the dimensions for the bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    
    # Plotting the bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap=cmap, color=plt.cm.get_cmap(cmap)(dz / np.max(dz)))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
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