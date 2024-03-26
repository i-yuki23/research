import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, data2, title):
    bins = range(min(min(data), min(data2)), max(max(data), max(data2)) + 2)
    plt.hist(data, bins=bins, align='left', color='skyblue', edgecolor='black')

    # Set titles and labels
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # plt.xticks(np.arange(0, 70, 10)) # １刻みにしても見にくいので２刻みにします
    # plt.yticks(np.arange(0, 310, 50)) # １刻みにしても見にくいので２刻みにします
    # Show histogram
    plt.show()