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

def plot_learning_curve(hist):
    for key in [k for k in hist.keys() if not k.startswith('val_')]:
        plt.figure()  # 新しい図を作成
        # トレーニングデータのプロット
        plt.plot(hist[key], label=f'Training {key}')
        # バリデーションデータのプロット
        val_key = f'val_{key}'
        plt.plot(hist[val_key], label=f'Validation {key}')
        plt.title(f'Training and Validation {key.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(key.capitalize())
        plt.legend()
        plt.show()