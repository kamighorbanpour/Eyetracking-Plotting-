
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def preprocess_data(data):
    return data.dropna(subset=['Gaze point X [DACS px]', 'Gaze point Y [DACS px]'])

def plot_heatmap(data, resolution=100):
    x = data['Gaze point X [DACS px]']
    y = data['Gaze point Y [DACS px]']

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.mgrid[xmin:xmax:complex(resolution), ymin:ymax:complex(resolution)]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.jet
    cmap.set_bad('white', 1.)

    heatmap = ax.imshow(np.rot90(f), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.colorbar(heatmap, label='Gaze Density')
    plt.title('Heatmap of Gaze Points')
    plt.xlabel('Gaze Point X [DACS px]')
    plt.ylabel('Gaze Point Y [DACS px]')
    plt.show()

if __name__ == '__main__':
    file_path = 'D:\Downloads\Project1 Data export.tsv' 
    data = load_data(file_path)
    filtered_data = preprocess_data(data)
    plot_heatmap(filtered_data, resolution=100)
