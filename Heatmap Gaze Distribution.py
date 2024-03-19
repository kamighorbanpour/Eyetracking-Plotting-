
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def load_data(file_path):
    return pd.read_csv('D:\Downloads\Project1 Data export.tsv', sep='\t')

def preprocess_data(data):
    return data.dropna(subset=['Gaze point X [DACS px]', 'Gaze point Y [DACS px]'])

def create_heatmap(data, x_resolution=100, y_resolution=100, sigma=2):
    x_bins = np.linspace(data['Gaze point X [DACS px]'].min(), data['Gaze point X [DACS px]'].max(), x_resolution)
    y_bins = np.linspace(data['Gaze point Y [DACS px]'].min(), data['Gaze point Y [DACS px]'].max(), y_resolution)

    hist, xedges, yedges = np.histogram2d(data['Gaze point X [DACS px]'], data['Gaze point Y [DACS px]'], bins=[x_bins, y_bins])

    hist_smoothed = gaussian_filter(hist, sigma=sigma)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(hist_smoothed.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label='Gaze Density')
    plt.title('Heatmap of Gaze Distribution')
    plt.xlabel('Screen X Coordinate')
    plt.ylabel('Screen Y Coordinate')
    plt.show()

if __name__ == '__main__':
    file_path = 'D:\Downloads\Project1 Data export.tsv'

    data = load_data(file_path)
    filtered_data = preprocess_data(data)

    create_heatmap(filtered_data, x_resolution=100, y_resolution=100, sigma=2)
