
import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter_data(file_path):
    data = pd.read_csv(file_path, sep='\t')

    data_filtered = data.dropna(subset=['Gaze point X [DACS px]', 'Gaze point Y [DACS px]'])
    
    return data_filtered

def plot_gaze_over_time(data_filtered):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data_filtered['Recording timestamp [ms]'], data_filtered['Gaze point X [DACS px]'], alpha=0.5, label='Gaze X')
    plt.title('Gaze X Coordinate Over Time')
    plt.xlabel('Time [ms]')
    plt.ylabel('Gaze Point X [DACS px]')

    plt.subplot(1, 2, 2)
    plt.scatter(data_filtered['Recording timestamp [ms]'], data_filtered['Gaze point Y [DACS px]'], alpha=0.5, color='red', label='Gaze Y')
    plt.title('Gaze Y Coordinate Over Time')
    plt.xlabel('Time [ms]')
    plt.ylabel('Gaze Point Y [DACS px]')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'D:\Downloads\Project1 Data export.tsv' 
    data_filtered = load_and_filter_data(file_path)
    plot_gaze_over_time(data_filtered)
