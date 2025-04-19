import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
import shutil


def make_spectrogramms(df, block_size, window_length, overlap):
    spectrogram_info = []
    time_series = df.iloc[:, 1:-2].to_numpy()
    for start_idx in range(0, len(time_series), block_size):
        cur_time_series = time_series[start_idx:start_idx + block_size]

        Sxx_accumulated = None
        
        for i in range(cur_time_series.shape[1]):
            f, t, Sxx = spectrogram(cur_time_series[:, i], nperseg=window_length, noverlap=overlap)
            
            if Sxx_accumulated is None:
                Sxx_accumulated = np.zeros_like(Sxx)
            
            Sxx_accumulated += Sxx
        
        Sxx_mean = Sxx_accumulated / cur_time_series.shape[1]
        
        if np.all(Sxx_mean <= 1e-10):
            print(f"Warning: Low signal power at index {start_idx}")
            continue

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx_mean + 1e-10), shading='gouraud')
        
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.colorbar().remove()
        
        # Сохранение изображения без полей
        filename = f"spectrogram_{start_idx}_{block_size}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        spectrogram_info.append({'filename': filename, 'start_time': start_idx})

    # Создание DataFrame из накопленной информации
    spectrogram_df = pd.DataFrame(spectrogram_info)
    # Сохранение DataFrame в CSV файл
    spectrogram_df.to_csv(f'spectrogram_info_{block_size}.csv', index=False)
    print(f"Спектрограммы созданы и информация сохранена в 'spectrogram_info_{block_size}.csv'.")
    filename = np.array([[filename] * block_size for filename in spectrogram_df["filename"]]).flatten()
    df["filename"] = filename.tolist() + [None] * (df.shape[0] - len(filename))
    cv_df = df[["filename", "is_AKR"]].dropna()
    cv_df.to_csv(f"cv_dataset_{block_size}.csv", index=False)
    df = cv_df.drop_duplicates()

    # создание директорий со спектрограммами
    base_dir = f'cv_classification_{block_size}'
    akr_dir = os.path.join(base_dir, 'AKR')
    not_akr_dir = os.path.join(base_dir, 'not_AKR')

    os.makedirs(akr_dir, exist_ok=True)
    os.makedirs(not_akr_dir, exist_ok=True)

    # Копирование файлов в соответствующие директории
    for _, row in df.iterrows():
        src_path = row['filename']
        dest_dir = akr_dir if row['is_AKR'] else not_akr_dir
        shutil.copy(src_path, dest_dir)


def main():
    df = pd.read_csv(f"windowed_df_{4}.csv")
    make_spectrogramms(df, 4, 2, 1)
    df = pd.read_csv(f"windowed_df_{16}.csv")
    make_spectrogramms(df, 16, 4, 2)
    df = pd.read_csv(f"windowed_df_{32}.csv")
    make_spectrogramms(df, 32, 4, 2)


if __name__ == "__main__":
    main()
