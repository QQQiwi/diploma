from spacepy import pycdf
import datetime
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def get_cdf_list():
    """
        Функция для считывания спутниковых данных в Python-объект
    """
    cdf1 = pycdf.CDF('data/wi_h1_wav_20200701_v01.cdf')
    cdf2 = pycdf.CDF('data/wi_h1_wav_20200702_v01.cdf')
    cdf3 = pycdf.CDF('data/wi_h1_wav_20200703_v01.cdf')
    cdf4 = pycdf.CDF('data/wi_h1_wav_20200704_v01.cdf')
    cdf5 = pycdf.CDF('data/wi_h1_wav_20200705_v01.cdf')
    cdf6 = pycdf.CDF('data/wi_h1_wav_20200706_v01.cdf')
    cdf7 = pycdf.CDF('data/wi_h1_wav_20200707_v01.cdf')
    cdf8 = pycdf.CDF('data/wi_h1_wav_20200708_v01.cdf')
    cdf9 = pycdf.CDF('data/wi_h1_wav_20200709_v01.cdf')
    cdf10 = pycdf.CDF('data/wi_h1_wav_20200710_v01.cdf')
    cdf11 = pycdf.CDF('data/wi_h1_wav_20200711_v01.cdf')
    cdf12 = pycdf.CDF('data/wi_h1_wav_20200712_v01.cdf')
    cdf13 = pycdf.CDF('data/wi_h1_wav_20200713_v01.cdf')
    cdf_list = [cdf1, cdf2, cdf3, cdf4, cdf5, cdf6, cdf7, cdf8, cdf9, cdf10,
                cdf11, cdf12, cdf13]
    return cdf_list


def create_AKR_df():
    """
        Функция для создания датафрейма с размеченными вручную отрезками
        времени, которые соответствуют всплескам АКР
    """
    detected_akr = pd.read_excel('data/WINDacr2020.xlsx')
    AKR_intervals = {}

    upper_bound = detected_akr[["конец", "Unnamed: 8", "Unnamed: 9",
                                "Unnamed: 10", "Unnamed: 11", "Unnamed: 12"
                               ]
                              ][1:]
    timestamp = []
    for j in range(upper_bound.shape[0]):
        i = j + 1
        timestamp.append(f'{upper_bound["конец"][i]}-{upper_bound["Unnamed: 8"][i]}-{upper_bound["Unnamed: 9"][i]} {upper_bound["Unnamed: 10"][i]}:{upper_bound["Unnamed: 11"][i]}:{upper_bound["Unnamed: 12"][i]}')
    upper_bound["timestamp"] = [datetime.datetime.strptime(datestamp, "%Y-%m-%d %H:%M:%S") for datestamp in timestamp]

    lower_bound = detected_akr[["начало", "Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5"]][1:]
    timestamp = []
    for j in range(upper_bound.shape[0]):
        i = j + 1
        timestamp.append(str(lower_bound["начало"][i]) + "-" + str(lower_bound["Unnamed: 1"][i]) + "-" + str(lower_bound["Unnamed: 2"][i]) + " " + str(lower_bound["Unnamed: 3"][i]) + ":" + str(lower_bound["Unnamed: 4"][i]) + ":" + str(lower_bound["Unnamed: 5"][i]))
    lower_bound["timestamp"] = [datetime.datetime.strptime(datestamp, "%Y-%m-%d %H:%M:%S") for datestamp in timestamp]

    AKR_intervals["begin"] = lower_bound["timestamp"]
    AKR_intervals["end"] = upper_bound["timestamp"]
    AKR_intervals = pd.DataFrame(AKR_intervals)
    return AKR_intervals


def get_total_df(cdf_list, AKR_intervals):
    """
        Создание общего датафрейма мультивариативного ряда, где каждый ряд
        соответствует показателям напряжения в единицу времени на конкретной
        частоте
    """
    cdf_df_list = []

    for cdf in cdf_list:
        cdf_dict = {}
        cdf_dict["Epoch"] = cdf["Epoch"]

        for freq_j in range(cdf["Frequency_RAD1"].shape[0]):
            voltage_list = cdf["E_VOLTAGE_RAD1"][:, freq_j]
            cdf_dict[cdf["Frequency_RAD1"][freq_j]] = voltage_list

        cdf_df = pd.DataFrame(cdf_dict)
        cdf_df = cdf_df.set_index('Epoch')

        # создание столбца с АКР
        is_AKR_list = []
        for i in range(cdf_df.shape[0]):
            cur_time = cdf_df.index[i]
            is_in_range = None
            for cur_interval in AKR_intervals.values:
                begin, end = cur_interval
                if begin <= cur_time <= end:
                    is_in_range = True
                    break
            if is_in_range is None:
                is_in_range = False
            is_AKR_list.append(is_in_range)

        cdf_df["is_AKR"] = is_AKR_list
        cdf_df_list.append(cdf_df)

    final_df = cdf_df_list[0]
    for cdf_df_i in range(1, len(cdf_df_list)):
        final_df = final_df.append(cdf_df_list[cdf_df_i])
    return final_df


################################################################################

def split_into_blocks_by_class(df):
    """
        Функция делит временные ряды на блоки, где каждый блок соответствует
        всплеску АКР или его отсутствию. Блоки не равномерные.
    """
    block_number = 0
    block_number_list = [block_number]
    for i in range(1, df.shape[0]):
        cur_value = df["is_AKR"][i]
        prev_value = df["is_AKR"][i - 1]
        if prev_value == cur_value:
            block_number_list.append(block_number)
        else:
            block_number += 1
            block_number_list.append(block_number)
    df["block"] = block_number_list
    return df


################################################################################

def main():
    # step 1
    # получение и первичная предобработка спутниковых данных
    cdf_list = get_cdf_list()
    AKR_df = create_AKR_df()
    df = get_total_df(cdf_list, AKR_df)

    # step 2
    # обработка спутниковых данных для задачи классификации по блокам
    df = split_into_blocks_by_class(df)
    df.to_csv("processed_data.csv")


if __name__ == "__main__":
    main()