import pandas as pd
import numpy as np
import pickle


def adjust_blocks_to_length(df, target_length, step=1):
    """
    Приводит блоки в датафрейме к указанной длине. Блоки с длиной меньше удаляются,
    а блоки с длиной больше формируются с помощью скользящего окна с заданным шагом.
    
    Параметры:
    - df (pd.DataFrame): Исходный датафрейм с блоками.
    - target_length (int): Желаемая длина блоков.
    - step (int): Шаг скользящего окна. По умолчанию 1.

    Возвращает:
    - pd.DataFrame: Новый датафрейм с блоками указанной длины.
    """
    # Список для хранения новых блоков
    new_blocks = []
    new_block_id = 0
    
    # Группируем по столбцу 'block'
    grouped = df.groupby('block')
    
    for block_id, group in grouped:
        # Если длина блока меньше целевой, пропускаем его
        if len(group) < target_length:
            continue
        
        # Если длина блока равна целевой, добавляем его как есть
        elif len(group) == target_length:
            group['block'] = new_block_id
            new_blocks.append(group)
            new_block_id += 1
        
        # Если длина блока больше целевой, применяем скользящее окно
        else:
            for start in range(0, len(group) - target_length + 1, step):
                # Формируем новое окно
                new_block = group.iloc[start:start + target_length].copy()
                new_block['block'] = new_block_id  # Присваиваем новый ID блока
                new_blocks.append(new_block)
                new_block_id += 1

    # Объединяем все блоки обратно в один DataFrame
    if new_blocks:
        new_df = pd.concat(new_blocks)
    else:
        new_df = pd.DataFrame(columns=df.columns)  # Пустой DataFrame, если нет подходящих блоков

    return new_df


def cleared_test_ds(X_dict, y, threshold=1):
    """
        Функция, которая удаляет из тренировочной выборки последовательности, в
        которых значение напряжения равно нулю больше чем threshold секунд 
    """
    new_X_dict = {}
    bad_idx = []
    # находим для каждой частоты индексы с данными, в которых больше одного 0
    freqs = list(X_dict.keys())
    for freq in freqs:
        cur_freq_test = X_dict[freq]
        for i in range(len(cur_freq_test)):
            if cur_freq_test[i].count(0) > threshold:
                bad_idx.append(i)
    bad_idx = np.unique(np.array(bad_idx)).tolist()
    new_y = pd.DataFrame(y).drop(bad_idx, axis=0).values.tolist()
    new_y = [y[0] for y in new_y]
    # удаляем для каждой частоты объекты с этими индексами (в т.ч. в 'y')
    for freq in freqs:
        cur_freq_test = X_dict[freq]
        new_X = pd.DataFrame(cur_freq_test).drop(bad_idx, axis=0).values.tolist()
        new_X_dict[freq] = new_X
    return new_X_dict, new_y


def balance_blocks_by_is_AKR(df):
    """
    Сбалансировать количество блоков по значению столбца `is_AKR`.
    Уравнивает количество блоков с `is_AKR = True` и `is_AKR = False` путем рандомного удаления избыточных блоков.
    
    Параметры:
    - df (pd.DataFrame): Датафрейм, содержащий столбцы `block` и `is_AKR`.
    
    Возвращает:
    - pd.DataFrame: Сбалансированный датафрейм с равным количеством блоков `is_AKR = True` и `is_AKR = False`.
    """
    # Разделение на группы по значению `is_AKR`
    true_blocks = df[df["is_AKR"] == True]
    false_blocks = df[df["is_AKR"] == False]

    # Определение минимального количества блоков
    min_count = min(len(true_blocks), len(false_blocks))

    # Если количество блоков уже сбалансировано, ничего не делаем
    if len(true_blocks) == len(false_blocks):
        return df

    # Если блоков с is_AKR=True больше или меньше, делаем undersampling
    true_blocks_balanced = true_blocks.sample(n=min_count, random_state=42)
    false_blocks_balanced = false_blocks.sample(n=min_count, random_state=42)

    # Объединяем сбалансированные блоки
    balanced_df = pd.concat([true_blocks_balanced, false_blocks_balanced]).reset_index(drop=True)

    return balanced_df


def main():
    df = pd.read_csv("processed_data.csv", index_col=0)
    ethalon_block_sizes = [4, 16, 32]
    for bs in ethalon_block_sizes:
        df_bs = adjust_blocks_to_length(df, bs)
        df_bs.to_csv(f"windowed_df_{bs}.csv")
        freqs = df_bs.columns[:-2]

        dfs_for_freqs = []
        for cur_freq in freqs:
            kekw = df_bs[[cur_freq, "is_AKR", "block"]]

            grouped = kekw.groupby("block").apply(lambda x: pd.Series({
                cur_freq: x[cur_freq].tolist(),
                "is_AKR": x["is_AKR"].iloc[0],
                "block": x["block"].iloc[0]
            })).reset_index(drop=True)

            grouped[cur_freq] = grouped[cur_freq].apply(lambda x: x)
            dfs_for_freqs.append(grouped.copy())
        
        balanced_dfs_for_freqs = [balance_blocks_by_is_AKR(df) for df in dfs_for_freqs]
        with open(f'balanced_dfs_for_freqs_{bs}.pkl', 'wb') as file:
            pickle.dump(balanced_dfs_for_freqs, file)


if __name__ == "__main__":
    main()