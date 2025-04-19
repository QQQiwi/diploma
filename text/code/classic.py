from pyts.classification import TimeSeriesForest
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


windowed_df = pd.read_csv("windowed_df.csv", index_col=0)
freqs = windowed_df.columns[:-2]


def cleared_train_ds(X, y, threshold=1):
    """
        Функция, которая удаляет из обучающей выборки последовательности, в
        которых значение напряжения равно нулю больше чем threshold секунд 
    """
    cleared_X = []
    cleared_y = []
    for i in range(len(X)):
        if not (X[i].count(0) > threshold):
            cleared_X.append(X[i])
            cleared_y.append(y[i])
    return cleared_X, cleared_y


def split_data_and_train_ensembles_catboost(dfs, threshold=1):
    # Используем первый DataFrame для инициализации
    cur_df = dfs[0]
    X = cur_df["20"]
    y = cur_df["is_AKR"].apply(int).to_list()

    indexed_y = list(zip(y, np.arange(len(y)).tolist()))
    _, _, y_train, y_test = train_test_split(X, indexed_y, test_size=0.2, random_state=42)
    i_train = [i for _, i in y_train]
    i_test = [i for _, i in y_test]
    y_train = [y for y, _ in y_train]
    y_test = [y for y, _ in y_test]

    freq_clf_list = {}
    X_test_df = {}

    # Параметры для поиска
    param_grid = {
        'iterations': [1, 2, 3, 4, 5],
        'learning_rate': [0.01, 0.1, 1],
        'depth': [1, 2, 3, 4, 5]
    }

    for i in range(len(dfs)):
        cur_df = dfs[i]
        freq = cur_df.columns[0]  # Получаем частоту
        freq_X = cur_df[freq]
        df_X = pd.DataFrame(np.array(freq_X))
        
        # Формирование обучающей и тестовой выборки
        train_freq_X = df_X.iloc[i_train]
        train_freq_X = [train_freq_X.iloc[j].tolist() for j in range(train_freq_X.shape[0])]

        test_freq_X = df_X.iloc[i_test]
        test_freq_X = [test_freq_X.iloc[j].tolist() for j in range(test_freq_X.shape[0])]

        X_test_df[freq] = test_freq_X

        # Очистка данных
        train_freq_X, freq_y_train = cleared_train_ds(train_freq_X, y_train, threshold=threshold)
        
        # Преобразование данных в строки
        train_freq_X = [[str(cur_num) for cur_num in cur_list] for cur_list in train_freq_X]

        # Определение категориальных признаков в зависимости от размера данных
        if len(train_freq_X[0]) > 1:
            cat_features = list(range(len(train_freq_X[0])))  # Индексы всех столбцов
        else:
            cat_features = [0]  # Единственный столбец

        model = CatBoostClassifier(cat_features=cat_features)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(train_freq_X, freq_y_train)
        best_model = grid_search.best_estimator_
        freq_clf_list[freq] = best_model

    return X_test_df, freq_clf_list, y_train, y_test


def split_data_and_train_ensembles_classic(dfs, threshold=1, algo="forest"):
    # Используем первый DataFrame для инициализации
    cur_df = dfs[0]
    X = cur_df["20"]
    y = cur_df["is_AKR"].apply(int).to_list()

    indexed_y = list(zip(y, np.arange(len(y)).tolist()))
    _, _, y_train, y_test = train_test_split(X, indexed_y, test_size=0.2, random_state=42)
    i_train = [i for _, i in y_train]
    i_test = [i for _, i in y_test]
    y_train = [y for y, _ in y_train]
    y_test = [y for y, _ in y_test]

    freq_clf_list = {}
    X_test_df = {}

    for i in range(len(dfs)):
        cur_df = dfs[i]
        freq = cur_df.columns[0]  # Получаем частоту
        freq_X = cur_df[freq]
        df_X = pd.DataFrame(np.array(freq_X))
        
        # Формирование обучающей и тестовой выборки
        train_freq_X = df_X.iloc[i_train]
        train_freq_X = [train_freq_X.iloc[j].tolist() for j in range(train_freq_X.shape[0])]

        test_freq_X = df_X.iloc[i_test]
        test_freq_X = [test_freq_X.iloc[j].tolist() for j in range(test_freq_X.shape[0])]

        X_test_df[freq] = test_freq_X
        train_freq_X = [underlist[0] for underlist in train_freq_X]
        train_freq_X, freq_y_train = cleared_train_ds(train_freq_X, y_train, threshold=threshold)
        if algo == "forest":
            cur_clf = TimeSeriesForest(500, random_state=43)
        else:
            cur_clf = KNeighborsClassifier()

        cur_clf.fit(train_freq_X, freq_y_train)
        freq_clf_list[freq] = cur_clf

    return X_test_df, freq_clf_list, y_train, y_test


def eval_test(X_test_df, forest_freq_clf_list, is_catboost=True):
    X_test_df = pd.DataFrame(X_test_df)
    y_pred = []
    for i in range(X_test_df.shape[0]):
        y_cur_pred = []
        for freq in freqs:
            cur_freq_X = X_test_df.iloc[i][freq]
            if is_catboost:
                cur_freq_X = [str(cur_elem) for cur_elem in cur_freq_X]
            cur_freq_y = forest_freq_clf_list[freq].predict([cur_freq_X])[0]
            y_cur_pred.append(cur_freq_y)
        pred_mode = stats.mode(np.array(y_cur_pred))[0]
        y_pred.append(pred_mode)
    return y_pred


def get_data_stats(train, test):
    print("(train) АКР:", train.count(1))
    print("(train) не АКР:", train.count(0))

    print("(test) АКР:", test.count(1))
    print("(test) не АКР:", test.count(0))


def build_conf_matrix(labels, predict, class_name):
    lab, pred = [], []
    for i in range(len(labels)):
        if predict[i] == class_name:
            pred.append(0)
        else:
            pred.append(1)
        if labels[i] == class_name:
            lab.append(0)
        else:
            lab.append(1)
    return confusion_matrix(lab, pred, normalize='true')


def eval_metrics(y_pred, y_test, id=""):
    print("f1_score: ", f1_score(y_test, y_pred))
    print("precision_score: ", precision_score(y_test, y_pred))
    print("recall_score: ", recall_score(y_test, y_pred))

    get_res = {0: "нет АКР", 1: "АКР"}

    for i in range(2):
        heatmap = sns.heatmap(build_conf_matrix(y_test, y_pred, i), annot=True, cmap='YlGnBu')
        heatmap.set_title(get_res[i], fontdict={'fontsize':14}, pad=10)
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.savefig(f"Класс {i} ({id}).png", dpi=300)
        plt.show()


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


def make_results(X_test_df, freq_clf_list, y_train, y_test):
    get_data_stats(y_train, y_test)
    new_X_test_df, new_y_test = cleared_test_ds(X_test_df, y_test)
    y_pred = eval_test(new_X_test_df, freq_clf_list)
    print(confusion_matrix(new_y_test, y_pred, normalize='true'))
    eval_metrics(y_pred, new_y_test)


def main():
    ethalon_block_sizes = [4, 16, 32]
    for bs in ethalon_block_sizes:
        with open(f'balanced_dfs_for_freqs_{bs}.pkl', 'rb') as file:
            balanced_dfs_for_freqs = pickle.load(file)
        
        print("CatBoost:")
        X_test_df, freq_clf_list, y_train, y_test = split_data_and_train_ensembles_catboost(balanced_dfs_for_freqs)
        make_results(X_test_df, freq_clf_list, y_train, y_test)

        print("RandomForest:")
        X_test_df, freq_clf_list, y_train, y_test = split_data_and_train_ensembles_classic(balanced_dfs_for_freqs, algo="forest")
        make_results(X_test_df, freq_clf_list, y_train, y_test)

        print("kNN:")
        X_test_df, freq_clf_list, y_train, y_test = split_data_and_train_ensembles_classic(balanced_dfs_for_freqs, algo="knn")
        make_results(X_test_df, freq_clf_list, y_train, y_test) 
        

if __name__ == "__main__":
    main()