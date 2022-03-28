import matplotlib.pyplot as plt
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
from genetic_selection import GeneticSelectionCV


# data = load_iris()
# X, y = data["data"], data["target"]
#
# # Add random non-important features
# noise = np.random.uniform(0, 10, size=(X.shape[0], 5))
# X = np.hstack((X, noise))
def main():
    data_path = '../../data/corn/m5.csv'  # 数据
    label_path = '../../data/corn/label.csv'  # 标签

    data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    label = np.loadtxt(open(label_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)  # 用于建模

    data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    X = data[:, :]
    label = np.loadtxt(open(label_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)  # 用于建模

    y = np.array(label[:, 1]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    estimator = SVR(gamma='auto')  # 回归函数
    selector = GeneticSelectionCV(estimator,
                                  cv=20,
                                  verbose=1,
                                  scoring="neg_mean_squared_error",
                                  max_features=10,
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.1,
                                  n_generations=200,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)

    # Train and select the features
    selector.fit(X_train, y_train.ravel())
    print(selector.support_.shape)
    for i in range(len(selector.support_)):
        if (selector.support_[i] == True):
            print(i * 2 + 1100)


    # 绘图
    plt.figure(500)
    # x_col = np.linspace(0,len(data[0,:]),len(data[0,:]))
    x_col = np.linspace(1100, 2498, len(data[0, :]))  # 返回介于start和stop之间均匀间隔的num个数据的一维矩阵
    Ga = [1112, 1120, 1136, 1140, 1270, 1272, 1288, 1292, 1314, 1338, 1376, 2300, 2302, 2304, 2306, 2312, 2418, 2462,
          2474]
    Ga = np.array(Ga)
    y_col = np.transpose(data[0, :])  # 数组逆序
    plt.plot(x_col, y_col)

    for i in range(len(Ga)):
        Ga[i] = (Ga[i] - 1100) / 2

    plt.scatter(x_col[Ga], y_col[Ga], marker='s', color='r')
    plt.legend(['First calibration object', 'Selected variables'])
    plt.xlabel('Variable index')
    plt.xlabel("Wavenumber(nm)")
    plt.ylabel("Absorbance")
    plt.title("Genetic feature selecting ", fontweight="semibold", fontsize='x-large')
    plt.legend(['First calibration object', 'Selected variables'])
    plt.xlabel('Variable index')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
