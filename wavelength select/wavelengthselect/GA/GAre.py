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




if __name__ == '__main__':
    main()
