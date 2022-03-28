# 光谱波长选择（特征选取）---遗传算法

将要解决的问题模拟成一个生物进化的过程，通过复制、交叉、突变等操作产生下一代的解，并逐步淘汰掉适应度函数值低的解，增加适应度函数值高的解。这样进化N代后就很有可能会进化出适应度函数值很高的个体。
数据来源
数据来源https://eigenvector.com/resources/data-sets/

该数据集包含在 3 个不同的 NIR 光谱仪上测量的 80 个玉米样品，波长范围为 1100-2498nm
1.导入数据

    data_path = '../../data/corn/m5.csv'  # 数据
    label_path = '../../data/corn/label.csv'  # 标签

    data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    X = data[:, :]
    label = np.loadtxt(open(label_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)  # 用于建模
    y = np.array(label[:, 1]).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



2.建立回归函数(遗传算法所需的适应度函数)

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



3.绘图




    plt.figure(500)
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

![在这里插入图片描述](https://img-blog.csdnimg.cn/4e4aa8875224424593d7f9d0920326a7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASHVtcGhyZXktQXBwbGVieQ==,size_17,color_FFFFFF,t_70,g_se,x_16)




