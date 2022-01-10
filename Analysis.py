import itertools
import time
from keras import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as sm

label = ['BENIGN', 'Bot', 'DDos', 'GlodenEye', 'Dos Hulk',
         'Slowhttp', 'SSH', 'FTP', 'PortScan', 'slowloris', 'BruteForce', 'XSS']


def HandleData(path):
    list_dir = os.listdir(path)
    fd_data = []
    for it in list_dir:
        data = pd.read_csv(path + '/' + it)
        fd_data.append(data)
    data = pd.concat([fd_data[0], fd_data[1]])
    for it in range(2, len(fd_data)):
        data = pd.concat([data, fd_data[it]])
    data = data.dropna(axis=0, how='any')
    data = data.replace(',,', np.nan, inplace=False)
    data.replace("Infinity", 0, inplace=True)

    data.replace('Infinity', 0.0, inplace=True)
    data.replace('NaN', 0.0, inplace=True)
    n_row, n_col = data.shape
    print('row:', n_row, 'col:', n_col)

    return data


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def Train(data, decomponent=False):
    print(data[' Label'].value_counts())
    x_columns = data.columns.drop(' Label')
    x = data[x_columns].values
    x = normalize(x, axis=0, norm='max')
    if decomponent:
        pca = PCA(n_components=20)
        x = pca.fit_transform(x)
    dummies = pd.get_dummies(data[' Label'])
    outcomes = dummies.columns
    print(outcomes)
    num_classes = len(outcomes)
    print('[traffic] 类别数:', num_classes)
    y = dummies.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)

    return x_train, y_train, x_test, y_test


def RF(train_X, train_Y, test_X, test_Y):
    print('[RF] train...')
    t1 = time.time()
    rfc = RandomForestClassifier()
    rfc.fit(train_X, train_Y)
    Y_pred = rfc.predict(test_X)
    acc = accuracy_score(test_Y, Y_pred)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), Y_pred.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y, Y_pred)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'RF Confusion matrix')


def KNN(train_X, train_Y, test_X, test_Y):
    print('[KNN] train...')
    t1 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    model = knn.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'KNN Confusion matrix')


def SVM(train_X, train_Y, test_X, test_Y):
    print('[SVM] train ...')
    train_Y = [np.where(r == 1)[0][0] for r in train_Y]
    test_Y = [np.where(r == 1)[0][0] for r in test_Y]
    t1 = time.time()
    clf = svm.SVC(decision_function_shape='ovr', max_iter=300, kernel='rbf')
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y, y_hat)
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'SVM Confusion matrix')


def NaiveBayes(train_X, train_Y, test_X, test_Y):
    print('[Naive Bayes] train ...')
    train_Y = [np.where(r == 1)[0][0] for r in train_Y]
    test_Y = [np.where(r == 1)[0][0] for r in test_Y]
    t1 = time.time()
    clf = BernoulliNB()
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y, y_hat)
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'NB Confusion matrix')


def MLP(train_X, train_Y, test_X, test_Y):
    print('[MLP] train ...')
    t1 = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,),
                          activation='logistic',
                          solver='adam',
                          learning_rate_init=0.0001,
                          max_iter=2000)
    model.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'MLP Confusion matrix')


def DT(train_X, train_Y, test_X, test_Y):
    t1 = time.time()
    clf = DecisionTreeClassifier(max_depth=6)
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'DT Confusion matrix')


def DNN(train_X, train_Y, test_X, test_Y):
    t1 = time.time()
    model = Sequential()
    model.add(Dense(16, input_dim=train_X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(train_Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    model.fit(train_X, train_Y, epochs=40, verbose=2)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'DNN Confusion matrix')


def ImageHandle(train_X, test_X, test_Y):
    path = './temp_img'
    if not os.path.exists(path):
        os.mkdir(path)
    train = []
    test = []
    for vector in train_X:
        vector = np.pad(vector, (0, 100 - len(vector)), 'constant', constant_values=(0, 0))
        vector = np.reshape(vector, (10, 10))
        vector = vector[:, :, np.newaxis]
        train.append(vector)
    train = np.array(train)
    for vector in test_X:
        vector = np.pad(vector, (0, 100 - len(vector)), 'constant', constant_values=(0, 0))
        vector = np.reshape(vector, (10, 10))
        vector = vector[:, :, np.newaxis]
        test.append(vector)

    test = np.array(test)
    return train, test


def CNN(train_X, train_Y, test_X, test_Y):
    print('[CNN] train ...')
    train_X, test_X = ImageHandle(train_X, test_X, test_Y)
    t1 = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(test_Y.shape[1])
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    model.save('model.h5')
    his = model.fit(train_X, train_Y, batch_size=128, verbose=2, epochs=30, validation_split=0.1)
    print(his.history)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(matrix)
    report = classification_report(test_Y.argmax(axis=1), y_hat.argmax(axis=1))
    print(report)
    print('-' * 20)
    plot_confusion_matrix(matrix, label, True, 'CNN Confusion matrix')


def main():
    data = HandleData('./input')
    train_X, train_Y, test_X, test_Y = Train(data)
    # RF(train_X, train_Y, test_X, test_Y)
    # KNN(train_X, train_Y, test_X, test_Y)
    # SVM(train_X, train_Y, test_X, test_Y)
    # MLP(train_X, train_Y, test_X, test_Y)
    # NaiveBayes(train_X, train_Y, test_X, test_Y)
    # DT(train_X, train_Y, test_X, test_Y)
    # DNN(train_X, train_Y, test_X, test_Y)
    # CNN(train_X, train_Y, test_X, test_Y)


if __name__ == '__main__':
    main()
