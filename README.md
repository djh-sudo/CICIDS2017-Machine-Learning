# CICIDS2017-MechineLearning
Multi-classification based CICIDS2017 with Machine Learning

## Introduction

[`CICIDS`数据集](https://www.unb.ca/cic/datasets/ids-2017.html)，分析使用了其中`CICIDS-2017`部分，都是`.csv`文件，数据集处理可以参考`Karggle`。实验效果如表所示。

| `Network` | `precision` | `time(sec)` |
| :-------: | :---------: | :---------: |
|   `RF`    |   `0.997`   |    `72`     |
|   `KNN`   |   `0.997`   |  `1163.18`  |
|   `SVM`   |   `0.373`   |  `1841.78`  |
|   `MLP`   |   `0.985`   |  `1438.70`  |
|   `NB`    |   `0.716`   |   `2.25`    |
|   `CNN`   |   `0.995`   |  `903.71`   |

## Environment

```
OS:windows 10/11
IDE:pyCharm
python-lib:
Keras == 2.2.5
tensorflow == 1.15.0
scikit-learn == 0.19.1
```

## Document

细节参见`doc`文件夹。

## Using

将下载好的数据集放入`input`文件夹，在`main()`中依次运行相应的函数即可。

## Question

1.`SVM`和`Naive Bayes`的`fit()`函数的第二个参数无需进行`one-hot`编码。

2.`one-hot`编码与一般`vector`之间的转换

`one-hot`转一般向量 ：`X = [np.where(r == 1)[0][0] for r in X]`

一般向量转`one-hot`：`X -> y`

```
dummies = pd.get_dummies(X)
y = dummies.values
```

