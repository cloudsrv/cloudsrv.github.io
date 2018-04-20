---
layout: post
title:  "针对机器学习初学者的预制Estimator"
date:   2018-04-19 15:22 +0800
categories: DeepLearning TensorFlow GettingStart
---

TensorFlow官方最新的教程[原文](https://www.tensorflow.org/get_started/get_started_for_beginners)翻译。

本文档阐述了如何使用机器学习来对鸢尾属植物的种类进行分类（分组）。本文深入TensorFlow的代码来了解它做了什么，并解释一些机器学习的基础知识。

如果符合以下情况，那么本文档就比较适合你：

* 你对机器学习完全不了解或了解较少。
* 你想学习如何编写TensorFlow的代码。
* 你可以用Python编程（至少有一些了解）。

如果你对机器学习很精通但是刚刚接触TensorFlow，可以阅读[开始使用TensorFlow：针对机器学习专家](https://www.tensorflow.org/get_started/premade_estimators)这篇文档。

## 鸢尾属植物分类问题

把自己想象成一个植物学家，你想找到一种方法来自动分辨出每个遇到鸢尾花的种类。机器学习提供了很多分类花的方法。比如一个复杂的机器学习程序可以基于花的照片来给它分类。我们的目标要求没有那么高——只想是想通过鸢尾属花的[萼片](https://en.wikipedia.org/wiki/Sepal)和[花瓣](https://en.wikipedia.org/wiki/Petal)的长宽来进行分类。

鸢尾属植物大概有300个种类，但是我们的程序只会对以下三种进行分类：

* 山鸢尾
* 维吉尼亚鸢尾
* 变色鸢尾

![鸢尾植物](/assets/img/tensorflow-estimators/iris_three_species.jpg)

从左至右依次为[山鸢尾](https://commons.wikimedia.org/w/index.php?curid=170298)（来源于BY-SA 3.0 Radomil）、[维吉尼亚鸢尾](https://commons.wikimedia.org/w/index.php?curid=248095)（源于BY-SA 3.0 Radomil）和[变色鸢尾](https://www.flickr.com/photos/33397993@N05/3352169862)（来源于BY-SA 2.0 Radomil）

幸运的是一些人已经创建了一个包含萼片和花瓣数据的[120鸢尾花的数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)。这个数据集已经成为机器学习分类问题的经典入门题。（[MNIST数据集](/deeplearning/tensorflow/2018/04/15/tensorflow-images-mnist.html)是包含手写数字的另外一个热门分类问题。）以下有5个鸢尾数据集的记录：

萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 种类
--- | --- | --- | --- | ---
6.4 | 2.8 | 5.6 | 2.2 | 2
5.0 | 2.3 | 3.3 | 1.0 | 1
4.9 | 2.5 | 4.5 | 1.7 | 2
4.9 | 2.5 | 4.5 | 1.7 | 0
5.7 | 3.8 | 1.7 | 0.3 | 0

让我们介绍一些术语：

* 最后一列（种类）称为[**标签**](https://developers.google.com/machine-learning/glossary/#label)；前面四个列称为[**特征**](https://developers.google.com/machine-learning/glossary/#feature)。特征是样本的特点，而标签是我们要进行预测的事物。
* 一个[**样本**](https://developers.google.com/machine-learning/glossary/#example)包含了一组特征和花种类的标签。前面的表格显示了120个样本中的5个。

每个标签本来是一些字符串（比如“山鸢尾”），但是机器学习一般要依靠数字值的计算。因此一些人就将字符串映射成数字。下面就是与数字的对应关系：

* 0 代表山鸢尾
* 1 代表维吉尼亚鸢尾
* 2 代表变色鸢尾

## 模型和训练

一个**模型**就是特征与标签之前的关系。对于鸢尾分类问题，模型定义了萼片和花瓣度量值与预测鸢尾种类之间的关系。一些简单的模型通过几行代数公式就能描述清楚，但是复杂机器学习模型有大量的参数很难描述清楚。

可以*不*使用机器学习来决定四个特征与鸢尾种类的关系吗？就是说可以使用传统的编程技术（比如大量的条件判断）来构建一个模型吗？可能。你需要处理很长的数据来决定正确的萼片花瓣度量与特定种类之间的关系。但是好的机器学习方法会*为你挑选模型*。也就是说如果你提供了足够有代表性的样本给了机器学习类型的模型，程序会决定萼片、花瓣和种类之间的关系。

**训练**是机器学习一个阶段，在这个阶段中模型会逐步优化（学习）。鸢尾种类问题是一个[监督学习](https://developers.google.com/machine-learning/glossary/#supervised_machine_learning)的例子，会通过包含标签的样本中进行训练。([无监督学习](https://developers.google.com/machine-learning/glossary/#unsupervised_machine_learning)中样本没有标签，模型通常是发现特征中的模式。)

## 获得事例程序

在获得本文档事例代码之前，进行以下操作：

1. 安装TensorFlow。
2. 如果使用virutalenv或者Anaconda安装的TensorFlow，激活你TensorFlow环境。
3. 安装或者升级pandas使用如下命令：``pip install pandas``

使用如下步骤获得事例程序：

1. 从github上克隆TensorFlow仓，输入以下命令：``git clone https://github.com/tensorflow/models``
2. 改变目录到本文档使用的代码：``cd models/samples/core/get_started/``

在``get_started``目录下你会找到``premade_estimator.py``程序。

## 运行事例程序

运行TensorFlow程序就像运行其他的Python程序一样。因此可以运行如下命令：

```bash
python premade_estimator.py
```

运行程序会有输出，最后结束时会有三行预测值，如下：

```
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

如果程序没有输出预测而产生了一些错误，来看一下是否有以下问题：

* 正确安装了TensorFlow吗？
* 使用了正确的TensorFlow版本吗？程序``premade_estimators.py``需要至少1.4以上的版本。
* 如果使用virtualenv或者Anaconda安装TensorFlow，激活环境了吗？

## TensorFlow程序堆栈

如下图所示TensorFlow提供一个包含多层API的程序堆栈：

![TensorFlow堆栈](/assets/img/tensorflow-estimators/tensorflow_programming_environment.png)

**TensorFlow程序环境**

如果你刚开始编写TensorFlow程序，我们强烈建议你关注以下两个高层级的API：

* Estimator
* Dataset

使用其他的API也很方便，但本文档将关注这两个API。

## 程序本身

感谢你的耐心，让我们深入讲解一下代码。``premade_estimator.py``大体的架构——其他很多TensorFlow程序也类似：

* 导入并解析数据集
* 构建特征列描述数据
* 选择模型的类型
* 训练模型
* 评估模型的有效性
* 让训练好的模型作出预测

以下部分将详细展开。

### 导入和解析数据集

鸢尾分类程序需要的数据来源于两个.csv文件：

* ``http://download.tensorflow.org/data/iris_training.csv``，包含了训练集。
* ``http://download.tensorflow.org/data/iris_test.csv``，包含了测试集。

**训练集**包含了我们要训练模型用的样本；测试数据集包含了我们用来评估训练好模型性能的样本。

训练集和测试集本来是一个数据集。有人将它分开，一大部分用于训练集一小部分用于测试集。增加更多的样本一般有助于构建更好的模型，但是增加更多的样本到测试集能更好的衡量模型的有效性。不管是否已经被分开，测试集和训练集的样本必须分开。否则你可能无法验证模型的有效性。

``premade_estimators.py``程序依赖相邻``iris_data.py``文件中``load_data``函数读取和解析训练集和测试集。这里有一个更多注释的版本：

```python
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)
```

Keras是一个开源的机器学习库，``tf.keras``是TensorFlow实现的Keras。``premade_estimator.py``程序只使用了一个``tf.keras``函数，名为``tf.keras.utils.get_file``工具函数，用来复制远程CSV文件到本地文件系统上。

调用``load_data``返回两个（``feature``、``label``）对儿，分别用于训练和测试集：

```python
    # Call load_data() to parse the CSV file.
    (train_feature, train_label), (test_feature, test_label) = load_data()
```

Pandas是使用了几个TensorFlow函数的开源Python库。一个pandas的[DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)是一个命名列名称和几个行的表。特征被``load_data``函数返回并打包在``DataFrams``中。例如``test_feature`` DataFrame看起来如下：

```
    SepalLength  SepalWidth  PetalLength  PetalWidth
0           5.9         3.0          4.2         1.5
1           6.9         3.1          5.4         2.1
2           5.1         3.3          1.7         0.5
...
27          6.7         3.1          4.7         1.5
28          6.7         3.3          5.7         2.5
29          6.4         2.9          4.3         1.3
```

### 数据的描述

**特征列**是一种数据结构，它是用来描述模型怎样如何解释每个特征中的数据。在鸢尾问题中我们想要模型每个特征的文字直接解释为浮点值，也就是说我们想要模型把输入的值比如5.4解释成数字5.4。但是其他机器学习问题通常很少这样直接使用特征的值，都需要做一定的转换。使用特征列来解释数据是一个很大的话题，需要继续深入的阅读[这个文档](https://www.tensorflow.org/get_started/feature_columns)来了解。

从代码的角度看，你通过[``tf.feature_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column)构建了一个``feature_column``对象列表。每个对象描述了一个模型的输入。并告诉模型将数据解释为浮点值，调用[``tf.feature_column.numeric_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)。在``premade_esimator.py``中，所有特征应该直接解释为浮点值，所以使用如下代码来创建一个特征列：

```python
# Create feature columns for all features.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

这里有另外一种写法，虽然不是很优雅但是会更清晰一些：

```python
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]
```

### 选择模型的类型

我们需要选择要训练模型的种类。有很多模型的类型，使用哪一种理想类型需要经验。我们选择神经网络来解决鸢尾问题。**[神经网络](https://developers.google.com/machine-learning/glossary/#neural_network)**可以发现特征和标签之间复杂的关系。一个神经网络是高度结构化的计算图，通过一个或多个[隐藏层](https://developers.google.com/machine-learning/glossary/#hidden_layer)组织起来的。每个隐藏层包含一个或着多个[神经元](https://developers.google.com/machine-learning/glossary/#neuron)。有几种类型的神经网络，我们这里要使用[全连接神经网络](https://developers.google.com/machine-learning/glossary/#fully_connected_layer)，这就意味着在一层的神经元会将所有前一层*每个*神经元的输出作为输入。举个例子，下图演示了一个全连接神经网络由三个隐藏层组成：

* 第一个隐藏层包含四个神经元
* 第二个隐藏层包含三个神经元
* 第三个隐藏层包含两个神经元

![神经网络](/assets/img/tensorflow-estimators/simple_dnn.svg)

**一个含有三个隐藏层的神经网络**

在实例化**[Estimator](https://developers.google.com/machine-learning/glossary/#Estimators)**类的时候指定模型的类型。TensorFlow提供了两种类别的Estimator：

* **[预制的Estimator](https://developers.google.com/machine-learning/glossary/#pre-made_Estimator)**，已经有人替你编写了代码。
* **[客户化Estimator](https://developers.google.com/machine-learning/glossary/#custom_estimator)**，你需要至少部分你编写代码。

实施一个神经网络，``premade_estimators.py``程序使用了一个预制的Estimator，名为``tf.estimator.DNNClassifier``。这个Estimator构建一个对样本分类的神经网络。以下调用实例化``DNNClassifer``：

```python
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```

使用``hidden_unites``参数定义在每个隐藏层的神经元的数量。参数使用一个列表，比如：

```python
        hidden_units=[10, 10],
```

指定列表的长度要等于隐藏层的数量（这里是2）。列表中每个数字代表了特定隐藏层神经元的数量（第一个隐藏层和第二个都是是10）。改变隐藏层或者神经元的数量，只要指定不同的列表值给参数``hidden_unites``即可。

理想的层数和神经元数量依赖于你的问题和数据集。与其他机器学习的方面类似，选择理想型号的神经网络需要知识和经验的积累。一个大体的指引是，增加隐藏层和神经元的数量*通常*会构建一个更强大的模型，需要对更多的数据进行有效的训练。

``n_classes``参数是指定神经网络可以预测的可能值的数量。因为鸢尾问题需要分出3种鸢尾种类，我们设置``n_classes``为3。

构造器``tf.Estimator.DNNClassifier``使用了另外一个可选参数``optimizer``，我们的样例代码没有设置。[**optimizer**](https://developers.google.com/machine-learning/glossary/#optimizer)控制了模型如何训练。当你更为专业的开发机器学习代码时，优化器和[学习率](https://developers.google.com/machine-learning/glossary/#learning_rate)会变得非常重要。

### 训练模型

``tf.Estimator.DNNClassifier``实例化构建了一个学习模型的框架。基本上说我们只是连接了一个网络，还没有让数据在其中流动起来。训练神经网络，调用Estimator对象的``train``方法。例如：

```python
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```

``steps``参数告诉``train``在特定数量迭代后停止训练。增加``steps``就会延长模型的训练时间，另外更长时间训练并不一定保证会得到更好的模型。``args.train_steps``默认值时1000。训练的步数是你可以优化的[超级参数](https://developers.google.com/machine-learning/glossary/#hyperparameter)。选择合适的步数通常需要经验和试验来验证。

``input_fn``参数指定了提供训练数据的函数。这个调用``train``方法指定了``train_input_fn``函数会提供训练数据。下面是这个函数的定义签名：

```python
def train_input_fn(features, labels, batch_size):
```

我们传递以下参数给``train_input_fn``：

* ``train_feature``是一个Python词典类型，其中：
	* 每个键以特征的名称命名。
	* 每个值是一个数组包含了在训练集中每个样本的值。
* ``train_label``是一个数组包含了训练集中每个样本的标签数据。
* ``args.batch_size``是一个整型值定义了[批次数量](https://developers.google.com/machine-learning/glossary/#batch_size)。

``train_input_fn``函数依赖于**Dataset API**。这是个高层级的TensorFlow API来读取数据并转换成``train``方法需要的形式。下面调用将输入的特征和标签转换为``tf.data.Dataset``对象，就是Dataset API的基础类：

```python
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
```

``tf.dataset``类提供了很多有用函数来准备训练用的样本。下面的代码调用了三个这样的函数：

```python
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
```

训练网络最好的方式是将训练样本按随机方式的输入。随机样本的输入需要调用``tf.data.Dataset.shuffle``。设置``buffer_size``成一个比样本数量（120）更大的值确保数据可以很好的洗牌。

在训练时，``train``方法通常处理样本很多次，调用``tf.data.Dataset.repeat``方法，不需要任何参数确保``train``方法无限制（没有再经过洗牌）的提供训练集的样本。

``train``方法每次处理一个样本[**批次**](https://developers.google.com/machine-learning/glossary/#batch)。``tf.data.Dataset.batch``方法创建一个由多个样本连接起来的批次。这些程序使用默认的**[批次数量](https://developers.google.com/machine-learning/glossary/#batch_size)**是100，意味着``batch``方法将100个样本连接成一组。理想的批次数量依赖于要解决的问题。一个大体的指引是更小的批次数量会让``train``方法训练模型更快但是要牺牲（某些时候）准确率。

下面``return``声明将一个批次的样本返回给调用（``train``方法）。

```python
   return dataset.make_one_shot_iterator().get_next()
```

### 评估模型

**评估**意思是查明模型作出预测的有效性。查明鸢尾分类模型的有效性，传递一些萼片和花瓣的度量值给模型，让模型作出预测鸢尾植物代表的种类。然后比较模型预测与实际标签。例如一个模型挑出了一半输入样本的正确种类，那么**[准确率](https://developers.google.com/machine-learning/glossary/#accuracy)**就是0.5。下面的模型就会更有效一些：

<table>
   <tr>
      <td colspan=“6”>测试集</td>
   </tr>
   <tr>
      <td colspan=“4”>特征</td>
      <td>标签</td>
      <td>预测</td>
   </tr>
   <tr>
      <td>5.9</td>
      <td>3</td>
      <td>4.3</td>
      <td>1.5</td>
      <td>1</td>
      <td>1</td>
   </tr>
   <tr>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.4</td>
      <td>2.1</td>
      <td>2</td>
      <td>2</td>
   </tr>
   <tr>
      <td>5.1</td>
      <td>3.3</td>
      <td>1.7</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
   </tr>
   <tr>
      <td>6</td>
      <td>3.4</td>
      <td>4.5</td>
      <td>1.6</td>
      <td>1</td>
      <td  bgcolor="red">2</td>
   </tr>
   <tr>
      <td>5.5</td>
      <td>2.5</td>
      <td>4</td>
      <td>1.3</td>
      <td>1</td>
      <td>1</td>
   </tr>
</table>

**一个模型预测的准确率为80%**

验证一个模型的有效性，每个Estimator提供了一个``evaluate``方法。``premade_estimator.py``程序调用``evaluate``方法如下：

```python
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

调用``classifier.evaluate``和调用``classifier.train``类似。最大的不同是``classifier.evaluate``必须是使用验证集的数据而不是训练集的数据。换句话说，为了验证模型有效性的公平，*验*证模型的样本必须与*训练*用的样本是不同的。``eval_input_fn``函数会从验证集中获取批次样本。以下是``eval_input_fn``函数：

```python
def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

大体上说``eval_input_fn``在调用``classifier.evaluate``时机型以下操作：

1. 将测试集的特征和标签转换成``tf.dataset``对象。
2. 为测试创建一个样本批次（这里不需要对测试集样本进行洗牌或者重复调用）。
3. 返回测试样本批次给``classifier.evaluate``。

运行这些代码会得到以下输出（或者近似的）：

```
Test set accuracy: 0.967
```

0.967的准确率是指我们训练过的模型会正确预测测试集中30个鸢尾种类中的29个。

### 预测

现在我们已经训练了一个模型并“证明”它可以但不是非常完美的对鸢尾种类进行分类。现在让我们使用训练过的模型来做一些**[没有标签样本](https://developers.google.com/machine-learning/glossary/#unlabeled_example)**的预测，也就是这些样本只有特征没有标签。

在现实世界中没有标签的样本可能会来源于各种途径，包括APP、CSV文件和数据输入。今天我们只是简单的用手工方式提供以下三个没有标签的样本：

```python
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
```

每个Estimator提供了一个``predict``方法，``premade_estimator.py``如下调用：

```python
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x,
                                  labels=None,
                                  batch_size=args.batch_size))
```

和``evaluate``方法一样我们的``predict``方法也从``eval_input_fn``方法中收集样本。

当我们进行预测的时候是*不会*传递标签给``eval_input_fn``方法的。因此，``eval_input_fn``会做如下操作：

1. 对特征值3个元素进行转换。
2. 从手工样本中创建一个3个样本的批次。
3. 给``classifier.predict``返回这些样本批次。

``predict``方法返回了一个python的迭代器，为每个样本产生了一个字典类型的预测值。这个字典包含几个键。``probabilities``键中保存了一个三个浮点值组成的列表，每个代表了输入样本针对不同鸢尾种类的可能性。比如下面的``probabilities``列表：

```python
'probabilities': array([  1.19127117e-08,   3.97069454e-02,   9.60292995e-01])
```

上面的列表指出：

* 可以忽略不计这是一个山鸢尾
* 3.97%可能是一个变色鸢尾
* 96.0%可能是一个弗吉尼亚鸢尾

一个``class_ids``键包含了一个元素的数组指出了最可能的种类，比如：

```python
'class_ids': array([2])
```

数字2对应了弗吉尼亚鸢尾。以下代码迭代返回``predictions``来报告每个预测：

```python
for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(iris_data.SPECIES[class_id], 100 * probability, expec))
```

运行程序产生如下输出：

```
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

## 总结

本文档提供了一个机器学习的简短介绍。

由于``premade_estimators.py``依赖高层级的API，机器学习复杂的数学原理被隐藏了。如果你想要对机器学习更为了解，我们建议还是要学习**[梯度递减](https://developers.google.com/machine-learning/glossary/#gradient_descent)**、批次和神经网络的概念。

我们建议下一步阅读[特征列](https://www.tensorflow.org/get_started/feature_columns)这篇文档，解析了在机器学习中如何使用它来表示不同的数据。