---
layout: post
title:  "特征列"
date:   2018-04-20 20:18 +0800
categories: DeepLearning TensorFlow GettingStart
---

TensorFlow官方最新的教程[原文](https://www.tensorflow.org/get_started/feature_columns)翻译。

本文档是关于特征列的详细内容。把**特征列**想象成原始数据和Estimator之间的中间人。特征列非常丰富，可以将范围很广的原始数据转换成Estimator可以使用的数据，且非常容易上手体验。

在[预制的Estimator](/deeplearning/tensorflow/gettingstart/2018/04/19/tensorflow-gettingstart-estimators.html)一文中，我们使用过预制的Estimator [``DNNClassifier``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)来训练一个模型通过四个特征来预测不同的鸢尾花种类。这个例子中只是创建了数字特征列（[``tf.feature_cloumn.numberic_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)）。尽管对萼片和花瓣的长度使用数字特征列非常有效，但现实世界中数据集包含各种各样的特征，很多都不是数字。

![特征标签云](/assets/img/tensorflow-feature-column/feature_cloud.jpg)

一些真实世界中的特征（比如坐标）是数字的，但很多不是。

## 向一个深度神经网络输入

什么样的数据可以在一个深度的神经网络中进行处理？答案当然是数字（比如``tf.float32``）。总之在一个神经网络中的每个神经元都要对权重和输入数据执行乘法和加法操作。但是现实世界中的数据很多都包含非数字（类别的）数据。想一下``product_class``特征可能包含以下三个非数字的值：

* 厨房用品
* 电器用品
* 运动用品

机器学习模型通常使用简单的向量代表代表类别值，这个向量中1表示有这个类别而0表示没有这个类别。例如当``product_class``被设置为运动用品，一个机器学习模型通常应该表示``product_class``为``[0, 0, 1]``，表示：

* ``0``：缺少厨房用品
* ``0``：缺少电器用品
* ``1``：有运动产品

所以尽管原始数据可能是数字的或者是类别，一个机器学习模型都会将所有的特征表示成数字。

## 特征列

如下图所示，你通过传递参数``feature_column``给一个Estimator（鸢尾问题的``DNNClassifier``）来指定模型的输入。特征列在输入数据（通过``input_fn``返回）与模型之间搭起了一个桥梁。

![特征列](/assets/img/tensorflow-feature-column/inputs_to_model_bridge.jpg)

特征列可以桥接原始数据和模型

创建特征列可以调用``tf.feature_column``模块中的函数。本文档将说明其中的九个函数。如下图所示所有的九个函数返回除了``bucketized_clomn``以外，不是一个Categorical-Column（类别列）就是一个Dense-Column（密集列）对象，所有的返回都是继承这两个对象：

![函数返回类](/assets/img/tensorflow-feature-column/some_constructors.jpg)

让我们详细看一下这几个函数。

### 数字型列

鸢尾分类器对所有的输入特征调用[``tf.feature_column.numeric_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)函数：

* ``SepalLength``
* ``SpalWidth``
* ``PetalLength``
* ``PetalWidth``

尽管``tf.numeric_column``提供可选的参数，可以不输入，如下，最好是使用默认的数据类型（``tf.float32``）作为模型的输入：

```python
# Defaults to a tf.float32 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")
```

指定非默认数据类型，使用``dtype``参数。例如：

```python
# Represent a tf.float64 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",
                                                          dtype=tf.float64)
```

默认情况下一个数字列创建一个单独的值（纯量）。使用``shape``参数来指定其他的形状。比如：

```python
# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix",
                                                         shape=[10,5])
```

### Bucketized列

我们通常不希望直接把数字提供给模型，但是将数字按照范围划分成不同的类别。我们可以使用[bucketized列](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column)来创建。举个例子，考虑原始数据表示了一个房子的建造时间。一般不是将年份作为纯量的数字列而是把年份划分到不同的bucket中：

![Bucket](/assets/img/tensorflow-feature-column/bucketized_column.jpg)

把年份数据分割成四个bucket

模型将使用下表表示bucket：

数据范围 | 代表...
--- | ---
< 1960 | [1, 0, 0, 0]
>= 1960 但 < 1980 | [0, 1, 0, 0]
>= 1980 但 < 2000 | [0, 0, 1, 0]
> 2000 | [0, 0, 0, 1]

为什么要把数字——一种完美的模型输入——分割成类别数据？注意这里将单独的输入数字分成四个元素的向量。这样模型就可以学习*四个单独的权重*而不是一个，四个权重会比一个权重创建更丰富的模型。更为重要的是分类使模型更加清晰的区分了不同年份的类别，因为它只将一个元素设置为1，而其他元素都被清除为0。比如当你只用单独的数字（一个年份）作为输入，一个线性模型只能学习一个线性的关系。所以分类提供了模型额外的灵活性以便模型学习。

下面的代码演示了如果创建一个bucketized特征：

```python
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```

注意指定*三*个元素的``boundaries``向量会产生*四*个元素的bucketized向量。

### 类别标识列

**类别标识列**可以被看作为bucketized列的特例。在传统的bucketized列中每个类别代表了一段值的范围（比如1960到1979）。在类别标识列中每个类别代表了一个、单独的整数。比如说你要标识整数范围``[0, 4]``。也就是你想表示整数0、1、2或3。在这种情况下类别标识列映射如下：

![类别标识列](/assets/img/tensorflow-feature-column/categorical_column_with_identity.jpg)

一个类别标识列的映射。注意这个是one-hot编码不是二进制编码。

作为分类列模型可以为每个在类别标识列中的分类分别学习权重。比如替换``product_class``的文本我们使用一个单一的整数来表示每个分类：

* ``0="kitchenware"``
* ``1="electronics"``
* ``2="sport"``

调用[``tf.feature_column.categorical_column_with_identity``](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_identity)实施类别标识列。例如：

```python
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)

# In order for the preceding call to work, the input_fn() must return
# a dictionary containing 'my_feature_b' as a key. Furthermore, the values
# assigned to 'my_feature_b' must belong to the set [0, 4).
def input_fn():
    ...
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
            [Label_values])
```

### 类别词汇列

我们不能直接输入字符串给模型。因此我们必须首先将字符串映射成数字或者类别值。类别词汇列提供了一种将字符串表示成one-hot向量的很好的方式。比如：

![类别词汇列](/assets/img/tensorflow-feature-column/categorical_column_with_vocabulary.jpg)

映射字符串值到词汇列上。

你可以看到类别词汇列是类别标识列的枚举类型的版本。TensorFlow提供了两个不同的函数来创建类别词汇列：

* [``tf.feature_column.categorical_column_with_vocabulary_list``](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list)
* [``tf.feature_column.categorical_column_with_vocabulary_file``](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file)

``categorical_column_with_vocabulary_list``基于一个声明词汇的列表映射每个字符串到整数上。例如：

```python
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature_name_from_input_fn,
        vocabulary_list=["kitchenware", "electronics", "sports"])
```

函数的执行非常的直接，但是也有明显的缺点。当词汇列表很长的时候需要输入太多的内容。本列中调用``tf.feature_column.categorical_column_with_vocabulary_file``可以将词汇放在一个单独的文件中，比如：

```python

# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature to our model by mapping the input to one of
# the elements in the vocabulary file
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_file(
        key=feature_name_from_input_fn,
        vocabulary_file="product_class.txt",
        vocabulary_size=3)
```

``product_class.txt``中应该每个分类元素一行。比如：

```
kitchenware
electronics
sports
```

### 哈希列

到目前为止，我们只是针对原来就很小数量的类别进行处理。例如我们的``product_class``样本只有3个类别。但是我们常遇到的情况是，类别的数量非常大每个单独的类别不可能有名字或者整数作为标识，因为这样做会耗用大量的内存。这样的情况下我们可以转换一下思路，问一个问题“我录入的类别要多少？”，实际上[``tf.feature_column.categorical_column_with_hash_bucket``](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket)可以指定类别的数量。对这种类型的特征列，模型会计算输入的哈希值，并使用模运算把它放入相应的``hash_bucket_size``类别中，伪代码如下：

```python
# pseudocode
feature_id = hash(raw_feature) % hash_buckets_size
```

代码会创建一个``feature_column``会像如下内容：

```python
hashed_feature_column =
    tf.feature_column.categorical_column_with_hash_bucket(
        key = "some_feature",
        hash_buckets_size = 100) # The number of categories
```

到这里你可能会理所当然的想“这太疯狂了！”毕竟我们强制将不同的输入值划分到一个小的类别集中。这意味着可能两个完全没有关系的输入映射到一个类别中，结果是神经网络认为它们都是一样的，下面的图示显示ketchenware和sports都指定给目录12（哈希类别）：

![哈希类别](/assets/img/tensorflow-feature-column/hashed_column.jpg)

表示数据的哈希类别

同很多直觉现象相反，在机器学习实践中哈希通常会有非常好的结果。因为哈希类别为模型提供一些隔离。模型会使用其他额外的特征来区分ketchenware和sports。

### 交叉列

将几个特征组合成一个特征，多称为[交叉特征](https://developers.google.com/machine-learning/glossary/#feature_cross)，让模型能够学习每个特征组合的单独权重。

更具体的说，假设我们想要模型计算Atlanta的房地产价格。在这个城市的房地产价格与位置相关性很高。作为隔离的经度和纬度特征对于房地产位置信息没什么用，但是把经度和纬度交叉起来作为一个单独的特征就可以定位。假设我们把Atlanta分割成100x100的正方形区域，通过经纬度交叉的特征来区分出10,000个地区。这个交叉特征就可以让模型训练价格与每个区域之间的关联关系，这就会比单独的经度和纬度关联性要高很多。

下图显示了我们的计划，就是在右上角显示红色的经纬度值。

![Atlanta](/assets/img/tensorflow-feature-column/Atlanta.jpg)

Atlanta地图。想象这个地图被分割成10,000个相同大小的区域。

解决方案是我们使用过的``bucketized_column``组合，并使用[``tf.feature_column.crossed_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column)函数。

```python
def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape

    features = {'latitude': latitude.flatten(),
                'longitude': longitude.flatten()}
    labels=labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))

# Bucketize the latitude and longitude usig the `edges`
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges))

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(atlanta.longitude.edges))

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

fc = [
    latitude_bucket_fc,
    longitude_bucket_fc,
    crossed_lat_lon_fc]

# Build and train the Estimator.
est = tf.estimator.LinearRegressor(fc, ...)
```

可以使用以下两种方式创建交叉特征：

* 特征名称，就是使用``input_fn``返回的``dict``的名称。
* 任何类别列，除了``categorical_column_with_hash_bucket``（因为``crossed_column``也要哈希输入）

当特征列``latitude_bucket_fc``和``longitude_bucket_fc``做交叉，TensorFlow会为每个样本创建``(latitude_fc, longitude_fc)``对。这将产生完整的网格，如下所示：

```python
 (0,0),  (0,1)...  (0,99)
 (1,0),  (1,1)...  (1,99)
   ...     ...       ...
(99,0), (99,1)...(99, 99)
```
除了完整的网格只适用于有限词汇表的输入。而不是构建这个可能巨大的输入表，``crossed_column``仅构建``hash_bucket_size``参数所请求的数量。特征列通过在输入的tuple上运行哈希函数，然后用``hash_bucket_size``进行模运算，为索引指定一个样本。

如前所述，执行哈希和模运算函数会限制类别的数量，但会导致类别冲突; 也就是说，多个（纬度、经度）交叉特征会在同一个哈希类别中结束。但实际上，执行特征交叉仍然会为模型的学习能力增加显着的价值。

有点违反直觉的是，创建交叉特征时，通常应该在模型中包含原始（未交叉）特征（如前面的代码段中所示）。独立的经度和纬度特征有助于模型区分交叉特征中发生哈希冲突的示例。

## 指示和嵌入列

指示列和嵌入列永远不会直接作为特征输入，而是使用类别列输入。

当使用一个指示列时，我们告诉TensorFlow去做在类别product_class例子一样的事情。就是**指示列**把每个类别作为一个在one-hot向量中的元素，在匹配类别的位置设置为1而其他位置为0：

![指示列](/assets/img/tensorflow-feature-column/categorical_column_with_identity.jpg)

指示列中表示数据

这里演示了如何使用[``tf.feature_column.indicator_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column)创建一个指示列：

```python
categorical_column = ... # Create any type of categorical column.

# Represent the categorical column as an indicator column.
indicator_column = tf.feature_column.indicator_column(categorical_column)
```

现在假设不仅仅有三个可能类别，我们有百万个设置千万个类别。由于几个原因随着类别数量变大，在神经网络中使用指示列不太可能了。

我们可以使用内嵌列来解决这个限制。一个**内嵌列**不是使用one-hot向量多维度来表示数据，而是使用低维度、普通的向量来表示数据，向量中每个值可以包含任何数字，不仅仅是0和1。通过每个值中允许更多的数字，一个内嵌列包含了比一个指示列更少的值。

让我们看一个例子来比较指示列和内嵌列。假设我们的输入样本由有限的81个单词组成。更进一步假设数据集提供了以下输入词4个单独的样本：

* ``"dog"``
* ``"spoon"``
* ``"scissors"``
* ``"guitar"``

在这个例子中，下图演示了处理内嵌或者指示列的处理路径。

![指示列VS内嵌列](/assets/img/tensorflow-feature-column/embedding_vs_indicator.jpg)

与一个指示列相比一个内嵌列会将类别数据存放在低维度的向量中。（我们只是在内嵌的向量中放入了随机数；训练会决定实际的数字。）

当样本处理时，一个``categorical_column_with...``函数会将样本的字符串与数字类别值进行映射。例如，一个函数映射“spoon”到``[32]``。（32是我们的想象——实际值要依赖于映射函数。）然后你可以使用这些数字类别用以下两种表示方式：

* 作为一个指示列。一个函数会将每个类别值转换成一个81个元素的向量（因为我们有81个词），分别在类别索引值为``(0, 32, 79, 80)``的位置放置1，而其他所有的位置都放0。
* 作为一个内嵌列。一个函数使用数字类别值``(0, 32, 79, 80)``作为查找表的索引。每个查找表中的记录都包含3个元素的向量。

嵌入向量中的值如何神奇地被赋值？实际上，这些任务在训练期间发生。也就是说，该模型学习了将输入的数字分类值映射到嵌入向量值，来找到解决问题的最佳方法。嵌入列会增加模型的能力，因为嵌入向量将从训练数据中学习类别之间的新关系。

为什么我们的例子中的嵌入向量的元素大小为3？下面的“公式”提供了关于嵌入维数的一般规则：

```python
embedding_dimensions =  number_of_categories**0.25
```

也就是说，嵌入向量维度应该是类别数量的第4个根。由于我们在这个例子中的词汇数量是81，建议的维数是3：

```python
3 =  81**0.25
```

注意这只是一个基本指引，你可以设置内嵌维度的数量。

调用[``tf.featur_column.embedding_column``](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column)来创建一个``embedding_column``，可以使用以下的代码片段：

```python
categorical_column = ... # Create any categorical column

# Represent the categorical column as an embedding column.
# This means creating a one-hot vector with one element for each category.
embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=dimension_of_embedding_vector)
```

[内嵌](https://www.tensorflow.org/programmers_guide/embedding)是机器学习一个非常重要的话题。这里的信息仅仅作为一个开始，让你把它作为特征列使用。

## 传递特征列给Estimator

根据以下列表，不是所有的Estimator都允许所有类型的``feature_columns``参数：

* [``LinearClassifier``](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier)和[``LinearRegressor``](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor)：接受所有类型的特征列。
* [``DNNClassifier``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)和[``DNNRegressor``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor)：只接受密集列。其他所有列的类型都必须封装在一个``indicator_column``或者``embedding_column``。
* [``DNNLinearCombinedClassifier``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)和[``DNNLinearCombinedRegressor``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedRegressor)：
	* ``linear_feature_columns``参数接受任何类型的特征列。
	* ``dnn_feature_columns``参数只接受密集列。

## 其他资源

更多关于特征列的例子参看以下内容：

* 使用[低层级API介绍](https://www.tensorflow.org/programmers_guide/low_level_intro#feature_columns)演示了TensorFlow低层级的API怎样使用``feature_columns``。
* [wide](https://www.tensorflow.org/tutorials/wide)和[Wide & Deep](https://www.tensorflow.org/tutorials/wide_and_deep)两个教程解决一个二分类问题在不同输入数据类上使用``feature_columns``。