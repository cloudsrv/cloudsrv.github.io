---
layout: post
title:  "怎样重训练图片分类器增加新类别"
date:   2018-04-17 10:44 +0800
categories: DeepLearning TensorFlow
---

TensorFlow官方最新的教程[原文](https://www.tensorflow.org/tutorials/image_retraining)翻译。

现在的图片识别模型都有成百万的参数。从头训练这些模型需要很多打好标签的训练用数据和大量的计算能力（几百个小时的GPU计算或者更多）。转移学习是一种技巧，使用一个已经在相关任务上接受过培训的模型，并在这个模型基础上重新训练来实现这一点。本教程将会重用基于ImageNet强大的图片分类器已经提取过的特征，并这个基础上进行简单的训练。更多信息参看[Decaf的论文](https://arxiv.org/abs/1310.1531)。

>注意：本教程也可以使用[codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)。

这个教程使用TensorFlow Hub来汲取已经一段训练好的模型，或者是他们命名为的*模型*。对于初学者我们使用Inception V3架构基于ImageNet训练提取的[图片特征模型](https://www.tensorflow.org/modules/google/imagenet/inception_v3/feature_vector/1)，以及更多的[选择](https://www.tensorflow.org/tutorials/image_retraining#other_architectures)，包括[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)/PNASNet和[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)及V2。

在开始之前需要使用PIP安装程序包``tensorflow-hub``，以及足够新的TensorFlow的版本。详细信息参看TensorFlow Hub的[安装文档](https://www.tensorflow.org/installation)。

## 训练花的分类

![图像分类](/assets/img/tensorflow-retrain/daisies.jpg)

Kelly Sikkema供图

在进行训练之前你需要一组图片来教会神经网络识别新的类别。后面的内容有解释怎样准备你自己的图片，开始你可以使用一些有共享许可的花的图片。获得这组图片使用如下命令：

```bash
cd ~
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

得到这些图片以后你可以从GitHub上下载示例代码（这些代码没有包含在库安装包中）：

```bash
mkdir ~/example_code
cd ~/example_code
curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py
```

在最简单的情况下可以直接运行如下命令进行重新训练（大概需要半个小时）：

```bash
python retrain.py --image_dir ~/flower_photos
```

这个脚本还有很多其他的参数。你可以通过如下命令得到帮助：

```bash
python retrain.py -h
```

这个脚本会加载训练好的模型，并基于你下载花的照片训练新的分类。原始的ImageNet模型没有训练过识别花的种类。转移学习的神奇之处在于，已经被训练过区分一些对象的较低网络层可以被重复用于更多的识别任务，而不需要做任何改变。

## 瓶颈值

这个脚本会运行大概30分钟或者更长时间，依赖于你的机器速度。第一个阶段是分析磁盘上所有图片，计算和缓存每个图片的瓶颈值。“瓶颈”是一个非正式的名称，经常用来指在最终输出分类层之前的分层。（TensorFlow Hub称之为“图片特征向量”）这个倒数第二层经过训练输出一组值，这组值已经能够很好的帮助我们将要求的分类区分的很好了。这就是说这组值必须是对图片有意义和紧凑的概要提取，因为它必须包含足够的信息才能使分类器在很小的一组值中做出正确的选择。我们最后的分层重训练就可以在新类别上使用了，同样区分ImageNet中所有1,000个类别所需的信息通常对区分新类别的对象也很有用。

由于每个图片在训练过程中被多次重复使用，计算每个图片的瓶颈值需要花费大量时间，因此将这些瓶颈值缓存在磁盘上可以加快速度，而无需重复计算。 默认情况下，它们存储在``/tmp/bottleneck``目录中，如果您重新运行脚本，它们将被重用而不必重复计算。

## 训练

当瓶颈值计算完成，神经网络的最上层开始实际上的训练。你可以看到一系列分步的输出，每个输出显示了准确率、验证准确率和交叉熵。准确率显示了当前打标签的训练批正确分类的百分比。验证准确率是随机选择一组其他图片的准确率。主要差异是准确率是基于神经网络可以学习的图片，所以可能会有可能会对训练数据的噪声出现过拟合的情况。而神经网络性能真正评价标准是通过没有包含在训练数据中的图片来验证的——这个标准就是验证准确率。如果训练准确率很高但是验证准确率仍然很低，这就是说神经网络已经过拟合了，从训练图片中记录的特征不具备通用性。交叉熵是损失函数，可以让我们大概了解目前学习的进展情况。学习的目标是让损失尽量的小，所以你可以通过观察这个值是不是不断在降低，从而判断学习是否有效，但是要忽略短期的噪音。

这个脚本默认会训练4,000步。每步从训练集中任意选择10个图片，从缓存中找到瓶颈值，将它们提供给最后一层得到预测。这些预测会与实际标签进行比较从而通过向后传播算法更新最后一层的权重值。这个过程你会看到报告的准确率在不断提升，当所有的步骤完成后，验证准确率会从与训练和验证照片不同的数据集中计算出来。这个评估测试对于评价训练好的模型是最好的方式。你应该看到一个准确率在90%到95%之间的数值，可能每次训练这个实际值会有不同，因为这个训练过程具有随机性。模型完全训练后这个数字是基于所有给出正确标签测试集的百分比。

## 使用TensorBoard可视化重新训练

这个脚本包含了TensorBoard的概要让这个过程更容易理解、调试和持续调优。比如你可以查看计算图和策略，以及在训练过程中权重和准确率是如何变化的。

```bash
tensorboard --logdir /tmp/retrain_logs
```

当TensorBoard开始运行后，使用浏览器登陆``localhost:6006``查看TensorBoard。

``rerain.py``脚本会默认记录TensorBoard概要到``/tmp/retrain_logs``中。可以通过指定参数``--summaries_dir``参数来修改这个目录。

[TensorBoard GitHub仓](https://github.com/tensorflow/tensorboard)有更多关于TensorBoard使用的详细信息，包括提示、各种坑以及调试信息。

## 使用重训练的模型

这个脚本会在你的目录中写入``/tmp/output_graph.pb``新的模型文件和一个文档文件``/tmp/output_labels.txt``包含了标签。新的模型文件中包含了TF-Hub模块和新的分类层。这两个格式的文件都可以使用[C++和Python图片分类器的样例代码](/deeplearning/tensorflow/2018/04/16/tensorflow-image-recognition.html)读取，这样你就可以立刻使用这个模型了。因为需要替换掉最后一层，所以需要在脚本中指定新的名称，例如如果是label_image使用参数``--output_layer=final_result``。

这里有一个用你重训练的计算图来运行label_image的例子。按照惯例所有的TensorFlow Hub模型接受输入的彩色值固定在范围[0, 1]的图片，所以你不需要设置``--input_mean``或者``--input_std``参数。

```bash
curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

你应该能够看到花的标签列表，大多数情况下雏菊会出现在上面（但是每次重训练的模型会有些许的差异）。可以替换掉参数``--image``指定你自己的图片来试一下。

如果你想在自己的Python代码使用重训练的模型，上面``label_image``脚本是个合适的开始。``label_image``目录也包含了C++的代码，可以用作模版用来集成到你自己的应用中。

如果觉得Inception V3对你的应用太大或者太慢，看一下下面其他架构模型部分加快或者瘦身你的神经网络。

## 训练你自己的分类

如果已经能够让脚本在花的例子中运行成功，你就可以让它识别你感兴趣的分类了。理论上你只需要指定一组子目录就可以了，这些子目录的名称按照要识别的分类进行命名，并且这些子目录下面只含有此分类下的图片。如果你将子目录的上层目录指定给参数``--image_dir``，脚本就会像花的分类一样训练它。

下面就是花分类图片的目录结构，你可以按照这个例子来组织自己的分类图片：

![目录结构](/assets/img/tensorflow-retrain/folder_structure.png)

实践中要达到一定的准确率需要一些工作。下面就我们会遇到的问题进行一些讨论。

## 创建一组图片

最先考虑的是你搜集到的图片，因为我们看到最为普遍的问题是训练数据的问题。

如果想训练有好的结果大概需要一百张以上要识别类别的图片。图片越多你的模型识别的准确率越高。你还要确保照片要有代表性，与你的应用在实际运行过程中遇到的情况一样。比如如果你的照片都是在室内空白墙壁前拍摄的，而用户却是使用室外的照片识别对象，你可能在实施过程中就不会好的结果。

另外一个需要避免的是在学习过程中避免使用除识别对象以外的共同特征，否则你可能得到一些没有价值的结果。举个例子，如果你要识别对象的照片都是在蓝色房间中拍摄的，而另外一种是在绿色房间拍摄的，那么模型就会将基于背景颜色进行预测，而不是真正关心的对象特征。为了避免这种情况，尽量收集近可能多情况下、不同的时间和设备拍摄的照片。

同时也需要思考一下你使用的类别。如果小的类别在视觉上更有独特性，那么将大的不同的物理形态分成较小类别可能是有帮助的。例如，可以使用“汽车”，“摩托车”和“卡车”来代替“车辆”。 同样值得考虑一下你使用“封闭世界”还是“开放世界”的问题。在封闭的世界里，你唯一要求分类的东西就是你所了解的对象类别。这可能适用于植物识别应用程序，你知道用户可能正在拍摄一朵花，所以你只需要确定哪些物种。相比之下，漫游机器人可能会通过其摄像头在世界各地漫游时看到各种不同的东西。在这种情况下，你希望分类器报告它是否不确定它看到的是什么。这可能很难做到，通常需要你收集大量没有关联对象的典型“背景”照片，将它们添加到额外“未知”类别的文件夹中。

检查也是很有价值的，以确保所有的图像都标有正确的标签。用户生成的标签对于我们的用途通常不那么可靠。例如：标有#daisy的图片也可能包含名字为Daisy的人和角色。如果你仔细检查你的图片并清除任何错误，它可以为你的整体准确率做出巨大贡献。

## 训练步数

如果你对收集的图片数据有足够的信心，可以通过改变学习过程的细节来改善您的结果。最简单的尝试是``--how_many_training_steps``。 默认值为4,000，但如果将其增加到8,000，则训练时间会延长两倍。提高准确率的速度会随着你训练的时间延长而减缓，并且在某些时候会完全停止（甚至由于过度拟合而下降），但是你可以尝试看看什么模型最适合你。

## 失真

改善图像训练结果的一种常见方式是以随机方式对训练输入的图片进行变形、裁剪或增亮。 这具有扩大训练数据的有效大小的优势，这要归功于相同图像的所有可能的变化，并且有助于神经网络学会应对在分类器的实际使用中将发生的所有失真。在我们的脚本中实现这些扭曲的最大缺点是瓶颈缓存不再有效，因为输入图像永远不会重复使用。这意味着训练过程需要更长的时间（几个小时），所以建议你在得到一个较满意结果后需要打磨模型时再尝试这种方法。

可以通过设置``--random_crop``、``--random_scale``和``--random_brightness``这三个参数来让脚本来启用这些扭曲处理。这些都是控制每个图像应用了多少扭曲的百分比的值。对于每个参数初始使用5或10是比较合适的，然后可以试着设置其他的值看哪些会对你的应用有帮助。``--flip_left_right``将水平随机镜像一半图像，只要这些反转很可能在您的应用程序中发生，这是有意义的。例如，如果你试图识别字母，这就不是一个好主意，因为翻转它们会破坏它们原来的意义。

## 超级参数

也可以尝试调整其他几个参数，来观察是否有助于提升训练的结果。 ``--learning_rate``控制训练过程中最后一层更新放大率的大小。直观地说，如果这个比较小，那么学习需要更长的时间，但最终可能会帮助整体的准确率。但情况并非总是如此，所以需要仔细试验以后找到适合你的情况。 ``--train_batch_size``控制在每个训练步骤中检查多少图片以评估最后一层的更新结果。

## 训练集、验证集和测试集

当你将脚本指向一个图片文件夹时，脚本在后台进行的操作过程中是将它们分成三个不同的集合。最大的通常是训练集，它是训练期间输入到神经网络的所有图片数据，结果用于更新模型的权重。你可能想知道为什么我们不使用所有图像进行训练？当我们进行机器学习时，一个很大的潜在问题是我们的模型可能只是记住了训练图片不相关的一些细节，就给出了正确的结果。例如，可以想象一个神经网络在每张照片的背景中记住了一个模式，并使用它来将标签与对象进行了匹配。这样可以在训练过程中看到在所有图片上都会有良好的效果，但是由于没有学习到对象的一般特征，而只是记住了训练图片的一些不重要的细节，因此新图片识别就会失败。

这个问题被称为过度拟合，为了避免这种情况发生，我们将一些数据保留在训练过程之外，这样模型就不能记住它们。然后，我们使用这些图片作为检查的数据来确保过度拟合不会发生，因为如果我们看到它们有很好的准确率，那么这是一个很好的迹象，表明神经网络没有过度配合。通常的做法是将80％的图像放入主要的训练集中，保留10％作为训练期间的频繁验证，然后最终使用10％作为测试集来预测分类器在实际应用中的表现。这些比率可以使用参数``--testing_percentage``和``--validation_percentage``进行控制。一般来说，可以直接保留默认值，因为结果通常无法通过调整它们获得任何提升。

请注意，该脚本使用图片文件名（而不是完全随机的函数）来划分培训集、验证集和测试集。 这样做是为了确保每次运行时图片不会在训练集和测试集之间来回移动，因为如果用于训练模型的图片随后用于验证集，可能会有问题。

你可能会注意到验证准确率在迭代过程中会有波动。这种波动的很大一部分原因是每个验证准确率的测量会选择验证集合的随机子集。通过设置``--validation_batch_size = -1``，可以大大降低波动性，但需要增加一些训练时间，每次准确率的计算将使用整个验证集。

一旦训练完成，你可能会发现在测试集中检查错误分类的图像是很有帮助。这可以通过增加参数``--print_misclassified_test_images``来实现。这可能帮助了解模型中哪些类型的图像最容易混淆，哪些类别最难区分。例如，你可能会发现某个特定类别的某个子类型或一些不寻常的照片角度特别难以识别，这样就鼓励你添加更多该子类型的训练图片。通常，检查错误分类的图像也可能会检查出输入数据集中的错误，如错误标记、低质量或模糊的图片。但是，通常应避免在测试集中修正个别错误，因为它们可能仅仅反映（更大）训练集中的更一般问题。

## 其他模型架构

默认情况下，脚本使用经过训练的Inception V3架构提取图像特征模型。这是一个很好的开始，因为它为重新训练脚本提供了准确的结果和适当的运行时间。但现在让我们来看看[TensorFlow Hub模型的更多选项](https://www.tensorflow.org/modules/image)。

一方面，列出了更新的、更强大的架构，比如[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)（尤其是``nasnet_large``和``pnasnet_large``），可以让你得到更高的准确率。

另一方面，如果你要把模型部署到移动设备或者其他资源紧张的环境中，可能不需要那么高的准确率而需要小一点的文件或者更快的速度（包括训练）。这样可以尝试不同[模型](https://www.tensorflow.org/modules/image#mobilenet)，这里实现了[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)或者V2架构，或者也可以使用``nasnet_mobile``。

训练其他模型非常容易：只需要将模型的URL作为参数``--tfhub_module``传递给脚本就可以了，比如：

```bash
python retrain.py \
    --image_dir ~/flower_photos \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1
```

这样会创建一个9MB大小的名为``/tmp/output_graph.pb``模型文件，这是一个MobileNet V2基线版本。在浏览器中打开模型的URL你可以看到模型的文档。

如果你只是想让它运行的更快，可以减小输入图片的大小（URL中第二个数字），从“224”减小到“192”、“160”或者“128”像素正方形大小，甚至是“96”（仅适用于V2）。更为激进的节约资源的做法可以选择使用百分比（第一个数字）“100”、“075”、“050”或者“035”（“025”只是针对V1）来控制“特征深度”或者每个位置神经元的数量。权重的数量（以及文件大小和速度）随着该数值的平方收缩。可以分别参看[MobileNet V1的博客](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)和[Mobile V2的GitHub页面](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)的报告来衡量在ImageNet分类中的表现。

MobileNet V2不会在瓶颈层来应用特征深度百分比。MobileNet V1会这样做，由于深度很小而让任务在分类层更难。使用原来的ImageNet的1001分类数值而不是用更为严格的瓶颈层的数值会更有帮助吗？可以简单地尝试在模型名称中将``mobilenet_v1.../feature_vector``替换为``mobilenet_v1.../classification``。

像以前一样，可以将所有重新培训的模型与``label_image.py``一起使用。例如，需要指定模型所需的图像大小：

```bash
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--input_height=224 --input_width=224 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

有关将重培训模型部署到移动设备的更多信息，请参阅本教程的[codelab版本](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)，特别是[第2部分](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)，其中介绍了[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)及其提供的其他优化（包括模型权重的量化）。