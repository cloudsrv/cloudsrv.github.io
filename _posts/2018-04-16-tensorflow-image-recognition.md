---
layout: post
title:  "图片识别"
date:   2018-04-16 12:38 +0800
categories: DeepLearning TensorFlow
---

TensorFlow官方最新的教程[原文](https://www.tensorflow.org/tutorials/image_recognition)翻译。

我们的大脑让视觉看起来很容易。 它不需要任何努力让人类分辨狮子和美洲虎，阅读标志或识别人类的脸部。 但实际上这些问题如果用计算机解决却非常困难，虽然它们看起来很容易，这都是因为我们的大脑非常善于理解图像。

在过去的几年中，机器学习领域在解决这些难题方面取得了巨大的进步。 特别是我们发现一种称为深[卷积神经网络](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)的模型可以在合理的性能基础上实现艰难的视觉识别任务——在某些领域达到甚至超过人类的表现。

研究人员的ImageNet工作验证了他们在计算机视觉领域方面的稳步进展——[ImageNet](http://www.image-net.org/)是计算机视觉的学术基准。持续的模型改进凸显出来，每次都能达到更高的发展水平：[QuocNet](https://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf)、[AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)、[Inception（GoogLeNet）](https://arxiv.org/abs/1409.4842)、[BN-Inception-v2](https://arxiv.org/abs/1502.03167)。 Google内部和外部的研究人员发表了大量描述这些模型的论文，但是结果仍然难以重现。我们现在正在更进一步，发布了最新模型[Inception-v3](https://arxiv.org/abs/1512.00567)代码来运行图像识别。

Inception-v3使用2012年的数据对[ImageNet](http://image-net.org/)大型视觉识别挑战进行了训练。这是计算机视觉的一项标准任务，模型尝试将整个图像分为[1000个类别](http://image-net.org/challenges/LSVRC/2014/browse-synsets)，如“Zebra”、“Dalmatian”和“Dishwasher”。 例如，以下是[AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)对一些图像进行分类的结果：

![图像分类](/assets/img/tensorflow-image/AlexClassification.png)

为了比较模型，我们检查了模型未能预测正确答案的频率，排名前5的猜测作为他们标准——被称为“前5错误率”。[AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)通过在2012年验证数据集上达到了15.3％的前5错误率；[Inception（GoogLeNet）](https://arxiv.org/abs/1409.4842)达到6.67％；[BN-Inception-v2](https://arxiv.org/abs/1502.03167)实现4.9％；[Inception-v3](https://arxiv.org/abs/1512.00567)达到3.46％。

>人类在ImageNet挑战中会表现如何？这里有个Andrej Karpathy的[博客](https://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)来衡量他自己的表现。他达到了5.1%前5的错误率。

本教程将会指引你如何使用[Inception-v3](https://arxiv.org/abs/1512.00567)。可以学习如何使用Python或C++中区分图片1000个类别。我们还将讨论如何从这个模型中提取更高级别的特征，这些特征可能会被其他视觉任务重用。

我们很高兴看到社区将使用这个模型能够做出什么。

## Python API的使用

当程序``classify_image.py``第一次运行，将从``tensorflow.org``下载和训练模型。你需要大概200M的剩余硬盘空间。

通过克隆Github的[TensorFlow模型的repo](https://github.com/tensorflow/models)。运行以下命令：

```bash
cd models/tutorials/image/imagenet
python classify_image.py
```

以上命令将会对提供的熊猫图片进行分类。

![熊猫图片](/assets/img/tensorflow-image/cropped_panda.jpg)

如果模型运行正确，脚本会产生如下输出：

```
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.88493)
indri, indris, Indri indri, Indri brevicaudatus (score = 0.00878)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00317)
custard apple (score = 0.00149)
earthstar (score = 0.00127)
```

如果你希望提供其他的JPEG图片，可以通过修改参数``--image_file``。

>如果你要下载模型数据到不同目录，需要通过指定``--model_dir``来修改使用的目录。

## C++ API的使用

你可以在产品环境使用C++运行[Inception-v3](https://arxiv.org/abs/1512.00567)模型。可以下载包含GraphDef的归档，像这样定义模型（从TensorFlow存储库的根目录运行）：

```bash
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C tensorflow/examples/label_image/data -xz
```

接着我们需要编译C++二进制文件，包含了装载和运行图片的代码。如果在特定平台下你遵循[TensorFlow下载和安装源代码的指引](https://www.tensorflow.org/install/install_sources)，可以从shell终端运行此命令来构建示例：

```bash
bazel build tensorflow/examples/label_image/...
```

这样就创建了一个可运行的二进制文件，可以像如下运行：

```bash
bazel-bin/tensorflow/examples/label_image/label_image
```

使用框架将使用默认的样本图片，输出的结果如下：

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```

在这个例子里我们使用默认的图片[Admiral Grace Hopper](https://en.wikipedia.org/wiki/Grace_Hopper)，你可以看到模型正确的识别出来了她穿着了军装，给出了0.8的高分。

![Admiral Grace Hopper](/assets/img/tensorflow-image/grace_hopper.jpg)

接着你可以通过指定参数``--image=my_image.png``使用自己的图片，比如：

```bash
bazel-bin/tensorflow/examples/label_image/label_image --image=my_image.png
```

可以查看[``tensorflow/examples/label_image/main.cc``](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc)文件详细内容来了解它是怎样工作的。我们希望这些代码能够帮助你将TensorFlow集成到自己的应用中，接下来我们会一步一步来看main函数：

命令行标志控制了从哪里加载文件以及输入图像的属性。 该模型预计将获得正方形299x299 RGB图像，所以这里有``input_width``和``input_height``标志。 我们还需要将像素值从0到255之间的整数缩放到图形操作的浮点值。我们用``input_mean``和``input_std``标志来控制缩放：我们首先从每个像素值中减去``input_mean``，然后用``input_std``对其进行分割。

你可以看到它们是如何使用[``ReadTensorFromImageFile()``](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L88)函数。

```c++
// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  tensorflow::GraphDefBuilder b;
```

开始创建一个``GraphDefBuilder``，可以作为一个对象运行和加载模型。

```c++
  string input_name = "file_reader";
  string output_name = "normalized";
  tensorflow::Node* file_reader =
      tensorflow::ops::ReadFile(tensorflow::ops::Const(file_name, b.opts()),
                                b.opts().WithName(input_name));
```

然后，我们开始为我们想要加载的小模型创建神经元节点，调整大小并缩放像素值，以获得主模型期望的结果作为其输入。第一个节点我们只是创建了一个``Const``操作，存放了一个包含想要加载图片文件名张量。然后第一个输入到``ReadFile``操作中。你可能注意到所有的创建操作函数最后一个参数都传递了``b.opts()``。这个参数确保节点添加到模型定义中保存了``GraphDefBuilder``。也可以命名``ReadFile``操作使用``WithName()``调用``b.opts()``。这样就可以命名一个节点，这样会十分必要，否则一个自动的名称将会自动指定给操作，这样调试就不会非常容易了。

```c++
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Node* image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = tensorflow::ops::DecodePng(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("png_reader"));
  } else {
    // Assume if it's not a PNG then it must be a JPEG.
    image_reader = tensorflow::ops::DecodeJpeg(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("jpeg_reader"));
  }
  // Now cast the image data to float so we can do normal math on it.
  tensorflow::Node* float_caster = tensorflow::ops::Cast(
      image_reader, tensorflow::DT_FLOAT, b.opts().WithName("float_caster"));
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  tensorflow::Node* dims_expander = tensorflow::ops::ExpandDims(
      float_caster, tensorflow::ops::Const(0, b.opts()), b.opts());
  // Bilinearly resize the image to fit the required dimensions.
  tensorflow::Node* resized = tensorflow::ops::ResizeBilinear(
      dims_expander, tensorflow::ops::Const({input_height, input_width},
                                            b.opts().WithName("size")),
      b.opts());
  // Subtract the mean and divide by the scale.
  tensorflow::ops::Div(
      tensorflow::ops::Sub(
          resized, tensorflow::ops::Const({input_mean}, b.opts()), b.opts()),
      tensorflow::ops::Const({input_std}, b.opts()),
      b.opts().WithName(output_name));
```

我们继续加入更多的神经元节点，对一个图片进行解码，将整数转化为一个浮点值，缩放它，并且最后对像素值运行减法和除法操作。

```c++
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph));
```

最后我们在``b``变量中存放了模型的定义，可以使用``ToGraphDef()``函数转换为全的图定义。

```c++
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
```

然后我们创建一个``tf.Session``对象，使用它来作为真正操作图片的接口，运行它，指定我们需要得到那个神经元节点的输出以及到哪儿存放输出数据。

这些将给出一个``Tensor``对象的向量，在这里我们知道就是一个单独的长整型对象。在这个场景中可以将``Tensor``想象为多维度的数组，存放了一个299像素高，299像素宽和3个频道的图片的浮点值。如果已经在产品环境中使用了图像处理框架，你可以继续使用它，但是需要在主要图片处理中使用这些数据的时候需要做以上类似的转换。

这是一个在C++中使用TensorFlow动态图片的小例子，但是对于预训练Inception模型我们要从文件加载更多的定义。可以参看在``LoadGraph()``函数中操作。

```c++
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
```

查看加载图片的代码会有很多术语看起来非常熟悉。我们直接加载一个``protobuf``文件包含了``GraphDef``，而不再使用``GraphDefBuilder``产生一个``GraphDef``对象。

```c++
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}
```

我们从``GraphDef``中创建了一个``Session``对象，并将它返回给调用者，这样它们就可以在后面运行这个对象。

``GetTopLables()``函数与图片加载非常像，除了我们想得到主要图片运行的结果，并将其转换成最高分标签的顺序列表。与图片加载相同，它创建了一个``GraphDefBuilder``，添加了一些神经元节点，然后运行这些小图片并得到一对张量输出。这里他们表示排序的分值和最高结果的索引位置。

```c++
// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  tensorflow::GraphDefBuilder b;
  string output_name = "top_k";
  tensorflow::ops::TopK(tensorflow::ops::Const(outputs[0], b.opts()),
                        how_many_labels, b.opts().WithName(output_name));
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
```

``PrintTopLabels()``函数将使用这些排序结果，使用友好的模式打印出来。``CheckTopLabel()``函数也非常类似，但只是为了调试确认给定的标签是否是最高值。

最后[``main()``](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L252)将所有的部分结合起来调用。

```c++
int main(int argc, char* argv[]) {
  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  Status s = tensorflow::ParseCommandLineFlags(&argc, argv);
  if (!s.ok()) {
    LOG(ERROR) << "Error parsing command line flags: " << s.ToString();
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(FLAGS_root_dir, FLAGS_graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
```

加载主要图片。

```c++
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(FLAGS_root_dir, FLAGS_image);
  Status read_tensor_status = ReadTensorFromImageFile(
      image_path, FLAGS_input_height, FLAGS_input_width, FLAGS_input_mean,
      FLAGS_input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];
```
加载、缩放和处理输入图片。

```c++
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({ {FLAGS_input_layer, resized_tensor}},
                                   {FLAGS_output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
```

这里我们运行加载图片的主图作为输入。

```c++
  // This is for automated testing to make sure we get the expected result with
  // the default settings. We know that label 866 (military uniform) should be
  // the top label for the Admiral Hopper image.
  if (FLAGS_self_test) {
    bool expected_matches;
    Status check_status = CheckTopLabel(outputs, 866, &expected_matches);
    if (!check_status.ok()) {
      LOG(ERROR) << "Running check failed: " << check_status;
      return -1;
    }
    if (!expected_matches) {
      LOG(ERROR) << "Self-test failed!";
      return -1;
    }
  }
```

最后打印发现的标签。

```c++
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
```

这里的异常处理使用了TensorFlow的``Status``对象，这样非常方便，因为它让你通过检查``ok()``标签检查是否有错误产生，并且打印一个合理的错误信息。

这里我们演示了对象识别，但是你可以在其他模型中使用类似的代码，用来在跨越各种领域中发现和识别。我们希望这个小例子给你在自己的产品中使用TensorFlow有一些启发。

## 更多的一些学习资源

学习一般的神经网络知识，Michael Nielsen的[免费在线书](http://neuralnetworksanddeeplearning.com/chap1.html)是一个非常棒的资源。对于特定的卷积神经网络，Chris Olah有一些[非常好的博客](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)，以及Michael Nielsen的的书也有[很多章节](http://neuralnetworksanddeeplearning.com/chap6.html)也覆盖了很多内容。

更多关于实施卷机神经网络，可以跳到TensorFlow的[深入卷机神经网络教程](https://www.tensorflow.org/tutorials/deep_cnn)，或者更为简约的教程[MNIST开始教程](/deeplearning/tensorflow/2018/04/15/tensorflow-images-mnist.html)。最后如果你想要加速在这个领域的研究速度，也可以阅读本教程引用的所有最新论文。


