---
layout: post
title:  "Oracle Compute云架构即代码"
date:   2017-07-05 08:14:41 +0800
categories: Oracle Cloud
---
# ——使用Terraform插件管理Oracle Compute云资源
“架构即代码（Infrastructure as Code简写IaC）是一种通过机器能够读取的配置文件方式来完成数据中心计算能力的管理和初始化的过程，这个过程替代了传统的物理硬件或者交互式工具的配置。裸金属服务器、虚拟机以及与其相关的配置资源都被统称为“基础架构”。定义的配置可以通过版本控制来进行管理，这些配置文件可以是脚本或者声明式的定义，而不是通过手工的方式完成配置，这个定义通常更为倾向于描述声明的方法。云计算是架构即代码的最强有力的推手，特别是IaaS服务更为热衷。“—— [WIKIPEDIA](https://en.wikipedia.org/wiki/Infrastructure_as_Code)
HashiCorp [Terraform](https://terraform.io/)是一个开源的云基础架构和资源的管理和编排的工具，它可以处理一些包含了基础架构及其变动的配置文件，而应用这些文件到相应的环境中就会创建和更新相应云中的资源。Terraform是基础架构即代码的典型代表，可以重复使用并与DevOps实践中的CI／CD自动化流程紧密的结合在一起。
2017年4月26日，Oracle宣布Terraform从0.9.4版本开始原生支持Oracle Compute云插件，具体内容请参看[这里](https://blogs.oracle.com/developers/announcing-built-in-terraform-provider-for-oracle-compute-cloud)。并且Oracle裸金属云也可以使用Terraform的社区[插件](http://blogs.oracle.com/developers/terraform-and-oracle-bare-metal-cloud-services)来进行管理。
Terraform的一个好处是，可以通过原生的插件管理在混合架构下的基础设施和服务，例如可以同时管理AWS云、Azure云和Oracle Compute云等，甚至是在自己机房内的私有云。在Terraform 0.9.4之前的版本Oracle Compute云的插件是通过外部社区的方式提供，而在0.9.4版本之后就合并到Terraform的主干版本中原生支持了。接下来本文将会简单介绍一下如何使用Terraform OPC的插件来管理Oracle Compute云。

## 安装Terraform

从[官网下载](https://www.terraform.io/downloads.html)安装文件，根据自己的平台选择安装。安装过程比较简单，需要强调的是安装完成后检查一下安装的版本：

```bash
$ terraform version
Terraform v0.9.4
```

这个版本号一定要高于上面例子中出现的0.9.4以上才行。安装完成后我们就可以直接使用Oracle Compute云的插件，而不需要像社区支持的插件那样再去编译设置才能使用了。

另外Terraform是有一定学习曲线的，配置文件的编写有自己的语法和规范，可以首先阅读一下它的[Getting Start](https://www.terraform.io/intro/index.html)文档来了解一下。

## Terraform Oracle Compute配置文件

首先为Terraform的配置文件创建一个新的目录，或者如果要使用版本管理可以使用Git等工具初始化一个工作目录。我们所有的配置文件将创建在这个目录下，并以.tf为后缀名，我们这次只是一个startup文档，因此我们只是创建一个名为main.tf文件即可。

```bash
$ mkdir startup && cd startup
$ touch main.tf
```

接下里你可以使用自己喜欢的文本编辑器来编辑我们刚刚创建的main.tf这个文件，推荐使用sumbline有相应的Terraform插件来高亮显示关键字。

我们的配置文件main.tf的最开始信息应该包含了OPC访问的各种配置，包括域、用户名、密码和访问的Endpoint。

```
provider "opc" {
  identity_domain = "mydomain"
  endpoint        = "https://api-z27.compute.us6.oraclecloud.com/"
  user            = "user.name@example.com"
  password        = "Pa$$w0rd"
}
```

这四个信息每个Oracle Compute云帐户是不同的，请根据自己的情况填写不要直接复制粘贴。其中endpoint需要在自己的Oracle Compute控制台上寻找，这个endpoint一定要准确，实际上就是在选择数据中心，填错一般会报没有权限的错误，如果你的帐号有多个数据中心的权限有可能会把你的资源初始化到别的数据中心。

接下来我们可以在main.tf配置文件中添加一个资源

```
resource "opc_compute_instance" "instance1" {
  name = "example-instance1"
  shape = "oc3"
  image_list = "/oracle/public/OL_7.2_UEKR3_x86_64"
}
```
这一段就是在配置文件里声明定义一个Compute虚拟机实例，三个属性分别对应了实例的名称、机型和操作系统映像。当然我们知道一个Compute实例只有这三个属性虽然可以启动，但是无法访问，还有一些网络和安装证书之类的辅助配置。下面我就将完整的main.tf贴出来：

```
provider "opc" {
  identity_domain = "mydomain"
  endpoint        = "https://api-z27.compute.us6.oraclecloud.com/"
  user            = "user.name@example.com"
  password        = "Pa$$w0rd"
}
 
resource "opc_compute_ssh_key" "sshkey1" {
  name = "example-sshkey1"
  key = "${file("~/.ssh/id_rsa.pub")}"
}
 
resource "opc_compute_instance" "instance1" {
  name = "example-instance1"
  shape = "oc3"
  image_list = "/oracle/public/OL_7.2_UEKR3_x86_64"
  ssh_keys = [ "${opc_compute_ssh_key.sshkey1.name}" ]
  networking_info {
    index = 0
    shared_network = true
    nat = ["${opc_compute_ip_reservation.ipreservation1.name}"]
  }
}
 
resource "opc_compute_ip_reservation" "ipreservation1" {
  name = "example-ipreservation1"
  parent_pool = "/oracle/public/ippool"
  permanent = true
}
 
output "public_ip" {
  value = "${opc_compute_ip_reservation.ipreservation1.ip}"
}
```

具体的Oracle Compute云插件资源的这些属性的定义，大家可以到Terraform官网上去[查询](https://www.terraform.io/docs/providers/opc/index.html)，其中配置文件中各个资源定义的顺序及依赖关系完全依靠Terraform的机制就可以保证，编写时按照语法要求编写即可不用考虑。配置文件另外一些需要强调的是：

* 资源之间互相引用是通过Terraform的[插入语法](https://www.terraform.io/docs/configuration/interpolation.html) *${}* 方式实现的，这种方法可以让资源之间互相引用资源内部的属性。例如在资源opc_compute_instance中属性ssh_keys中就通过这种方式把资源sshkey1的名称引用过来。
* 在opc_compute_ssh_key资源的属性使用了Terraform[插入函数](https://www.terraform.io/docs/configuration/interpolation.html#built-in-functions) *${file()}* 来实现，这个函数会读取本地的一个文件，这里是将本地读取的密钥对的公钥字符串返回给SSH key这个属性。
* 在networking_info块中定义了实例的网络属性，需要指出的是这里定义的是Compute的Shared Network，并且引用了Shared Network的一个保留公网IP。
* 最后的output模块将最后在模版apply后返回一些信息，这里我们返回了公网的IP地址。

## 部署Terraform配置文件

接下来我们就可以使用刚刚创建好的模版来启动Oracle Compute的资源了。首先我们在前面创建好的目录下使用Terraform的plan命令来检查整个模版是否正确。

```bash
$ terraform plan
+ opc_compute_instance.instance1
...
+ opc_compute_ip_reservation.ipreservation1
...
+ opc_compute_ssh_key.sshkey1
...
```

Terraform将会输出资源的详细信息，其中 + 号代表增加的资源，~ 号代表修改的资源，删除的资源会使用 - 号表示。在这个例子中我们将看到三个增加的资源，在确保输出正常的情况下我们就可以部署这个配置文件了。

```bash
$ terraform apply
opc_compute_ip_reservation.ipreservation1: Creating...
...
Apply complete! Resources: 3 added, 0 changed, 0 destroyed.
 
Outputs:
public_ip = 129.144.xx.126
```

大概几分钟的时间，Oracle Compute将实例启动起来，我们可以通过ssh登录到新建的实例中

```bash
 ssh opc@129.144.xx.126 -i ~/.ssh/id_rsa
```

这里需要注意的是我们在模版中没有指定security list，这时Oracle Compute会自动使用默认的Default Security List，因此你必须确保在Default Security List中已经允许了SSH端口连接，这样才能保证SSH登录成功。

如果你的需求发生了变化可以直接修改这个配置文件，并使用版本管理系统实现配置文件的版本化，通过terraform plan/apply就可以将变化应用到Oracle Compute云的资源中，这时Oracle Compute只会改动那些修改过的资源，没有修改配置的资源就不会有任何变化。

最后完成了我们整个Getting Start的练习，我们可以把所有资源都删除掉

```bash
$ terraform destroy
...
Destroy complete! Resources: 3 destroyed.
```
