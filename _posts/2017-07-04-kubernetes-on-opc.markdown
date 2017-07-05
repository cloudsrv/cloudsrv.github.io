---
layout: post
title:  "Oracle Compute云搭建Kubernetes集群"
date:   2017-07-04 08:14:41 +0800
categories: Oracle Cloud
---

## 架构简介
容器是将来大型集群类软件交付的事实标准，而Kubernetes是目前业界最为流行的企业级产品环境就绪的容器编排平台，因此历来复杂企业级软件的交付过程中会有越来越多的用户采用Kubernetes平台。Oracle Compute云提供可预测的性能和网络隔离的基础设施服务也同样被企业级用户广泛采用，用户可以轻松迁移负载并进行扩展，这样就会有大量的需求要在Oracle Compute云上部署Kubernetes集群来运行和交付自己的应用系统，本文详细讲解如何在Compute云上部署Kubernetes并在这个集群上运行一个示例电商应用。

下图是我们这次要在Oracle Compute云上部署的架构
![Kubernetes Architecture](/assets/img/k8s-arch.png)

整个架构采用三个Oracle Compute云主机实例，其中一个节点作为Master节点安装Kubernetes的所有管理节点中的组件同时将Etcd也安装在这个节点上，另外两个作为Kubernetes的Node节点，除了运行Kubernetes节点守护进程kubelet以外所有的受管理的容器都运行在这两个节点上。
我们知道Kubernetes作为容器编排平台组件众多，各个组件的版本迭代也很快，在On Promise（传统企业机房）环境下部署非常复杂，令很多非IT领域的企业级用户望而却步。造成这种状况的原因非常简单，Kubernetes本身就是应“云”而生的，在各种云端部署才是她正确的打开方式，因此企业级用户选用Oracle Compute云来部署Kubernetes才是最佳的选择。在部署过程中我们要借助很多Compute云以及其他Oracle IaaS的服务来帮助我们部署Kubernetes集群。下面我们就详细的分步骤来介绍如何在Oracle Compute云上部署Kubernetes集群。

## 开始部署之前
第一步我们需要有一个Oracle Compute云的帐号，目前我们在中国区可以通过申请免费获得一个测试帐号，这个测试帐号可以免费使用一个月如果需要可以通过申请再延长一个月，具体的申请步骤不是本文的叙事重点，大家可以根据下面的[链接](https://cloud.oracle.com/zh_CN/tryit)来按照要求的步骤一步一步来完成。
测试帐号开通以后Oracle会向你在注册免费试用的邮箱发一封开通邮件，其中包含了登录需要的域名称、用户名和初始化密码，有了这些信息以后我们就可以登录到Oracle Compute云Web端控制台。当然第一次登录系统会要求你修改原始密码，修改完密码后就可以正常登录了，以后就可以使用这个入口[登录](https://cloud.oracle.com/home)。

![OPC Login](/assets/img/k8s-opc-01.png)

这时系统会提示选择数据中心，如果是测试帐户一般要选择US Commercial 2 (us2)并点击My Services，如果是非测试帐户可以参考Oracle发送的服务开通邮件。

![OPC Login](/assets/img/k8s-opc-02.png)

这个界面要输入在申请测试帐户时填写域的名称，点击Go输入用户名密码。

![OPC Login](/assets/img/k8s-opc-03.png)

点击Sign In后我们就进入了My Services的Dashboard，在这里我们可以看到所有在这个域帐号下的各种云服务的状态，也可以通过这个Dashboard进入到各个服务的控制台。这个Dashboard的内容比较丰富可定制的幅度也非常大，我们要寻找的是Compute区域：

![OPC Login](/assets/img/k8s-opc-04.png)

如果没有出现Compute区域我们可以通过Customize Dashboard这个链接将它显示出来。另外大家可以注意一下Dashboard中间区域的几个数字，这些是测试帐号资源的使用情况及限额，如果突破这些限额中的任何一项你的测试帐号的云服务将被暂停。

![OPC Login](/assets/img/k8s-opc-05.png)

点击Compute区域右下方的图标打开操作菜单，并点击Open Service Console链接，打开Compute的控制台。

![OPC Login](/assets/img/k8s-opc-06.png)

进入到Compute控制台后可以看到目前使用的资源情况：

![OPC Login](/assets/img/k8s-opc-07.png)

前面我们已经提到Kubernetes集群部署很复杂，但在云平台上借助一些云的功能将会使部署非常简单，这里我们就将使用Oracle Compute云的Orchestration功能中文称为“编排”，来快速定义Compute云中需要的各种资源并自动部署整个Kubernetes集群。当然这个过程需要一些Shell脚本的支持，同时我们需要Oracle Storage服务来存储这些脚本及集群间共享的密钥，在Orchestration服务需要调用这些脚本时自动下载到Compute云主机中运行。在后面的介绍中我们将主要围绕Orchestration这个服务展开，其最核心的工作是编写Orchestration模版。如果大家对于Orchestration功能不了解建议大家参考一下两个详细的[文档](https://docs.oracle.com/cloud/latest/stcomputecs/STCSG/GUID-874C5C30-C628-4CAF-A8DF-4C8D9CCA9902.htm#STCSG-GUID-874C5C30-C628-4CAF-A8DF-4C8D9CCA9902)以及如何通过Orchestration及Compute云服务的opc-int功能实现Compute云主机自动初始化部署的详细[文档](https://docs.oracle.com/cloud/latest/stcomputecs/STCSG/GUID-C63680F1-1D97-4984-AB02-285B17278CC5.htm#STCSG-GUID-C63680F1-1D97-4984-AB02-285B17278CC5)。

## Compute云安全设置
在说明Compute云编排模版之前，我们需要在Compute云中进行一些与云安全相关的设置，这些安全设置对于保护我们在云中的资源非常重要，无论是测试还是正式环境我都建议大家使用Oracle的最佳实践来进行设置，而不要像在On Promise时代那样简单粗暴的将所有的安全限制全部放开。
首先是使用SSH证书来登录云主机Linux操作系统。在On Promise时代我们大多使用SSH用户名密码的方式登录Linux，这样的方式在云平台中是非常不安全的，我们强烈建议大家使用SSH证书来登录Compute云Linux主机。这个SSH证书需要提前通过自动的工具生成并将公钥上传到Compute云中，在Compute云主机初始化过程中就会将在编排模版中定义好的公钥初始化到相应的操作系统用户中，这样我们就可以通过与之相对应的私钥登录到Linux操作系统了。对于如何生成SSH证书大家可以自行在网络中搜索相应的教程。当你有了SSH证书后我们可以通过Compute控制台将证书的公钥上传到Compute云：

![OPC Security](/assets/img/k8s-opc-08.png)

给SSH证书的公钥起个名称，这个名称非常重要将在后面的编排模版中引用，在这里我们使用orclA，下面在Value一栏中我们可以直接将已经生成的公钥值字符串粘贴过去，或者通过Select File按钮在操作系统中选择一个公钥文件。完成输入后点击Add按钮，系统会提示添加成功。这样我们就完成了SSH证书公钥上传。

![OPC Security](/assets/img/k8s-opc-09.png)

接着我们需要设置Oracle Compute云的网络防火墙，Compute云的SDN的功能也非常强大，分为Shared Network和IP Network两种网络环境，本文为简化和方便测试采用开箱即用的Shared Network网络，产品环境还是建议大家使用IP Network完全从头定义自己的网络环境。因此我们的网络防火墙也是需要在Shared Network网络中进行定义，大家要做好这两种网络环境的区分。Oracle Compute云防火墙是要实现On Promise硬件防火墙的功能，因此它的设置也相对比较灵活和复杂。Oracle Compute云防火墙完整的设置需要： 

* Security IP Lists——IP地址范围设置
* Security Applications——应用端口范围设置
* Security Lists——Compute云主机集和定义（便于云主机重复引用安全规则）
* Security Rules——安全规则定义（将上述对象组合关联形成灵活的安全设置）

在Shared Network中有很多预定义的设置项可以帮助我们快速设置我们自己的防火墙规则，因此本文只是通过先设置两条Security Rules就可以将整个Kubernetes集群安全的保护在Oracle Compute云防火墙中且满足集群对内和对外通讯的需求，本文后面我们还会增加一条规则以便于我们访问集群中部署的电商应用。

![OPC Security](/assets/img/k8s-opc-10.png)

现在我们要设置的两条Security Rules规则详细设置如下：
保证所有集群间通讯的安全规则具体设置，这里的Security Application我们选择了All，表示所有的tcp/udp端口都是允许访问的，源我们选择instance是一个Security IP List中定义的IP地址集合，具体的范围就是登录帐户下Shared Network的私网地址（根据帐号不同CIDR地址会有不同），目标是默认的Security List的Compute云主机集合，这样就限定了只有在Shared Network内部的Compute云主机之间是无限制互相访问的。

![OPC Security](/assets/img/k8s-opc-11.png)

确保能够使用SSH通过互联网访问的安全规则，Security Application我们选择的是SSH协议，具体应该就是开放tcp 22端口，源选择的是Security IP List中定义的public-internet，实际上就是任何地址都能访问，用CIDR地址表示就是0.0.0.0/0，目标仍然是默认的Security List中定义的Compute云主机集合。

![OPC Security](/assets/img/k8s-opc-12.png)

这里再次强调一下不要为了访问方便将Oracle Compute云防火墙的规则全部端口放开，这样的云主机就相当于在互联网上裸奔。

## Oracle Compute编排模版及脚本

Oracle Compute云的编排功能模版有V1和V2两个版本的区别，这两个版本目前都可以使用并互相兼容，本文将使用V1这个版本来编写模版。Orchestration或者云编排功能的作用实际上与Kubernetes容器编排的功能非常类似，是用来管理和调配所有Compute云计算、存储、网络等资源的，我们可以使用云编排功能实现各种自定义功能的部署以及整个虚拟化计算拓扑的生命周期管理。Compute云编排模版是通过流行的JSON格式编写易于大家的理解。
下面我先将Orchestration模版直接全部贴出来。同时为了方便大家按照自己帐户的情况修改模版的内容，我把这个模版的源代码共享出来供大家[下载](https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-orig.json)，后面我们将详细介绍这个模版的细节。

```js
{
    "relationships": [{
        "to_oplan": "k8s_cluster_vol",
        "oplan": "k8s_master_instances",
        "type": "depends"
    }, {
        "to_oplan": "k8s_cluster_vol",
        "oplan": "k8s_node_instances",
        "type": "depends"
    }, {
        "to_oplan": "k8s_master_instances",
        "oplan": "k8s_node_instances",
        "type": "depends"
    }],
    "account": "/Compute-<Identity Domain>/default",
    "name": "/Compute-<Identity Domain>/<OPC Account>/k8s_cluster",
    "oplans": [{
        "obj_type": "storage/volume",
        "label": "k8s_cluster_vol",
        "objects": [{
            "size": "100G",
            "name": "/Compute-<Identity Domain>/<OPC Account>/master_disk_vol",
            "properties": [
                "/oracle/public/storage/default"
            ]
        }, {
            "size": "100G",
            "name": "/Compute-<Identity Domain>/<OPC Account>/node1_disk_vol",
            "properties": [
                "/oracle/public/storage/default"
            ]
        }, {
            "size": "100G",
            "name": "/Compute-<Identity Domain>/<OPC Account>/node2_disk_vol",
            "properties": [
                "/oracle/public/storage/default"
            ]
        }]
    }, {
        "obj_type": "launchplan",
        "ha_policy": "active",
        "label": "k8s_master_instances",
        "objects": [{
            "instances": [{
                "networking": {
                    "eth0": {
                        "seclists": [
                            "/Compute-<Identity Domain>/default/default"
                        ],
                        "nat": "ippool:/oracle/public/ippool",
                        "dns": [
                            "k8s-master"
                        ]
                    }
                },
                "name": "/Compute-<Identity Domain>/<OPC Account>/k8s_master",
                "storage_attachments": [{
                    "volume": "/Compute-<Identity Domain>/<OPC Account>/master_disk_vol",
                    "index": 1
                }],
                "boot_order": [],
                "hostname": "k8s-master",
                "label": "k8s_master",
                "shape": "oc3",
                "imagelist": "/oracle/public/OL_7.2_UEKR4_x86_64",
                "sshkeys": [
                    "/Compute-<Identity Domain>/<OPC Account>/orclA"
                ],
                "attributes": {
                    "userdata": {
                        "pre-bootstrap": {
                            "scriptURL": "https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-init.sh",
                            "failonerror": true
                        }
                    }
                }
            }]
        }]
    }, {
        "obj_type": "launchplan",
        "ha_policy": "active",
        "label": "k8s_node_instances",
        "objects": [{
            "instances": [{
                "networking": {
                    "eth0": {
                        "seclists": [
                            "/Compute-<Identity Domain>/default/default"
                        ],
                        "nat": "ippool:/oracle/public/ippool",
                        "dns": [
                            "k8s-node1"
                        ]
                    }
                },
                "name": "/Compute-<Identity Domain>/<OPC Account>/k8s_node1",
                "storage_attachments": [{
                    "volume": "/Compute-<Identity Domain>/<OPC Account>/node1_disk_vol",
                    "index": 1
                }],
                "boot_order": [],
                "hostname": "k8s-node1",
                "label": "k8s_node1",
                "shape": "oc3",
                "imagelist": "/oracle/public/OL_7.2_UEKR4_x86_64",
                "sshkeys": [
                    "/Compute-<Identity Domain>/<OPC Account>/orclA"
                ],
                "attributes": {
                    "userdata": {
                        "pre-bootstrap": {
                            "scriptURL": "https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-init.sh",
                            "failonerror": true
                        }
                    }
                }
            }, {
                "networking": {
                    "eth0": {
                        "seclists": [
                            "/Compute-<Identity Domain>/default/default"
                        ],
                        "nat": "ippool:/oracle/public/ippool",
                        "dns": [
                            "k8s-node2"
                        ]
                    }
                },
                "name": "/Compute-<Identity Domain>/<OPC Account>/k8s_node2",
                "storage_attachments": [{
                    "volume": "/Compute-<Identity Domain>/<OPC Account>/node2_disk_vol",
                    "index": 1
                }],
                "boot_order": [],
                "hostname": "k8s-node2",
                "label": "k8s_node2",
                "shape": "oc3",
                "imagelist": "/oracle/public/OL_7.2_UEKR4_x86_64",
                "sshkeys": [
                    "/Compute-<Identity Domain>/<OPC Account>/orclA"
                ],
                "attributes": {
                    "userdata": {
                        "pre-bootstrap": {
                            "scriptURL": "https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-init.sh",
                            "failonerror": true
                        }
                    }
                }
            }]
        }]
    }]
}
```

这个模版在Compute云Orchestration服务直接运行后会自动启动三个Compute云主机，并自动初始化好一个Kubernetes Master节点和两个Node节点。这个模版还引用了一个Shell脚本也贴出来：

```bash
#! /bin/bash

cat <<EOF > /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://packages.cloud.google.com/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg
        https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF

mv /etc/yum.repos.d/public-yum-ol7.repo /etc/yum.repos.d/public-yum-ol7.repo.bak
sed '/ol7_addons/{ n; n; n; n; n; s/enabled=0/enabled=1/; }' /etc/yum.repos.d/public-yum-ol7.repo.bak > /etc/yum.repos.d/public-yum-ol7.repo

yum install -y wget python-dateutil docker-engine kubelet kubeadm kubernetes-cni

mv /etc/sysconfig/docker /etc/sysconfig/docker.bak
sed 's/--selinux-enabled/--selinux-enabled --exec-opt native.cgroupdriver=systemd/g' /etc/sysconfig/docker.bak > /etc/sysconfig/docker

mkfs -t ext4 /dev/xvdb && mkdir -p /var/lib/docker && mount /dev/xvdb /var/lib/docker

mkdir /etc/docker
cat <<EOF > /etc/docker/daemon.json
{
  "storage-driver": "overlay"
}
EOF

systemctl enable docker && systemctl start docker
systemctl enable kubelet && systemctl start kubelet

if [ $HOSTNAME = "k8s-master" ]; then
	wget https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-master.sh
	chmod +x k8s-master.sh
	./k8s-master.sh
	rm k8s-master.sh
else
	wget https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/k8s-nodes.sh
	chmod +x k8s-nodes.sh
	./k8s-nodes.sh
	rm k8s-nodes.sh
fi
```

如果大家对于Compute云编排功能完全不了解还是建议大家先阅读一下我上面提到过的两个文档。下面我将只会对一些重点介绍一下这两个文件的细节。Kubernetes集群的模版文件中对云的资源主要做了三个部分的划分，分别对应到三个oplan中：

* k8s_master_instances
* k8s_node_instances
* k8s_cluster_vol

其中

* k8s_master_instances定义了Kubernetes集群master节点的主机资源及初始化运行的脚本
* k8s_node_instances定义了集群node节点的主机资源及初始化脚本
* k8s_cluster_vol定义了集群所有Compute云块存储（云硬盘）

这三个资源组有明显的逻辑依赖关系，必须先创建k8s_cluter_vol，接着创建k8s_master_instances，最后创建k8s_node_instances，因此我们在模版的最前面使用relationships标签定义好了三组资源的依赖关系。

在k8s_master_instances和k8s_node_instances中我们都使用instances标签来定义Compute云主机资源，其中的重点是：

* networking标签定义了Compute云主机的网络设置，包含了我们在前面定义的防火墙规则就是在这里通过名称进行了引用，同时还定义了外网IP映射以及Compute云内部DNS主机名（在Shared Network内通过主机名或者DNS来访问云主机）；
* storage_attachments标签定义了云主机需要挂载的在k8s_cluster_vol中定义好的块存储；
* shape标签定义了Compute云主机机型决定了主机的内存和OCPU的数量，在我们的模版中统一使用了oc3（1个OCPU／7.5G内存）这个机型；
* imagelist标签定义了Compute云主机使用的操作系统镜像，这个镜像可以是Oracle统一提供的也可以是自定义的，在我们的模版中统一使用了标准的Oracle Linux 7.2 UEK4这个版本的镜像，这里需要说明的是Kubernetes集群和Docker高版本的安装都对Linux操作系统内核是有一定要求的，因此在这里我们选用了Oracle Linux这个版本以满足对于较高内核版本的需求；
* sshkeys标签定义了Compute主机的SSH证书的公钥，就是我们提前上传到Compute云的；
* 最后的attributes标签中嵌套引用了一个脚本的公网http网址，我们就是通过这个shell脚本在主机启动后自动下载并运行安装和初始化Kubernetes集群的，后面我们也会重点介绍一下这个shell脚本的细节，在这里我们还要明确一点脚本的存放可以是互联网通过http/https能够访问的任何地址，在本文的模版中我们使用Oracle Storage对象存储云，大家在测试本文的安装部署时可以不用修改这个地址，直接使用我们提供的地址，这个地址有效期到2017年8月1日。如果想使用自己的Oracle Storage服务可以把这些模版和脚本下载下来，上传到自己的开放为Public的Container中，再把所有的地址替换成你自己的地址就可以了。具体如何使用Oracle Storage存储云大家可以参考如下的[文档](https://docs.oracle.com/en/cloud/iaas/storage-cloud/index.html)。

接下来我们了解一下k8s-init.sh这个脚本，脚本的主要作用是通过kubeadm工具来安装和初始化Kubernetes集群。其实Kubernetes集群的安装和初始化有很多种方式，具体内容大家可以[参考](https://kubernetes.io/docs/setup/pick-right-solution/)，对于kubeadm工具的具体信息可以[参考](https://kubernetes.io/docs/setup/independent/install-kubeadm/)，kubeadm工具的使用我们都已经固化到shell脚本中了。同时这个脚本不但通过yum的方式安装了Kubernetes的组件还安装了docker引擎，而且还将我们挂载到云主机中的100G的块存储进行了格式化，并mount到/var/lib/docker这个目录下，作为docker镜像的存储。

脚本中还解决了几个Kubernetes集群与Oracle Linux 7.2操作系统兼容性的问题：首先建议使用Oracle的yum源来安装docker引擎不要使用Kubernetes的yum源来安装；修改docker引擎的默认存储驱动为overlay模式；添加docker daemon运行参数将Kubernetes和docker的cgroup统一到systemd命名空间下。脚本的最后将根据是否为master节点判断运行k8s-master.sh或者k8s-nodes.sh脚本。
其中k8s-master.sh脚本内容如下：

```bash
#! /bin/bash

kubeadm init --pod-network-cidr=10.244.0.0/16

mkdir /root/.kube
cp /etc/kubernetes/admin.conf /root/.kube/config

kubectl --kubeconfig=/etc/kubernetes/admin.conf taint nodes --all node-role.kubernetes.io/master-

curl -sSL https://rawgit.com/coreos/flannel/master/Documentation/kube-flannel-rbac.yml |  kubectl --kubeconfig=/etc/kubernetes/admin.conf create -f -
curl -sSL https://rawgit.com/coreos/flannel/master/Documentation/kube-flannel.yml |  kubectl --kubeconfig=/etc/kubernetes/admin.conf create -f -


kubeadm token list | sed -n '2, 1p' | awk '{print $1}' > secret

wget https://github.com/s3tools/s3cmd/archive/master.zip
unzip master.zip
wget https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/s3cfg

s3cmd-master/s3cmd -c s3cfg put secret s3://shared/secret

rm -fr s3cmd-master s3cfg secret master.zip
```

这个脚本主要作用就是通过kubeadm来初始化Kubernetes集群，在集群上安装flannel overly网络，并将kubeadm初始化后的密钥上传到Oracle Storage存储中让k8s-node.sh脚本取出连接master节点。

```bash
#! /bin/bash

echo 1 > /proc/sys/net/bridge/bridge-nf-call-iptables

sleep 120

TOKEN=$(curl https://citiccloud.storage.oraclecloud.com/v1/Storage-citiccloud/shared/secret)

kubeadm join --token $TOKEN k8s-master:6443
```

节点初始化脚本比较简单，就是将集群密钥通过Oracle Storage取出并运行kubeadm工具初始化节点。

## 启动模版

讲解过这几个脚本及模版的重点内容后，我们看一下如何使用这个模版和脚本来从无到有在Oracle Compute云上创建一个Kubernetes集群。首先将上面的Orchestration模版复制粘贴到一个文本文件中，并要修改一下模版的内容，主要是用户的域名城和Compute云用户，需要按照你自己的实际情况修改，具体操作可以将现有模版中的<Identity Domain>和<OPC Account>这两个字符串替换为你自己Oracle Compute帐户的相应字符串。接下来进入Compute控制台的Orchestration功能点击Upload Orchestration按钮：

![Orchestration](/assets/img/k8s-opc-13.png)

将保存为文本文件的Orchestration通过弹出的对话窗口上传到Compute云中：

![Orchestration](/assets/img/k8s-opc-14.png)

上传成功后将在Compute控制台看到名为k8s_cluster的编排模版：

![Orchestration](/assets/img/k8s-opc-15.png)

由于已经事先把我们在模版文件中引用的shell脚本文件存放到Oracle Storage云中，接下来我们就可以直接启动这个模版，点击模版行最右侧的图标再点击弹出菜单中的Start链接启动模版。这个时候模版的状态会从Stop转变为Starting，这时我们要等候一段时间，我们可以通过不断刷新Instances页面来查看启动Kubernetes的三个节点的状态：

![Orchestration](/assets/img/k8s-opc-16.png)

当所有的Instance的状态变成Running时，再等待10分钟我们就可以首先登录到k8s_master这个Compute主机上来看一下Kubernetes集群的状态，通过SSH工具使用私钥证书登录到k8s_master的Public IP，用sudo权限运行
```bash
kubectl get nodes
```
查看所有Kubernetes节点的注册情况。

![Kubernests Nodes](/assets/img/k8s-opc-17.png)

看到了三个节点的Ready状态后表明我们的Kubernetes集群已经全部安装和初始化完成了。

## 部署容器应用

接下来我们将只通过两条条命令在Kubernetes集群上启动一个纯粹的微服务架构下的全功能的电商网站。这个电商网站示例代码的Git[地址](https://github.com/microservices-demo/microservices-demo)大家可以到这个网页上参考具体的信息。在k8s_master节点上运行以下命令：

```bash
kubectl create namespace sock-shop
kubectl apply -n sock-shop -f "https://github.com/microservices-demo/microservices-demo/blob/master/deploy/kubernetes/complete-demo.yaml?raw=true"
```
![Container](/assets/img/k8s-opc-18.png)

接着在命令行工具中我们就会看到有很多Kubernetes的服务被创建出来，接着我们要设置Oracle Compute云的防火墙来让我们可以通过互联网能够访问到这个新部署电商网站。这个网站最终会发布一个Kubernetes对外端口30001让用户能够通过浏览器访问，因此我们要新建一个Security Applications：

![Container](/assets/img/k8s-opc-19.png)

填写名称及要开放的端口：

![Container](/assets/img/k8s-opc-20.png)

新建一个Security Rules：

![Container](/assets/img/k8s-opc-21.png)

填写以下信息：

![Container](/assets/img/k8s-opc-22.png)

然后我们再回到k8s-master的SSH命令行工具运行
```bash
kubectl get pods –n sock-shop
```
![Container](/assets/img/k8s-opc-23.png)

来查看示例电商的应用所有的pods是否都已经启动，如果状态都为running就表明整个系统已经可以使用了，我在浏览器中输入Kubernetes三个节点的任意一个节点的公网IP加30001端口号就能访问这个电商系统了：

![Container](/assets/img/k8s-opc-24.png)

接下来我们可以在这个系统里随意创建一个用户，并加购物车、下单等电商的所有主要功能都可以实现。

## 删除所有资源

最后我们测试完毕后如果要释放所有资源时也非常方便，只需要停止Orchestration模版就可以删除所有的Oracle Compute资源，这里需要注意的是你所有的测试数据也将被删除不会有任何恢复的可能，因此每次关闭Orchestration模版时我们都要考虑清楚。

![Container](/assets/img/k8s-opc-25.png)

## 总结

通过使用Oracle Compute云的各种服务特别是Orchestration功能应该非常快速的部署一个Kubernetes集群，接着我们还借助Kubernetes对容器封装下微服务良好支持快速部署了一个电商网站。

