---
layout: post
title:  "Oracle Compute云快速搭建MySQL Keepalived高可用架构"
date:   2017-07-17 09:25 +0800
categories: Oracle Cloud
---

最近有个客户在测试Oracle Compute云，他们的应用需要使用MySQL数据库，由于是企业级应用一定要考虑高可用架构，因此有需求要在Oracle Compute云上搭建MySQL高可用集群。客户根据自身的技术储备想要使用Keepalived组件来配合MySQL实现。今天结合Oracle Compute刚刚[宣布](/oracle/cloud/2017/07/05/opc-terraform.html)terraform支持的架构即代码方式，交付给客户一个快速搭建MySQL+Keepalived高可用架构，来帮助他们快速搭建测试环境甚至将来使用到正式环境。

## MySQL主主复制模式

MySQL的主主复制模式主要是针对主从复制模式提出的，虽然现在的MySQL主从复制集群能够实现一定的读写分离分担负载，同时当主库发生问题时通过从库提升为主库的方式提升架构的健壮性。但是由于读写分离对应用程序的侵入性以及从库提升为主库响应时间问题，还是有很多客户选择使用主主复制的方式实现高可用架构。特别对于只是希望通过集群复制模式来保障高可用而不是提升MySQL负载能力的场景下，使用主主复制的模式更令广大的MySQL用户所认可，再结合如Keepalived的方案很容易实现自主恢复的高可用架构。

MySQL无论是主主复制还是主从复制都是基于MySQL binlog操作日志来实现的，其基本过程如下：

* 从数据库执行*start slave*开启主从复制
* 从数据库会有一个单独的io线程通过授权的数据库用户连接主库，发送读取主库的*binlog*日志的请求，请求信息包含了主库的*binlong*文件名及日志位置
* 主库接收到从库的io线程的读取请求后，主库上负责复制的io线程根据请求信息读取指定*binlog*文件的指定位置，返回给从库的io线程，返回的信息包括：
	* 本次请求的日志
	* 返回的日志后主库上新的*binlog*文件名及位置
* 从库的io线程获取到主库的*binlog*后，将*binlog*的内容写入到从库的*relay log*（中继日志）文件*mysql-info-realy-bin.XXXX*，最后，并将新的*binlog*文件名和位置记录到*master-info*文件中
* 从库的sql线程会实时监测本地*relay log*中新增的日志，然后把log文件中内容解析成sql语句，并在从库上按日志顺序执行这些sql语句
* 从库的io线程根据MySQL参数设定的频率再次发送*binlog*日志请求到主库，形成下次循环

![](/assets/img/mysql-keepalived-opc/mysql-replica.png)

上面提到的无伦是主主复制还是主从复制都是依赖于上述步骤实现，只是在主主复制的模式下，两个MySQL数据库之间分别实现对对方数据库的从库复制就完成了主主复制模式，也就是说两个数据库之间互为主从，分别在各自数据库中执行上述步骤。

实际上在主主复制模式下有一个在OLTP数据库中比较难解决的问题，就是两个数据库主键冲突的问题比较麻烦，通常需要引入其他的技术组件如redies才能解决，但是由于我们这次构建这个架构主要目标是实现MySQL高可用，没有同时访问两个主主数据库的需求，而不是数据库负载均衡提升数据库的吞吐量，因此我们可以将这个问题忽略。

从以上MySQL的集群复制模式原理我们可以看出来，无论是主主复制还是主从复制的MySQL集群都会有数据延迟的问题，这一点在我们使用MySQL集群复制模式时一定要牢记在心，在整个应用系统架构设计的时候就要考虑。

## Keepalived高可用架构Oracle Compute实现

Keepalived是互联网应用架构思维**Desgin for Failure**典型的代表，使用这个开源组件我们可以很容易在不侵入原有技术组件的条件下通过*Float IP*实现高可用的架构。

它的工作原理是以VRRP协议为实现基础的，VRRP全称*Virtual Router Redundancy Protocl*，中文为[虚拟路由冗余协议](https://en.wikipedia.org/wiki/Virtual_Router_Redundancy_Protocol)。虚拟路由冗余协议可以认为是实现路由高可用的协议，就是将多台提供相同功能的路由组成一个路由组，这个组里有一个master和多个backup，master上面有一个对外提供服务的*vip*或称为*float ip*，当其他局域网内的计算机将*vip*作为路由地址时master会发组播，当backup收不到VRRP包时就认为master宕了，这时就根据VRRP的优先级来选举一个backup当master，这样就保证了路由的高可用。

Keepalived主要有三个模块，分别是core、check和vrrp。Core模块为keepalived的核心模块，主要负责主进程启动、维护以及全局配置文件的加载和解析；check模块负责健康检查，包括常见的各种检查方式，我们一般使用tcp监测；vrrp模块主要来实现VRRP协议。

在Oracle Compute云中，我们主要依靠IP Network来实现传统的OP数据中心企业级内网的功能，但是在云中为了安全与我们传统OP还是有一些差别的，具体到使用VRRP协议上来，主要体现是IP Nework不支持ARP协议，因此VRRP协议中生成的*Float IP*不能在局域网中通过ARP进行广播，我们需要通过IP Network特有的一些功能实现*Float IP*，这其中包括：

* *vNICSet* -- 虚拟网卡集，可以将已有的虚拟网卡定义到这个集合中作为路由的目标；
* *Route* -- 虚拟路由，通过指定特定的IP地址范围将其路由到定义好的*vNICSet*，用这种方式替换了ARP *Float IP*地址注册的方式；
* *Oracle Compute CLI* -- 通过命令行的方式动态修改*Route*改变路由指向的*vNICSet*来实现*Float IP*。

Keepalived与Oracle Compute CLI要结合起来使用来实现路由的动态修改从而实现*Float IP*，Keepalived的check模块可以设置当检测失败后的*通知*来运行脚本，可以编写脚本来调用Oracle Compute CLI；keepalived在集群模式下的VRRP协议需要通过设置的权重来选择替换Mater的backup，但需要将backup的权重设置成至少大于其他keepalived节点50以上且需要relaod配置信息才能切换，由于我们MySQL主主复制模式只有两个节点，只需要在这两个节点间切换就可以了，且在IP Network环境下不使用ARP的方式来注册IP地址，因此我们只是使用VRRP协议给我们在IP Network中的网卡添加*Float IP*就可以了。

具体的在第一台MySQL主机上Keepalived配置信息如下：

```
#! Configuration File for keepalived
global_defs {
router_id mysql01 #修改为自己的主机名
             }
##################第一部分###################
vrrp_instance VI_1 {
     state BACKUP #都修改成BACKUP
     interface eth0
     virtual_router_id 60 #默认51 主从都修改为60
     priority 100 #在mysql-ha2上LVS上修改成80
     advert_int 1
     nopreempt #不抢占资源，意思就是它活了之后也不会再把主抢回来
     authentication {
     auth_type PASS
     auth_pass 1111
     }
virtual_ipaddress {
     192.168.2.88
     }
}
##################第二部分###################
virtual_server 192.168.2.88 3306 {
     delay_loop 6
     lb_algo wrr
     lb_kind DR
     nat_mask 255.255.255.0
     persistence_timeout 50
     protocol TCP
 real_server 192.168.2.11 3306 {
     weight 1
     notify_down /usr/local/mysql/bin/mysql.sh
     TCP_CHECK {
         connect_timeout 10
         nb_get_retry 3
         connect_port 3306
         }
     }
}
```

在第二台MySQL主机上Keepalived配置信息如下：

```
#! Configuration File for keepalived
global_defs {
router_id mysql02 #修改为自己的主机名
             }
##################第一部分###################
vrrp_instance VI_1 {
     state BACKUP #都修改成BACKUP
     interface eth0
     virtual_router_id 60 #默认51 主从都修改为60
     priority 80 #在mysql-ha2上LVS上修改成80
     advert_int 1
     nopreempt #不抢占资源，意思就是它活了之后也不会再把主抢回来
     authentication {
     auth_type PASS
     auth_pass 1111
     }
virtual_ipaddress {
     192.168.2.88
     }
}
##################第二部分###################
virtual_server 192.168.2.88 3306 {
     delay_loop 6
     lb_algo wrr
     lb_kind DR
     nat_mask 255.255.255.0
     persistence_timeout 50
     protocol TCP
 real_server 192.168.2.12 3306 {
     weight 1
     notify_down /usr/local/mysql/bin/mysql.sh
     TCP_CHECK {
         connect_timeout 10
         nb_get_retry 3
         connect_port 3306
         }
     }
}
```

大家要额外注意两台主机上Keepalived配置是有差别的主要体现在VRRP的优先级和各自要检测的服务上。

## 测试环境搭建

我们使用Terraform内置的Oracle Compute支持编写`tf`脚本来代码化Compute的资源，具体Terraform脚本请[参阅](https://github.com/zorrofox/opc-mysql-keepalived)。在这个MySQL主主复制集群架构的测试环境中我们将在Oracle Compute云中创建以下资源：

- 3个计算实例: `mysql_01`, `mysql_02`, `nat`
- 1个IP Networks: `Private_IPNetwork`
- 1个SSH Key: `mysql-example-key`
- 1个公网IP Reservations: `reservation1`
- 2个vNICSet: `mysql_01`, `mysql_02`
- 2个Route: `nat_route`, `mysql_vip`

详细的架构图如下：

![](/assets/img/mysql-keepalived-opc/mysql-keepalived-opc.png)

其中`nat`实例作为跳板机和NAT实例的双重角色拥有一个公网IP，而两个MySQL节点只有私网IP只能通过跳板机访问，两个实例访问公网的YUM仓的资源以及Oracle Compute API的端点都需要通过NAT实例进行网络流量转发。这样的架构设计是按照Oracle推荐的网络安全[最佳实践](/oracle/cloud/2017/07/07/opc-network-nat.html)来实现的。

要部署这个terraform模版，首先需要一个Oracle Compute帐号，并根据自己帐号的情况修改`variables.tf`这个文件中的几个变量：

```

variable user {
  default = "<OPC_USER>"
}
variable password {
  default = "<OPC_PASS>"
}
variable domain {
  default = "<OPC_DOMAIN>"
}
variable endpoint {
  default = "https://api-z50.compute.us6.oraclecloud.com/"
}

variable ssh_user {
  description = "User account for ssh access to the image"
  default     = "opc"
}

variable ssh_private_key {
  description = "File location of the ssh private key"
  default     = "~/keys/orcl.pem"
}

variable ssh_public_key {
  description = "File location of the ssh public key"
  default     = "~/keys/orcl_pub.pem"
}

```

上面的代码中除了与OPC有关的**用户名**、**密码**、**域名称**以及**API端点地址**以外，`ssh_user`是不能修改的，就是操作系统的用户，但是需要提供一个SSH密钥对。这个密钥对需要提前生成，具体生成密钥对的方法可以参考[这里](https://www.liaohuqiu.net/cn/posts/ssh-keygen-abc/)。要在这里同时提供密钥和私钥，本地存放的能够访问的目录就行，需要私钥的原因是在`main.tf`脚本中需要通过Terraform Provisioner模块对MySQL集群进行初始化，包括了MySQL和Keepalived软件的安装和配置等等。修改完这个文件后，在terraform脚本所在的目录执行：

```bash
$ terraform plan
Refreshing Terraform state in-memory prior to plan...
The refreshed state will be used to calculate this plan, but will not be
persisted to local or remote state storage.

data.template_file.pwd: Refreshing state...
data.template_file.default_profile: Refreshing state...

...

+ opc_compute_vnic_set.nat_set
    description: "NAT vnic set"
    name:        "nat_vnic_set"


Plan: 21 to add, 0 to change, 0 to destroy.
```

Terraform会检测整个脚本是否正确，如果没有任何报错可以执行：

```bash
$ terraform apply
data.template_file.default_profile: Refreshing state...
data.template_file.pwd: Refreshing state...
opc_compute_ip_reservation.reservation1: Refreshing state... (ID: f0a98333-9b12-437a-808f-9cba58bb1431)
data.template_file.mysql1_sh: Refreshing state...
data.template_file.mysql2_sh: Refreshing state...
opc_compute_vnic_set.nat_set: Creating...

...


```

将整个架构应到Oracle Compute云中，这时我们需要等待一段时间才能完成整个架构的部署，最后Terraform会将一个外网IP返回，这个IP就是我们NAT实例或者跳板机的IP地址，通过这个地址我们可以访问我们的MySQL集群进行测试。

例如我们可以通过下面SSH客户端登录到NAT实例上，测试我们的*VIP*是否可用，并通过这个IP通过mysql客户端登录到我们的MySQL集群中：

```bash
[opc@nat ~]$ ping 192.168.2.88
PING 192.168.2.88 (192.168.2.88) 56(84) bytes of data.
64 bytes from 192.168.2.88: icmp_seq=1 ttl=64 time=0.586 ms

[opc@nat ~]$ mysql -u ha -pXXXXXX -h 192.168.2.88

```

最后我们如果需要将测试资源释放出来，那么可以使用：

```
$ terraform destroy
```
删除所有资源。
>注意：这样做会将删除所有的测试数据，在执行这个操作以前请确保已经做过备份。

