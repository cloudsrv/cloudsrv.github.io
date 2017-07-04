---
layout: post
title:  "很久以前NodeJS学习笔记"
date:   2012-11-03 08:14:41 +0800
categories: Node NodeJS
---

# 前言

学习Node.js，这东西发展太快，看了一个非常不错的教程是2010年写的，刚还不到两年里面所有的示例代码在新框架的升级下基本都不能用了，自己打算把它们重新整理一下发布在这里，这也是开这个博客的最直接的动机。

那个系列教程最初发表在[这里](http://dailyjs.com)，系列一的[地址](http://dailyjs.com/2010/11/01/node-tutorial/)

# 让我们基于Node.js创建一个Web应用：记事本（一）

欢迎来到让我们基于Node.js创建一个Web应用的第一部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

我们要创建一个web的记事本名为Nodepad，不是特别有创意但是明了，而且容易理解。

#### 选择框架和工具

现代的web应用程序依赖于以下几个技术部件：

* 存储：关系型数据库或者NoSQL数据库
* 存储库：简单的（simple）或者ORM
* 应用服务器
* 包管理器
* 服务端框架
* 客户端框架
* 测试库
* 版本控制

最终我们会依据具体应用场景进行选择。我们必须基于部署环境来选择某种框架。而当我构建开源软件的架构时，喜欢使用一些我认为大家都能够容易安装的技术部件。

在本指南的情形中我的选择标准会基于读者们反馈，他们对什么感兴趣，以及我自己精通的领域

#### 服务端

使用Node构建web应用通常需要一个某种类型的框架。如果你订阅了我们每周的Node Roundups，就会知道每周都会有很多非常优秀的web应用程序框架涌现。有一些是为了提供一个全面的解决方案比如Node的Rails或者Django，其他更多的则是关注于路由（Routing）和HTTP协议层。

例如流行的像Rails的框架Geddy，它大概已经发布了7个月，并且在持续的更新。一个更简单的，Sinatra风格的框架是Express自2009年7月发布，并且也在持续更新。

大一些的框架通常通过模型（Model）、界面（View）和控制器（Control）的抽象提供一对多的资源到文件的映射。一个工程通常会像如下组织：

* 模型：user.js note.js
* 控制器：users.js notes.js
* 界面：index.html（项目列表） edit.html new.html

并不是所有的框架使用MVC模式。在Node世界中也有很多微框架，我原以为Express是微框架，但是它将抽象处理整个web应用，因此它会显得更沉重一些。

我觉得Express对于这个指南是非常合适的，我们也可以在本指南中使用其他任何框架，但是Express封装了足够多的技术细节能够使这个项目愉快进行而不显的更加沉重。

#### 前端

一个UI框架可以减少琐碎的界面开发的工作。JavaScript UI框架在最近几年得到了爆炸式的发展，有如下几种类型：

像桌面程序的富框架：Cappuccino、Sprout、Ext.js
提供底层功能的框架：jQuery、Prototype、MooTools
混合底层功能的富框架：YUI
基于底层功能并专注界面的框架：Scriptaculous、jQuery UI
我觉得SproutCore和Cappuccino对于这个项目太沉重，而jQuery UI和Aristo主题一起配合使用可以得到非常大的成果同时而不会使应用程序感觉特别沉重。

#### 测试

在让我们建一个框架系列讲座中，我提到有一个CommonJS联合测试的规范，Nodeunite基于CommonJS的断言（assert）模块，并且这些测试的方法将与你之前写的联合测试脚本非常相像。

另外Expresso提供了assert.response，可以更为简洁地测试服务器。Expresso与Express是同一个作者（TJ Holowaychuk），所有她能实现这一点并不奇怪。

我现在并不确定哪一个更适合，所以两个我都会用一下看看哪一个更适合。

#### 存储

存储在最近几年也变得非常疯狂，在一开始时只有关系型数据库或者对象型数据库，但我从来没有真正的使用过，包括Oracle、PostgreSQL或者MySQL。现在NoSQL却占据了焦点，我们现在有CouchDB、MongoDB、Riak、Redis、Tokyo Cabinet和很多很多。

如果你曾经使用过这些NoSQL的框架，那么你的知识将足够应付，再加上JavaScript的壳，那样将会令我们的工作非常容易。这些框架也经常被分为键值存储和文档存储系统两种类型，我们要建一个记事本，因此文档存储听起来更贴切一些。

选择存储系统是一回事，但是接下来需要为你的语言和框架选择一个程序库。Express没有对模型层抽象进行描述，因此我们可以使用任何我们喜欢的存储程序库和系统。

我想在这个项目中使用MongoDB，其实CouchDB也会是非常好的选择，但是我已经用Mongoose写了很多的代码，我还是想更多的利用它们。

Mongoose API使用并不是非常复杂，非常简洁的语法就能异步的与数据库进行交互。Node流行的“默认”MongoDB库到处使用回调，而不是整洁的抽象和函数调用链，因此非常难读。

我还使用过Heroku和MongoHQ，这些PasS的云服务简化了我的系统管理工作。但是MongoDB是开源技术，我们可以下载到本地开发，而在完成后再部署到这些服务中，如果有时间我还会部署到自己的服务器中。

#### 资源

我写这一部分的目的是为了演示如何为一个真正的开源或者商业项目选择合适的技术，如果你也面临同样的问题，下面有一些有用的资源：

* [Comparison of JavaScript frameworks](http://en.wikipedia.org/wiki/Comparison_of_JavaScript_frameworks)
* [Node modules on the Node wiki](http://github.com/ry/node/wiki/modules)
* [NoSQL resources](http://nosql-database.org/)
* [NPM package list](http://npm.mape.me/)

#### 下一步

在接下来的部分，我将演示如何搭建开发环境和创建一个基本的应用。将涉及如下内容：

安装所有东西
创建一个简单的Express应用
写一个Node应用程序测试
使用jQuery UI创建一个富界面的应用
在Node中使用Mongoose
部署
有一些部分可能在几个星期内完成，这取决于它的参与程度，例如框架系列。