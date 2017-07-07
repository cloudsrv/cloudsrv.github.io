---
layout: post
title:  "让我们建一个Web应用：记事本（七）"
date:   2013-02-21 20:25 +0800
categories: Node NodeJS
---

欢迎来到让我们建一个Web应用的第四部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

* 第一部分：介绍这个系列以及讨论如何为你的Node项目选择合适的库。
* 第二部分：安装和骨架应用，源代码提交：
* 第三部分：RESTful方法和测试，源代码提交：
* 第四部分：模板、模板引用以及创建和编辑Document，源代码提交：
* 第五部分：授权、会话和中间件控制权限，代码提交：
* 第六部分：界面基础，代码提交：

## 程序包版本

我更新了Nodepade的README文档包含了Node和Mongo的版本，同时也包含了使用的程序包的版本。如果你的代码有问题的话可能就是因为程序包版本的问题。这些代码我在Mac OS和Debian系统上经过测试。

同样要记住当你改了代码需要重新启动Node（但是Jade模板的修改除外）。

我们使用npm安装程序包，有必要还要指定特定的版本。要指定版本，需要运行如下命令：

```bash
npm install express@1.0.0
```

使用这个程序包：

```js
var express = require('express@1.0.0');
```

要验证是否生效输入node并输入如以上的代码：

```js
express = require('express@1.0.0'){ version: '1.0.0', Server: { [Function: Server] parseQueryString: [Function] }, createServer: [Function]}
```

## Jade技巧

开始演示Jade使用的时候我写死了所有的属性，实际上可以通过写一些选择器作为CSS类和ID的简写来降低我们的劳动强度。

```jad
div#left.outline-view  div#DocumentTitles    ul#document-list      - for (var d in documents)        li          a(id='document-title-' + documents[d].id, href='/documents/' + documents[d].id)            =documents[d].title
     
```

注意一个ID的选择器可以与一个CSS类的选择器名称连起来写：

```jad
div#left.outline-view
```

默认的标签是div，这样上面的代码可以简化为如下：

```jade
#left.outline-view  #DocumentTitles    ul#document-list      - for (var d in documents)        li          a(id='document-title-' + documents[d].id, href='/documents/' + documents[d].id)            =documents[d].title
            
```

## 错误页面

Express允许我们通过定义app.error，来定义一个错误的处理器：

```js
// Error handlingfunction NotFound(msg) {  this.name = 'NotFound';  Error.call(this, msg);  Error.captureStackTrace(this, arguments.callee);}
sys.inherits(NotFound, Error);
// This method will result in 500.jade being renderedapp.get('/bad', function(req, res) {  unknownMethod();});
app.error(function(err, req, res, next) {  if (err instanceof NotFound) {    res.render('404.jade', { status: 404 });  } else {    next(err);  }});
app.error(function(err, req, res) {  res.render('500.jade', {    status: 500,    locals: {      error: err    }   });});
```

错误处理器有四个参数，分别为error、req、res和next。其中next方法可以把错误提交给下一个处理器进行处理。上面的例子中404处理器传递一个NotFound的错误，并且我们将捕捉所有其他的错误作为500进行处理。

在浏览器中访问/bad链接可以显示客制化的500页面。注意我在render的选项中指定了HTTP的状态代码，这非常重要，如果没有指定为200那么将会返回404或者是500。

## 根据Mongoose代码进行错误处理

方法next在所有我们应用的HTTP的动作中都是可用的，这就意味着我们可以利用它来实现客制的404页面：

```js
app.get('/documents/:id.:format?/edit', loadUser, function(req, res, next) {  Document.findById(req.params.id, function(d) {    if (!d) return next(new NotFound('Document not found'));    // Else render the template...
```

我发现当使用Mongoose的时候这样做非常容易实现，只需要在Mongoose的回调函数中使用throw new NotFound就可以防止应用崩溃。

## 结论

当发布和部署Node应用程序的时候指定各种程序包的版本是非常重要的，很多重要的程序包还在频繁的修改，所以要保持稳定的发布还是非常困难的。

Express使用模板创建一个客制的错误处理器非常容易，但是在回调函数中一定要指定next(exception)中HTTP状态编码。

这个版本的代码提交：





