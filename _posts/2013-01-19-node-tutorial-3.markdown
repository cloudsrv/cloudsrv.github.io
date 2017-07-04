---
layout: post
title:  "让我们基于Node.js创建一个Web应用：记事本（三）"
date:   2013-01-19 14:23 +0800
categories: Node NodeJS
---

欢迎来到让我们基于Node.js创建一个Web应用的第三部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

[第一部分](/node/nodejs/2012/11/03/node-tutorial.html)：介绍这个系列以及讨论如何为你的Node项目选择合适的库。

[第二部分](/node/nodejs/2012/11/03/node-tutorial-2.html)：安装和骨架应用，源代码提交：[4ea936b](https://github.com/alexyoung/nodepad/commit/4ea936b4b426012528fc722c7576391b48d5a0b7)

下面的部分我们将修改上面的骨架应用。我已经添加了一个简单的Document模型，因此让我们简单的回顾一下。下面的教程需要你有相应的源代码，可以访问nodepad获得。

## 日志

让我们添加一些日志。Express就有一个日志模块，可以在app.configure块中进行配置。你只需要确保use它：

```js
app.configure(function() { 
   app.use(express.logger()); 
   // Last week's configure options go here 
});
```

最好根据环境的不同使用不同的日志配置，下面我就是这样配置的：

```js
app.configure('development', function() { 
     app.use(express.logger()); 
     app.use(express.errorHandler({ dumpExceptions: true, showStack: true }));  
    });

app.configure('production', function() { 
     app.use(express.logger()); 
     app.use(express.errorHandler()); 
});
```

## API

我们可以使用基于HTTP CRUD（Create/Read/Update/Delete）方法的RESTful接口模型操作documents：

* GET /documents – 索引方法返回整个documents的列表
* POST /documents/ – 创建一个新的documents
* GET /documents/:id – 返回一个document
* PUT /documents/:id – 更新一个document
* DELETE /documents/:id – 删除一个document

注意索引和创建方法都使用同样的URL，但是根据HTTP的GET和PUT动作响应是相差很大的。

## HTTP动作非常重要

如果之前你没有这么使用过HTTP协议，那么只要记住动作非常重要，例如之前我们定义过这样的方法：

```js
app.get('/', function(req, res) { 
    // Respond to GET for '/' 
    // ... 
});
```

如果你使用同样的URL但是使用post请求，Express会返回一个错误，因为没有设置任何的路由。

同样回忆一下之前我们添加了一个express.methodOverride配置。原因是我们不能依赖于浏览器能够理解HTTP动作比如DELETE，但是我们可以用一个简便的方法解决这个问题，form可以使用隐藏的变量，而Express可以将其解释成相应的HTTP方法。

这种方法在某种程度上使用RESTful的HTTP接口有点不太优雅，但是使用这种简便的方法的好处是可以让很多的web浏览器适应这种方法。

## CURD参考方法

作为参考，下面的代码创建了很多的路由使用CRUD，比如：

```js
// List 
app.get('/documents.:format', function(req, res) { 
});

// Create 
app.post('/documents.:format?', function(req, res) { 
});

// Read 
app.get('/documents/:id.:format?', function(req, res) { 
});

// Update 
app.put('/documents/:id.:format?', function(req, res) { 
});

// Delete 
app.del('/documents/:id.:format?', function(req, res) { 
});
```

注意Express使用del而不是delete。

## 异步使用数据库

在我们编写每个REST方法之前，让我们看一个例子：装载一个document列表，你可以使用如下代码：

```js
app.get('/documents', function(req, res) { 
   var documents = Document.find().all();

   // Send the result as JSON 
   res.send(documents);

}
```

我们基本上在Node中异步调用数据库的代码库，这意味着我们需要使用如下代码：

```js
app.get('/documents', function(req, res) { 
   Document.find().all(function(documents) { 
      // 'documents' will contain all of the documents returned by the query 
     res.send(documents.map(function(d) { 
         // Return a useful representation of the object that res.send() can send as JSON 
         return d.__doc; 
       })); 
   }); 
});
```

不同的地方是使用回调函数来操作记录，这个例子并不是特别好，因为它会将每条记录装载到一个数组中，更好的方式是应该在这些记录可用的时候直接将其以流的方式传递给客户端。

格式

我将要支持HTML和JSON，使用如下的方式实现：

```js
// :format can be json or html 
app.get('/documents.:format?', function(req, res) { 
   // Some kind of Mongo query/update 
   Document.find().all(function(documents) { 
   switch (req.params.format) { 
      // When json, generate suitable data 
      case 'json': 
        res.send(documents.map(function(d) { 
           return d.__doc; 
        })); 
        break;

      // Else render a database template (this isn't ready yet) 
      default: 
         res.render('documents/index.jade'); 
       } 
   }); 
});
```

这段代码演示一个Express/Connect装载数据的核心功能：路由字符串是使用:format来判断客户端请求的JSON还是HTML，问号表示可以不用必须给出格式代码。

注意这种方式数据库操作的回调函数包含了实际的响应代码，同样在保存和删除动作中也将使用这种模式。

重定向

创建document的方法返回JSON格式的document，如果客户端请求HTML格式客户端将重定向：

```js
app.post('/documents.:format?', function(req, res) { 
   var document = new Document(req.body['document']); 
   document.save(function() { 
      switch (req.params.format) { 
        case 'json': 
          res.send(document.__doc); 
          break;

        default: 
          res.redirect('/documents'); 
      } 
   }); 
});
```

这里使用了res.redirect将浏览器重新定向到document的列表页面，将其定向到编辑表单也同样容易。我们将在添加用户界面的时候做更为详细的介绍。

## 测试

我通常使用测试的API来开始构建应用，这将在没有写客户端代码的时候做很多它们需要完成的任务。最先做的事情就是为测试数据库添加一个数据库连接：

```js
app.configure('test', function() { 
   app.use(express.errorHandler({ dumpExceptions: true, showStack: true })); 
   db = mongoose.connect('mongodb://localhost/nodepad-test'); 
});
```

然后在test/app.test.js中强制是使用测试环境：

```js
process.env.NODE_ENV = 'test';
```

使用测试数据库意味着我们可以安全得将里面的数据删除。

测试代码本身也需要一些工作要做。Expresso测试对于测试Express的应用非常适合，但是更为精细的用法以及对源码的阅读是非常必要的。

下面将介绍一个例子：

```js
'POST /documents.json': function(assert) { 
    assert.response(app, { 
        url: '/documents.json', 
        method: 'POST', 
        data: JSON.stringify({ document: { title: 'Test' } }), 
        headers: { 'Content-Type': 'application/json' } 
     }, { 
        status: 200, 
        headers: { 'Content-Type': 'application/json' } 
    },

    function(res) { 
        var document = JSON.parse(res.body); 
        assert.equal('Test', document.title); 
    }); 
}
```

测试的名称POST /documents.json可以使用任意的字符串，测试框架实际上不会解析它们，请求将定义在第一组参数中。在这个例子中我指定了请求头的Content-Type，如果没有指定正确的类型，Connect中间件将不会解析data。

我特别的写了一个测试JSON和application/x-www-form-urlencoded的请求，读者可能自己会在测试方法中写入自己的代码。必须特别注意的是Express不会自动的处理表单的编码，这就是为什么我们需要在配置块儿中设置methodOverride的原因。

参考代码提交：[commit 39e66cb](https://github.com/alexyoung/nodepad/commit/39e66cb9d11a67044495beb0de1934ac4d9c4786)

## 结论

你现在应该理解怎样：

在Express中使用CRUD的方法对应HTTP相应的动作；
用Express、Expresso和Mongoose来构建可以测试的应用；
编写简单的Expresso测试。
下面我么将完成document API的方法并添加一些基本HTML模板，我打算添加一个基于jQuery UI库的用户界面，但是在这之前我们需要使用这些测试和API。