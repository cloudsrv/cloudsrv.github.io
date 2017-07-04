---
layout: post
title:  "让我们基于Node.js创建一个Web应用：记事本（二）"
date:   2013-01-18 19:58 +0800
categories: Node NodeJS
---

欢迎来到让我们基于Node.js创建一个Web应用的第二部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

第一部分：介绍这个系列，以及讨论如何为你的Node项目选择合适的程序库。

这一部分将涉及安装基本的工具和程序库，我们也将使用框架的脚手架功能创建一个应用的骨架并且看一下产生的代码。

## 前提

这个项目依赖于以下内容：

一个已经安装的
* Node
* MongoDB
* npm
我将带领大家安装以上所有的组件，至于为什么选择这些技术组件请参考第一部分。

## 安装：Node

如果还没有安装Node，下载它并且解压。我使用0.2.4版本。如果已经安装了程序包管理器也可以用它来安装，比如Debian、Homebrew recipes等等。

你应该可以编译并安装Node如下：

{% highlight bash %}
./configure 
make 
make install
{% endhighlight %}

你可能需要在运行make install命令之前修改安装目录的权限，或者使用sudo / su命令执行。

这些步骤是在类Unix系统上执行的步骤，如果你使用windows步骤将有所不同，但是在windows上运行node是完全可以的。

## MongoDB

我想使用MongoDB作为我们的数据库。它安装起来也非常简单——有相应的安装包和可供编译的源码。网站上还有很多可以直接使用的二进制文件。

MongoDB需要一个数据目录，可以使用mkdir -p /data/db命令来创建。这是一个默认的路径，但是可以通过运行MongDB启动命令的参数修改，在我所有的开发机器中我都是用这个路径。

MongoDB Quikstart Guide是一个可以更多安装帮助的地方。

## npm

使用npm管理Node的程序包会比直接下载源代码更为快捷和容易。建议安装的方法是在一个有写权限的位置运行命令下载一个脚本并且安装npm。

如果在你自己的机器上开发，可以执行chown -R $USER /usr/local并且运行

{% highlight bash %}
curl http://npmjs.org/install.sh | sh
{% endhighlight %}

或者在你的home目录使用prefix创建Node：

{% highlight bash %}
./configure --prefix=~/local
{% endhighlight %}

npm将使用这个设置并将包安装在Node中，详细内容请参考gist 579814

## 程序包

现在我们有了npm可以安装需要的包：

{% highlight bash %}
npm install express mongoose jade less expresso
{% endhighlight %}

不要将verbosity选项关掉，命令产生的消息不是很复杂，知道确认每个包打印出Success就可以了。

一个简单的使用Express和MongoDB的应用

当我使用Mongo的时候，经常直接在本地直接执行：

## mongod

这个命令执行中将打印出使用的端口，记下来，在Mongoose连接串中会使用这个设置。

Express有一个命令行的脚手架工具可以创建简单的应用，在相应的目录中执行如下命令产生一个应用的骨架：

{% highlight bash %}
express nodepad
{% endhighlight %}

得到的代码可以执行node app.js来运行，可以通过如下链接来查看结果：

[http://localhost:3000](http://localhost:3000)

## 骨架分析

第一行是标准的CommonJS：引用express，创建模块app并导出，这样做是为了测试放方便，现在可能觉得导出app有点令人费解。

Express有了很大的变化，所以在使用旧教程的时候一定要小心，它们可能使用不同的API。比较原始的版本添加了中间件层的connect。这样就可以使用大量的HTTP堆栈和web应用框架并进行互换。特定的配置也发生了很大的变化。

你应该注意以下的代码：

{% highlight js %}
app.configure(function() { 
   app.set('views', __dirname + '/views'); 
   app.use(express.bodyDecoder()); 
   app.use(express.methodOverride()); 
   app.use(express.compiler({ src: __dirname + '/public', enable: ['less'] })); 
   app.use(app.router); 
   app.use(express.staticProvider(__dirname + '/public')); 
});
{% endhighlight %}

Express默认的应用程序非常简单，注意指定了视图路径，静态的文件使用staticProvider进行指定。express.bodyDecoder仅仅用来解析application/x-www-form-urlencoded数据，其实就是表单数据。中间件methodOverride让Express应用就像流行的Rails一样提供RESTful服务，HTTP方法像PUT使用隐藏参数。这样做有很大的争论，大概也是Holowaychuk将其作为可选项的原因。

方法体内使用jade模板来产生HTML，并且设置一个变量传递给模板：

```js
app.get('/', function(req, res) { 
   res.render('index.jade', { 
       locals: { 
       title: 'Express' 
       } 
   }); 
});
```

这个方法调用一个路由和相应的HTTP动作“/”和GET，这就说明这段代码将不会适用于一个请求“/”的POST方法。

最后几行很有趣，因为它们会检查这个app是否是在顶层模块运行：

```js
if (!module.parent) { 
     app.listen(3000); 
     console.log("Express server listening on port %d", app.address().port) 
}
```

这段代码同样是为了测试使用的，不用担心它们看起来很奇怪。

## 连接MongoDB

Mongoose可以让我们方便的使用Mongo的各种集合类。我们首先需要先装载库并初始化一个数据库连接实例：

```js
mongoose = require('mongoose').Mongoose 
db = mongoose.connect('mongodb://localhost/nodepad')
```

我还做了一个模型的例子名为models的文件：

```js
var mongoose = require('mongoose').Mongoose;

mongoose.model('Document', { 
     properties: ['title', 'data', 'tags'];

      indexes: [ 'title'] 
});

exports.Document = function(db) { 
     return db.model('Document'); 
};
```

模型可以作为模块引入app.js文件，比如：

```js
Document = require('./models.js').Document(db);
```

数据库实例作为参数传递进模型，而db.model将返回一个基于mongoose.moel('Document',...)的模型实例，我认为将模型放在它们自己的文件里会让Mongoose的行为更为清晰，因此应用程序的控制器代码会更加容易理解。

## 模板

默认使用Jade来作为HTML的产生器，如下：

```js
h1= title 
p Welcome to #{title}
```

这跟Haml很像，主要是为了减少HTML模板的各种杂乱的代码，如果你喜欢HTML模板也可以比如ejs。

## 运行测试

Express也会产生一个骨架的测试，可以使用expresso来运行这个测试。

## 获取代码

代码在GitHum网站上可以得到alexyong/nodepad

## 结论

你现在应该可以使用npm和mongo来搭建一个Node的开发环境，并且可以使用Express来产生骨架的app，同时能够运行Expresso测试。