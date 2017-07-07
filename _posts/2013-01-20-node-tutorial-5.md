---
layout: post
title:  "让我们建一个Web应用：记事本（五）"
date:   2013-01-20 20:25 +0800
categories: Node NodeJS
---

欢迎来到让我们建一个Web应用的第四部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

* 第一部分：介绍这个系列以及讨论如何为你的Node项目选择合适的库。
* 第二部分：安装和骨架应用，源代码提交：
* 第三部分：RESTful方法和测试，源代码提交：
* 第四部分：模板、模板引用以及创建和编辑Document，源代码提交：

在开始本部分教程之前，如果你的系统没有自动启动mongodb，请先将它开启。

## 授权

我们已经创建了一个可以提供服务的应用，但是如果没有任何授权系统的话它将没有任何意义。尽管很多生产系统和客户项目都有类似的OpenID和OAuth等等授权系统，很多商业的项目还是喜欢使用自己的登录系统。

这个系统通常需要完成以下任务:

* 用户填写一个有用户名和密码的表单
* 密码会被哈希算法和一个随机数加密
* 这个值会与数据库中值进行比对
* 如果相同，产生一个会话钥匙标识这个用户

我们需要如下的内容来管理用户和会话：

* 数据库中用户
* 可以存放已经登录的用户ID的会话
* 密码加密
* 限制路由的访问，只允许已登录的用户访问

## Express中的会话

Express使用Connect的会话中间件来管理会话，这种方式后台会有一个存储数据的机制。现在有基于内存的存储以及第三方的存储模式比如connect-redis和connect-mongodb。另外一种方式是使用cookie-sessions把会话数据存放在用户的cookies中。

一个会话可以进行如下配置：

```js
app.use(express.cookieDecoder());
app.use(express.session());
```

这些配置选项的位置非常重要，如果不正确会话变量将不会出现在请求对象中。我吧它放在了bodyDecoder之后methodOverride之前，请参考GitHub上的源代码。

现在我们的HTTP响应就可以进入req.session了：

```js
app.get('/item', function(req, res) {  req.session.message = 'Hello World';});
```

## MongoDB会话

安装connect-mongodb，运行npm install connect-mongodb
connect-mongodb与其他的会话存储工作方式相同，在应用配置的时候我们需要指定连接信息：

```js
app.configure('development', function() {  app.set('db-uri', 'mongodb://localhost/nodepad-development');});
var db = mongoose.connect(app.set('db-uri'));
function mongoStoreConnectionArgs() {  return { dbname: db.db.databaseName,           host: db.db.serverConfig.host,           port: db.db.serverConfig.port,           username: db.uri.username,           password: db.uri.password };}
app.use(express.session({  store: mongoStore(mongoStoreConnectionArgs())}));
```

如果使用标准格式连接选项的API不需要这么多的代码，我写了一段代码将Mongdb的连接信息直接从Mongoose中取出。在这个例子中，db存放了Mongoose的连接实例，Mongoose需要将连接的详细信息通过URI的方式指定，我喜欢这么做，会非常容易记。我已经把每个环境的URI连接串都用app.set存放起来。

在编写Express应用时使用app.set('name','value')是一个很好的主意，只是要记住app.set('name')重新获取设置的值而不是app.get。

在mongo控制台中运行db.sessions.find()可以返回已经创建的所有会话。

## 授权控制中间件

Express提供了一个优雅的方式来限制非授权用户的访问。当HTTP处理器定义的时候有一个可选的中间件参数可以实现额外的权限控制：

```js
function loadUser(req, res, next) {  if (req.session.user_id) {    User.findById(req.session.user_id, function(user) {      if (user) {        req.currentUser = user;        next();      } else {        res.redirect('/sessions/new');      }    });  } else {    res.redirect('/sessions/new');  }}
app.get('/documents.:format?', loadUser, function(req, res) {  // ...});
```

现在所有的路由的可以通过添加loadUser来控制只能由登录用户访问。中间件自身会获得路由的其他参数以及next参数，这个参数可以使路由处理任何逻辑。在我们的项目中用户的加载使用一个在会话中的use_id，如果用户没有找到next就不会被执行而是将浏览器重定向到登录界面。

##会话的RESTful模型

我为会话建了与document相类似的模型，同样有new、delete和create路由：

```js
// Sessionsapp.get('/sessions/new', function(req, res) {  res.render('sessions/new.jade', {    locals: { user: new User() }  });});
app.post('/sessions', function(req, res) {  // Find the user and set the currentUser session variable});
app.del('/sessions', loadUser, function(req, res) {  // Remove the session  if (req.session) {    req.session.destroy(function() {});  }  res.redirect('/sessions/new');});
```

## 用户模型

User模型比Document模型要复杂的多，原因是其中包含了一些与授权相关的逻辑代码。我使用的策略与其他你们可能之前在OO类的web框架中像类似：

* 密码加密后存储，并有一个随机数
* 对于一个用户的授权可以通过比对提供密码的加密值与存储密码的加密值是否一致来实现
* 将明文密码设定为一个“虚拟”的password属性来方便注册和登录表单使用
* 这个属性会有设置器将密码在存储前自动加密
* 一个唯一性索引将保证没有email地址只会注册一个用户

密码的加密使用Node标准的crypto库来实现：

```js
var crypto = require('crypto');
mongoose.model('User', {  methods: {    encryptPassword: function(password) {      return crypto.createHmac('sha1', this.salt).update(password).digest('hex');    }  }});
```

实例方法encryptPassword会返回一个sha1的哈希密码和一个随机数，随机数是在密码设置器加密密码之前生成的：

```js
mongoose.model('User', {  // ...
setters: {    password: function(password) {      this._password = password;      this.salt = this.makeSalt();      this.hashed_password = this.encryptPassword(password);    }  },
methods: {    authenticate: function(plainText) {      return this.encryptPassword(plainText) === this.hashed_password;    },
makeSalt: function() {      return Math.round((new Date().valueOf() * Math.random())) + '';    },
// ...
```

随机数可以是任何你喜欢的值，这里我产生了一个非常随机的字符串。

## 保存用户和注册

Mongoose允许重写save方法来非常容易的实现在记录保存的时候要处理的逻辑：

```js
mongoose.model('User', {  // ...  methods: {    // ...
save: function(okFn, failedFn) {      if (this.isValid()) {        this.__super__(okFn);      } else {        failedFn();      }    }
// ...
```

我重写save的目的是添加一个错误保存的处理方法，这就会让处理错误的注册非常简单：

```js
app.post('/users.:format?', function(req, res) {  var user = new User(req.body.user);
function userSaved() {    switch (req.params.format) {      case 'json':        res.send(user.__doc);      break;
default:        req.session.user_id = user.id;        res.redirect('/documents');    }  }
function userSaveFailed() {    res.render('users/new.jade', {      locals: { user: user }    });  }
user.save(userSaved, userSaveFailed);});
```

现在还没有错误信息显示出来，我会在下面的教程中添加。

尽管这个验证没有做任何事情，但是索引对于整个应用是非常重要的：

```js
mongoose.model('User', {  // ...
indexes: [    [{ email: 1 }, { unique: true }]  ],
// ...});
```

这个将能够阻止重复用户的保存。格式与MongDB的ensureIndex是一样的。

## 结论

代码提交：

* MongDB会话
* 用户模型，支持sha1密码加密
* 路由中间件控制document的授权
* 用户注册和登录
* 会话管理

我已经更新了Jade模板包含了一个简单的登录表单。

目前我们应用的版本还少一些东西：

* Documents没有考虑拥有者
* Expresso测试的时候对于会话的控制有一些问题

后边的教程我将处理这些问题。
