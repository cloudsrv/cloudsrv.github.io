---
layout: post
title:  "让我们建一个Web应用：记事本（九）"
date:   2013-02-23 20:25 +0800
categories: Node NodeJS
---

欢迎来到让我们建一个Web应用的第四部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

* 第一部分：介绍这个系列以及讨论如何为你的Node项目选择合适的库。
* 第二部分：安装和骨架应用，源代码提交：
* 第三部分：RESTful方法和测试，源代码提交：
* 第四部分：模板、模板引用以及创建和编辑Document，源代码提交：
* 第五部分：授权、会话和中间件控制权限，代码提交：
* 第六部分：界面基础，代码提交：
* 第七部分：Node库的版本、Jade技巧和错误页面，代码提交：
* 第八部分：闪存消息和帮助器，代码提交：

## 更新connect-mongodb

如果你记得这个系列教程开始时，我曾经写了一段代码来映射connect-mongodb需要的mongo连接串。我在GitHub上联系了作者，他建议更新一下代码库，这样就可以不用mongoStoreConnectionArgs了。

安装需要的版本：

```bash
npm install connect-mongodb@0.1.1
```

现在更新apps.js：

```js
// This is near the top of the file in the var declaration
// The mongoStoreConnectionArgs function can be removed
// In the app configure block, setting up connect.mongodb looks like this
```

## 记住我功能
在web应用保持登录状态的功能涉及到一些服务端的工作，通常以如下的方式运作：

1. 当用户登录时创建一个额外的“记住我”的cookie；
2. cookie包含有用户名和两个随机的数（一个序列号的token和一个随机的token）
3. 这些值也会存放在数据库中；
4. 当某人访问并没有登录时，如果有这个cookie它会与数据库中的进行比对，token将会更新并发回给用户；
5. 如果用户名匹配但是两个token任意一个不匹配，将向用户发出一个警告并将所有的会话删除；
6. 或者cookie将被忽略。

这个方案是为了防止cookie欺骗而设计的，在Barry Jaspan的文章Improved Persistent Login Cookie Best Practice中有详细的描述。

## 创建记住我

在models.js文件我添加了一个LoginToken模型：

```js
mongoose.model('LoginToken', {
indexes: [
methods: {
save: function() {
getters: {
exports.LoginToken = function(db) {
// Load from app.js like this:
```

这是基本的Mongoose代码，它会在模型保存时自动创建一个token。

## 视图
现在添加一个简单的Jade模板views/sessions/new.jade：

```jade
div
```

## 控制器

会话的POST方法应该更新以便在需要的时候创建一个LoginToken：

```js
app.post('/sessions', function(req, res) {
// Remember me
res.redirect('/documents');
```

当退出时应该删除token：

```js
app.del('/sessions', loadUser, function(req, res) {
```

## Express Cookie的使用

Express cookie API的基本使用如下：

```js
// Create a cookie:
// Read a cookie:
// Delete a cookie:
```

cookie的名称总是小写的。注意任何写的操作的结果都将发回到浏览器（res），而读的操作通过一个请求对象req来完成。

## 更新loadUser中间件

现在我们需要做一个loadUser检验是否有一个LoginToken：

```js
function authenticateFromLoginToken(req, res, next) {
LoginToken.find({ email: cookie.email,
User.find({ email: token.email }).first(function(user) {
token.token = token.randomToken();
function loadUser(req, res, next) {
```

注意我已经将LoginToken代码放在它自己的函数中，这使loadUser的可读性更强。

## 结论

这是Barry Jaspan建议的有些简化方法的版本，但是非常容易理解并演示出Express的cookie处理的高级功能。

这个版本的代码提交：
