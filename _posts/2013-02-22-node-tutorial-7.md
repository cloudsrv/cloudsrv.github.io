---
layout: post
title:  "让我们建一个Web应用：记事本（八）"
date:   2013-02-22 20:25 +0800
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

## 闪存消息

闪存消息是立刻需要显示的服务端消息。在显示之前通常使用会话用来存储闪存消息，同时也会在会话中删除。Express使用Connect闪存中间件支持闪存会话：

```js
req.flash('info', '%s items have been saved.', items.length);
```

第一个参数是类别，我通常用显示错误消息的CSS类与其相关联，用以区分需要不同反馈的消息。第二个参数是要显示的消息，其中可以包含一个格式器（默认只有%s可用）。

## Helpers

Express有两种视图帮助器：静态的和动态的。帮助器可以作为函数的变量，可以如下方式加入到应用中：

```js
app.helpers({  nameAndVersion: function(name, version) {    return name + ' v' + version;  },
appName: 'Nodepad',  version: '0.1'});
```


我喜欢使用一个名为helpers.js的文件将我所有的帮助器都放在里面，并使用require来来加载我需要的帮助器：

```js
app.helpers(require('./helpers.js').helpers);
```

现在更新我们的Jade模板使用帮助器：

```jade
#{nameAndVersion(appName, version)}
```

我把这段代码添加到Nodepad的头部。

动态帮助器可以进入req和res对象，这样我们就可以使用它来得到闪存消息。下面我将演示如何使用帮助器来显示闪存消息。

> 注意：动态帮助器有趣的是它会在视图之前渲染，这意味着它们将作为变量而不是函数出现。

## 为Nodepad添加闪存消息
我们需要一个帮助器来显示闪存消息。添加一个helpers.js：

```js
exports.dynamicHelpers = {
  flashMessages: function(req, res) {
    var html = '';
    ['error', 'info'].forEach(function(type) {
      var messages = req.flash(type);
      if (messages.length > 0) {
        html += new FlashMessage(type, messages).toHTML();
      }
    });
    return html;
  }
};
```

这个循环将遍历所有的消息类型并使用FlashMessage产生一个闪存消息。这是一个新的类以方便我重用jQuery UI的样式：

```js
function FlashMessage(type, messages) {  this.type = type;  this.messages = typeof messages === 'string' ? [messages] : messages;}
FlashMessage.prototype = {  get icon() {    switch (this.type) {      case 'info':        return 'ui-icon-info';      case 'error':        return 'ui-icon-alert';    }  },
get stateClass() {    switch (this.type) {      case 'info':        return 'ui-state-highlight';      case 'error':        return 'ui-state-error';    }  },
toHTML: function() {    return '<div class="ui-widget flash">' +           '<div class="' + this.stateClass + ' ui-corner-all">' +           '<p><span class="ui-icon ' + this.icon + '"></span>' + this.messages.join(', ') + '</p>' +           '</div>' +           '</div>';  }};
```

中间件flash返回多种基于type的消息，所以上面代码使用分隔符连接每个消息来处理这种情况。

这是一个使用switch来处理flashMessages帮助器的逻辑，根据不同的消息类型使用不同的CSS类。它会产生一些HTML代码来与jQuery一起工作：

现在在app.js中装载动态的帮助器：

```js
app.dynamicHelpers(require('./helpers.js').dynamicHelpers);
```

并且在一些地方添加闪存消息：

```js
app.post('/sessions', function(req, res) {  User.find({ email: req.body.user.email }).first(function(user) {    if (user && user.authenticate(req.body.user.password)) {      req.session.user_id = user.id;      res.redirect('/documents');    } else {      req.flash('error', 'Incorrect credentials');      res.redirect('/sessions/new');    }  }); });
```

现在在views/layout.jade中添加帮助器：

```jade
#{flashMessages}
```

## 反馈显示问题

现在的问题是目前的代码与编辑器的设计融合的不够好：

解决这个问题是在styles.less中使用一些样式：

```css
.flash {  position: absolute;  top: 0;  bottom: 0;  z-index: 1001;  width: 100%;  opacity: 0.75;  background-color: #111;}
.flash span {  float: left;  margin-right: .7em;}
.flash .ui-corner-all {  width: 300px;  margin: 50px auto 0 auto;  padding: 0 5px;  opacity: 0.9;  font-weight: bold;  -moz-box-shadow: 0 0 8px #111;  -webkit-box-shadow: 0 0 8px #111;  box-shadow: 0 0 8px #111;}
```

这样会占据整个页面并在消息出现的时候逐渐变黑。为了隐藏他们我在application.js中添加了如下内容：

```js
function hideFlashMessages() {  $(this).fadeOut();}
setTimeout(function() {  $('.flash').each(hideFlashMessages);}, 5000);$('.flash').click(hideFlashMessages);
```

## 结论
Express有静态和动态的帮助器，动态的帮助器就在视图之前渲染，并且他们会传递req和res对象，在视图中可以作为变量访问。

创建一个隔离的文件来存放应用中所有的帮助器是非常容易的，也可以方便的在应用使用require来引用。

我创建了一个FlashMessages类来演示为什么使用一个帮助器文件是很必要的，并且也演示了一些简单的JavaScript OO的方法来实现getter方法。你可能习惯将闪存消息通过动态帮助器来直接开放给模板，那就意味着将要使用Jade来产生闪存消息的HTML，这可以作为一个对于我们自己的一个挑战来完成。

当前版本的Nodepad代码提交：
