---
layout: post
title:  "让我们建一个Web应用：记事本（六）"
date:   2013-02-20 20:25 +0800
categories: Node NodeJS
---

欢迎来到让我们建一个Web应用的第四部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

* 第一部分：介绍这个系列以及讨论如何为你的Node项目选择合适的库。
* 第二部分：安装和骨架应用，源代码提交：
* 第三部分：RESTful方法和测试，源代码提交：
* 第四部分：模板、模板引用以及创建和编辑Document，源代码提交：
* 第五部分：授权、会话和中间件控制权限，代码提交：

在开始本部分教程之前，如果你的系统没有自动启动mongodb，请先将它开启。
在上一部分我们看了一下授权和会话。我们使用了非常酷的中间件来构建控制系统。下面我们将演示如何使用jQuery让我们的界面更有趣。

## 界面设计

在设计界面的时候我习惯在开发应用之前计划出大概的轮廓，这通常被称为自上而下的设计。当界面设计看上去可以的时候，我开始开发API和简单的界面以及测试，就与我们在整个这个教程的过程一致。

我喜欢在纸上把界面的草图画出来，就使用传统的铅笔和橡皮。我尽量保证草图的看上去很粗糙，以便让我的同事一直认为设计还没有完成，当我们讨论的时候他们的意见会被采纳进设计。

![](/assets/img/nodepad/nodepad-sketch.jpg)

一个简单的Nodepad草图设计如下：

* 界面使用类似桌面应用两个面板，一个作为笔记的列表另外一个显示笔记的内容
* 保存按钮显示在底部，但是如果有自动保存将更好一些
* 点击一个笔记将装载内容，双击将可以编辑笔记的标题
* 我们需要帐号设置来修改email/password
* 编辑可以使用XMLHttpRequest，因为我们已经有了JSON的支持。

## 新建

构建界面最重要的原则之一就是尽量的重用。当我们写代码的时候不需要花费大量的时间来重新实现很多已经很成熟库的功能，设计也同样是这样的情况。并不需要画出每个我需要的图标，并且并不需要重新构造所有的布局。

现在有很多解决方案可以用，从CSS框架到富GUI项目像Cappuccino。对于Nodepad我打算使用jQuery UI比较中型的一个框架，它会封装很多内容以及非常健壮的主题。

对于主题，我决定使用Aristo（演示）。它不一定是最好的，但是我有很多使用它的经验，而且它看上去非常棒。

## 引用Aristo和jQuery UI库

我从GitHub上下载了Aristo并把它放在了public/stylesheets/aristo目录中，然后我们只需要在views/layout.jade中引用装载这个新的样式表以及jQuery UI库就可以了：

```html
link(rel='stylesheet', href='/stylesheets/aristo/jquery-ui-1.8.5.custom.css')script(type='text/javascript', src='https://ajax.googleapis.com/ajax/libs/jqueryui/1.8.7/jquery-ui.min.js')
```

## 页面结构

我们的界面需要两栏，一个标题头，一个可以编辑的document内容栏，和一些管理document的按钮。在Jade中内容如下：

```jade
div(class='outline-view', id='left')  div(id='DocumentTitles')    ul(id='document-list')      - for (var d in documents)        li          a(id='document-title-' + documents[d].id, href='/documents/' + documents[d].id)            =documents[d].title
ul(class='toolbar')    li      a(href='/documents/new')        +    li      a(href='#', id='delete-document')        -
div(class='content-divider')
div(class='content')  div(id='editor-container')    textarea(name='d[data]', id='editor')      =d.data || ''
ul(id='controls',class='toolbar')  li    a(href='#', id='save-button') Save
```

第一部分outline-view是一个div包含了document的列表，这在之前我们就使用这段代码，我会把这个div的位置属性设置为absolute，并且使用一些JavaScript在resize和focus来改变document列表和按钮工具条的大小。

被选择的document使用一些圆角css来突出显示：

```css
.outline-view ul .selected {  color: #fff;  background-color: #8897ba;  background: -webkit-gradient(linear, left top, left bottom, from(#b2bed7), to(#8897ba));  background: -moz-linear-gradient(top,  #b2bed7,  #8897ba);}
```

如果浏览器不支持CSS3也不要紧，它看起来也可以就是颜色会暗一些。

## 选择Document

回忆一下我们的API需要在请求的链接后面加上.json来返回JSON数据，我们只需要一些简单的jQuery事件来处理从服务端返回的document数据：

```js
$('#document-list li a').live('click', function(e) {  var li = $(this);
$.get(this.href + '.json', function(data) {    $('#document-list .selected').removeClass('selected');    li.addClass('selected');    $('#editor').val(data.data);    $('#editor').focus();  });
e.preventDefault();});
```

这里绑定一个点击事件，当document标题被点击时会被触发。它会自动将JSON响应中的data值赋给testarea。与普通的事件不同，live可以应对document列表的动态变化。

## 保存Document

当动态界面设置id属性的时候通常基于数据库的id，做一些简单的转换就可以了。在Nodepad中我就是用了诸如document-844ce1799ba1b87d359000001的DOM id。得到数据库的id我们在上面的教程已经涉及，我们只是把它放在了连字符的后面。

建立好这样的约定以后，我就可以使用jQuery的一个小插件来得到item的id：

```js
$.fn.itemID = function() {  try {    var items = $(this).attr('id').split('-');    return items[items.length - 1];  } catch (exception) {    return null;  }};
```

这样当点击save按钮的时候就能直接保存documents：

```js
$('#save-button').click(function() {  var id = $('#document-list .selected').itemID(),      params = { d: { data: $('#editor').val(), id: id } };  $.put('/documents/' + id + '.json', params, function(data) {    // Saved, will return JSON  });
```

在jQuery中没有真正意义上的HTTP动作put，所以我定义了一个：

```js
$.put = function(url, data, success) {  data._method = 'PUT';  $.post(url, data, success, 'json');};
```

## 进展

![](/assets/img/nodepad/nodepad-progress.png)

我们现在没有使用任何jQuery UI，但是我们会在下面的教程中使用。目前我们有一个简单的编辑器，而且看起来比较友好并容易使用。

我还没有来得及做浏览器测试，现在仅限于WebKit或者Firefox。

最后的代码提交：




