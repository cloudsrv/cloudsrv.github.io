---
layout: post
title:  "让我们基于Node.js创建一个Web应用：记事本（四）"
date:   2013-01-19 20:25 +0800
categories: Node NodeJS
---

欢迎来到让我们基于Node.js创建一个Web应用的第三部分，关于使用Node创建一个web应用的新的学习指南。这个系列会引领你使用Node创建一个web应用，涵盖了在搭建你自己应用程序时需要面临的所有主要技术领域。

[第一部分](/node/nodejs/2012/11/03/node-tutorial.html)：介绍这个系列以及讨论如何为你的Node项目选择合适的库。

[第二部分](/node/nodejs/2012/11/03/node-tutorial-2.html)：安装和骨架应用，源代码提交：[4ea936b](https://github.com/alexyoung/nodepad/commit/4ea936b4b426012528fc722c7576391b48d5a0b7)

[第三部分](/node/nodejs/2012/11/03/node-tutorial-3.html)：RESTful方法和测试，源代码提交：[39e66cb](https://github.com/alexyoung/nodepad/commit/39e66cb9d11a67044495beb0de1934ac4d9c4786)

下面的部分我们将先会得到一些成果。在本段教程结束的时候将会得到下面的界面：

![Your Documents](/assets/img/your-doc.png)

在教程中并没有将所有的代码列出来，我只是挑选了一些代码片段作为例子，并且没有用任何的CSS。所有的代码都存放在GitHub中，所以可以下载下来，放在你的工程中。

## 更新Expresso

Expresso更新为0.70，使用rpm update expresso来进行升级。我们开始使用的版本已经与文档对应不起来了，特别是beforeExit方法的使用。

## 渲染模板

document列表方法（/documents）应该渲染一个我们可以编辑的文档列表，我们添加一个相应的render调用：

```js
res.render('documents/index.jade', {
   locals: { documents: documents }
});
```

并且使用相应的模板：

```jade
ul
 - for (var d in documents)
  li=d.title
```

记住我们使用的模板的是Express默认的模板语言Jade。

## Jade

Jade在一开始使用的时候有点奇怪，但是实际上掌握以后就会很容易。下面有一些要点需要牢记：

* 缩进表示标签的嵌套
* 等号表示应用一个变量的值
* 不等号表示在应用一个变量前不要进行转义
* 连字符表示应用一段JavaScript
注意要尽量进行转义，这样最大程度的降低XSS攻击的风险。

## Jade的引用

Jade和Express使用引用来获得一些可以重复使用的模板，下面有一个新建document(views/documents/new.jade)模板：

```jade
h2 New Document
form(method='post', action='/documents')
 !=partial('documents/fields', { locals: { d: d } })
```

这个引用的渲染是调用了partial(template file name, options)实现的。输出没有进行转义，因为我们想要得到的是HTML标签，对定义好的字段不进行转义还是很安全的。

## 创建和编辑表单

在创建令人印象深刻并且敬畏的Ajax界面之前，让我们首先做一些简单的模板。我们的REST API定义了create和update方法，因此我们应该创建相应的new和edit模板。

我通常将这样的表单拆分为三个模板，其中有一个是可以多次引用的表单字段模板，其他两个分别是创建和编辑模板，其中包含了除字段以外的所有HTML代码。

创建表单在前面已经演示了，编辑表单的模板views/documents/edit.jade应该如下所示：

```jade
h2 Edit Document
form(method='post', action='/documents/' + d.id)
 input(name='document[id]', value=d.id, type='hidden')
 input(name='_method', value='PUT', type='hidden')
 !=partial('documents/fields', { locals: { d: d } })
```

这跟创建表单一样，但是添加了一个隐藏的字段，_method字段允许我们使用POST方法将表单提交给put路由处理，在前面的教程里我们已经创建了相应的RESTful API。

字段模板引用的views/partials/documents/fields.jade也非常简单：


```jade
div
 label Title:
  input(name='document[title]', value=d.title || '')
div
 label Note:
  textarea(name='document[data]')
   =d.data || ''
div
 input(type='submit', value='Save')
```
 

到现在我们应该对Jade有一些感觉了，我虽然不是一个haml/Jade的粉丝，但是也不得不说这些例子的语法非常简洁。

新建和编辑的后台方法

所有新建和编辑的服务端方法都是把document数据取出并渲染成表单：

```js
app.get('/documents/:id.:format?/edit', function(req, res) {
  Document.findById(req.params.id, function(d) {
    res.render('documents/edit.jade', {
      locals: { d: d }
    });
  });
});

app.get('/documents/new', function(req, res) {
  res.render('documents/new.jade', {
    locals: { d: new Document() }
  });
});
```
 

新建方法其实就创建一个空的Document并将其作为变量传递给表单模板。

## Mongo ID

你注意到模板里引用了一个d.id的变量了吗？Mongoose会自动创建一个_id的字段，数据类型为ObjectID，这个在web应用中并不是很方便，所以我用了一个getter方法把它转成了字符串并添加到models.js中：

```js
getters: {
  id: function() {
    return this._id.toHexString();
  }
}
```
 

使用toHexString我们得到了一个使用起来方便一点的ID，比如像4cd733fb20a558cee5000001

## 更新和删除

更新和删除方法都是先取出document数据然后调用save或者remove方法，基本的模式如下：

```js
app.put('/documents/:id.:format?', function(req, res) {
  // Load the document
  Document.findById(req.body.document.id, function(d) {
    // Do something with it
    d.title = req.body.document.title;
    d.data = req.body.document.data;

    // Persist the changes
    d.save(function() {
      // Respond according to the request format
      switch (req.params.format) {
        case 'json':
          res.send(d.__doc);
          break;
        default:
          res.redirect('/documents');
      }
    });
  });
});
```
 

删除的代码基本相同，除了使用remove调用替换save。

## 删除JavaScript

Express使用del方法的方式比较古怪，因为需要使用post提交参数中包含一个隐藏的_method="delete"参数，很多框架都会在客户端的JavaScript来实现。

在教程的第一部分我说过，会使用jQuery。可以通过编辑layout.jade模板来将jQuery库包含在所有页面中：

```html
!!!
html
 head
  title= 'Nodepad'
  link(rel='stylesheet', href='/stylesheets/style.css')
  script(type='text/javascript',src='https://ajax.googleapis.com/ajax/libs/jquery/1.4.4/jquery.min.js')
 body!= body
  script(type='text/javascript', src='/javascripts/application.js')
```
 

这也将我们需要的JavaScript文件包含在了末尾。Express已经将静态文件放在了一个public目录中以便客户端访问。

客户端的JavaScript需要做如下操作：

* 使用confirm()确认用户真的想删除；
* 动态插入一个表单包含一个名为_method的参数值为delete；
* 提交这个表单。
当然，这些操作使用jQuery将非常容易实现，代码如下：

```js
$('.destroy').live('click', function(e) {
  e.preventDefault();
  if (confirm('Are you sure you want to delete that item?')) {
    var element = $(this),
    form = $('<form></form>');
    form
     .attr({
        method: 'POST',
        action: element.attr('href')
      })
     .hide()
     .append('<input type="hidden" />')
     .find('input')
     .attr({
       'name': '_method',
       'value': 'delete'
      })
     .end()
     .submit();
  }
});
```
 

其中使用了live事件绑定执行所有的代码，因此我们并不需要编写嵌套在HTML中的JavaScript。

## 主页

我已经将默认的动作重定向到/douments，并且document的索引动作执行如下：

```jade
h1 Your Documents

p
 a(class='button', href='/documents/new') + New Document
ul
 - for (var d in documents)
  li
   a(class='button', href='/documents/' + documents[d].id + '/edit') Edit
   a(class='button destroy', href='/documents/' + documents[d].id) Delete
   a(href='/documents/' + documents[d].id)
    =documents[d].title
```
 

这是一个在Jade里面使用迭代的例子，一个可行的办法是使用引用模板，但是这是一个展示在Jade模板中如何控制块儿的很好的例子。

## 结论

我们现在有了一个基本能够工作的nodepad了，代码提交：[commit f66fdb5](https://github.com/alexyoung/nodepad/commit/f66fdb5c3bebdf693f62884ffc06a40b93328bb5)

在我们继续下面的教程以前，你可以自己加一些你认为非常酷的小功能。