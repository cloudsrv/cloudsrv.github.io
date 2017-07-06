---
layout: post
title:  "Oracle RDS Testing Single AZ vs. Multi-AZ"
date:   2016-12-12 08:14:41 +0800
categories: AWS Cloud
---
These days have some testing on AWS Oracle RDS in Beijing region for understand the influence of the Multi-AZ feature. Following are the recording:

* RDS Setup:

![](/assets/img/oracle-rds-multiaz/1.png)

* [SwingBench](http://dominicgiles.com/swingbench.html) Setup:

![](/assets/img/oracle-rds-multiaz/2.png)

![](/assets/img/oracle-rds-multiaz/3.png)

* Testing Result:

![](/assets/img/oracle-rds-multiaz/4.png)

![](/assets/img/oracle-rds-multiaz/5.png)

![](/assets/img/oracle-rds-multiaz/6.png)

* Multi-AZ setup testing result:

![](/assets/img/oracle-rds-multiaz/7.png)

![](/assets/img/oracle-rds-multiaz/8.png)

![](/assets/img/oracle-rds-multiaz/9.png)

![](/assets/img/oracle-rds-multiaz/10.png)

![](/assets/img/oracle-rds-multiaz/11.png)

* RDS Monitor

![](/assets/img/oracle-rds-multiaz/12.png)

![](/assets/img/oracle-rds-multiaz/13.png)

![](/assets/img/oracle-rds-multiaz/14.png)

Conclusion: Multi-AZ will lost 10%-30% performance for TPS
