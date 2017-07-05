---
layout: post
title:  "Using Oracle GoldenGate with Amazon RDS"
date:   2016-12-12 08:14:41 +0800
categories: AWS Cloud
---
Oracle GoldenGate is used to collect, replicate, and manage transactional data between databases. It is a log-based change data capture (CDC) and replication software package used with Oracle databases for online transaction processing (OLTP) systems. GoldenGate creates trail files that contain the most recent changed data from the source database and then pushes these files to the target database. You can use Oracle GoldenGate with Amazon RDS for Active-Active database replication, zero-downtime migration and upgrades, disaster recovery, data protection, and in-region and cross-region replication.

We can use AWS standard [DOCs](http://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Appendix.OracleGoldenGate.html) but this page will discuss how to setup this archeticture in Beijing region which there will some diffrent form other AWS global region.

There will have some import poit in following contents:

* License model: Bring-your-own-licence* RDS Oracle Edition Support: SE1 / SE / EE* Oracle Database Version: 11.2.0.3 / 11.2.0.4* GoldenGate Version: 11.2.1* Support Across Oracle but no prevent heterogeneous db* Support TDE but should encrypted pipeline* DDL is not CURRENTLY support

There may be 5 scenarios in the OGG with RDS archeticture:

Scenario 1: An on-premises Oracle source database and on-premises Oracle GoldenGate hub, that provides data to a target Amazon RDS DB instance.

![](/assets/img/ogg-aws/1.png)

Scenario 2: An on-premises Oracle database that acts as the source database, connected to an Amazon EC2 instance hub that provides data to a target Amazon RDS DB instance.

![](/assets/img/ogg-aws/2.png)

Scenario 3: An Oracle database on an Amazon RDS DB instance that acts as the source database, connected to an Amazon EC2 instance hub that provides data to a target Amazon RDS DB instance.

![](/assets/img/ogg-aws/3.png)

Scenario 4: An Oracle database on an Amazon EC2 instance that acts as the source database, connected to an Amazon EC2 instance hub that provides data to a target Amazon RDS DB instance.

![](/assets/img/ogg-aws/4.png)

Scenario 5: An Oracle database on an Amazon RDS DB instance connected to an Amazon EC2 instance hub in the same region, connected to an Amazon EC2 instance hub in a different region that provides data to the target Amazon RDS DB instance in the same region as the second EC2 instance hub.

![](/assets/img/ogg-aws/5.png)

This page will use scenario 2 to test the OGG with AWS architecture in Beijing Region.

## Setting Up an Oracle GoldenGate Hub On EC2### Import Oracle Linux AMI
Firstly we will setup an Oracle GoldenGate instance in EC2 and Oracle suggest to use Oracle Linux to install the OGG, so we need a Oracle Linux AMI. There is Oracle Linux 6.7 AMI in the Marketplace in Global Region but Beijing Region can't access the Marketplace the we need move this AMI to Bejing Region. The detail steps bellow:

* Create a EC2 instance from Marketplace with hold the Oracle Linux 6.7 AMI in any Global Region
* Stop this Oracle Linux EC2 instance and deattache the root EBS volume
* Create anthor EC2 instance and attache the Oracle Linux root volume to this EC2 instance
* Create a EC2 instance in Beijing Region and create a blank EBS volume which size must be 100G to attach this valume to the EC2
* Login in the Linux OS exist in the Global Region to run:

```bash
sudo dd if=/dev/xvde bs=1M | pv -s 8g | ssh -i your-bjs.pem root@oraclelinux-6-cn-north-1-hostname “sudo dd of=/dev/xvdf bs=1M oflag=direct“
```
> Need to clarify that:
> 
> * *if=/dev/xvde* should be your Global Region EC2 Oracle Linux volume in the OS
> * *oraclelinux-6-cn-north-1-hostname* should be your Beijing Region EC2 public IP
> * *of=/dev/xvdf* should be output EC2 in Beijing Region Linux volume in the OS

* When the command before run successfully you can terminate both the Global Region instances and delete the volumes you have created
* Stop the EC2 instance in China region and create the snapshot for the import EBS volume
* Create an AMI from the snapshot of the imported EBS volume

### Install Oracle DBMS 11g

After import the AMI, we can start a EC2 Oracle Linux 6.7 intance in Beijing region and continuse to install the Oracle DBMS 11g for OGG using.

At this point I have to emphasize the OGG version must be 11.2.0.3 or 11.2.0.4 and the version 11.2.0.0 on [Oracle Technology Network](http://otn.oracle.com) is *not* valid. Another import thing is the DBMS must be patched with 13328193. We can use silent or GUI installation the DBMS.

### Install and Setup OGG

Down load the OGG 11.2.1 from Oracle [e-delivery](http://edelivery.oracle.com) website and unzip the source file to to a installation folder named OS environment vairable *$OGG\_HOME*, setup the environment variable *$LD\_LIBERAY\_PATH* to $OGG\_HOME and *$ORACLE_HOME/lib*.

```bash
export OGG_HOME=<Unzip folder>
export LD_LIBERAY_PATH=$OGG_HOME:$ORACLE_HOME/lib
```

After this all the OGG command should change the current folder to $OGG_HOME. Next we should add 3 alias to the tnsnames.ora file:

* EC2 local oracle instance
* Source RDS instance
* Target RDS instance

Run following OGG command to create a sub directories:

```bash
GGSCI> create subdirs
```

Create or update OGG GLOBALS parameter file with the correct Heartbeat table name

```
CheckpointTable oggadm.oggchkpt
```

Configure the mgr.prm file 

```
PORT 8199PurgeOldExtracts ./dirdat/*, UseCheckpoints, MINKEEPDAYS 5
```

and start the *manager*

```
GGSCI> start mgr
```

## Setup Databases for Amazon RDS

When create RDS instance we need configure source and target Oracle DBs

### Source Database Setup

Source DB parameters with RDS Parameter Group:

```
compatible = 11.2.0.4ENABLE_GOLDENGATE_REPLICATION = True
```

Next we should setup the retention period for archived REDO logs in both source and target instances:

```sql
SQL> exec rdsadmin.rdsadmin_util.set_configuration('archivelog retention hours',24);
```
> Please keep in mind that the source instance downtime may cause some communication and networking issues. And if the retention period set too small the OGG may have OGG-02028 error.

We can also use following SQL to confirm enough storage:

```sql
SQL> select sum(blocks * block_size) bytes from v$archived_log where next_time>=sysdate-X/24 and dest_id=1;
```
Create a OGG DB user account

```sql
SQL> CREATE tablespace admin;
SQL> CREATE USER oggadm  IDENTIFIED BY “welcome1"  default tablespace ADMIN temporary tablespace TEMP;```Grant privileges to OGG user

```sql
grant create session, alter session to oggadm;grant resource to oggadm;grant select any dictionary to oggadm;grant flashback any table to oggadm;grant select any table to oggadm;grant select_catalog_role to root with admin option;exec RDSADMIN.RDSADMIN_UTIL.GRANT_SYS_OBJECT('DBA_CLUSTERS', 'OGGADM');grant execute on dbms_flashback to oggadm;grant select on SYS.v_$database to oggadm;grant alter any table to oggadm;EXEC DBMS_GOLDENGATE_AUTH.GRANT_ADMIN_PRIVILEGE(grantee=>'OGGADM',privilege_type=>'capture',grant_select_privileges=>true, do_grants=>TRUE);
```

### Target Database Setup

Also create the Parameter Goup and OGG DB user account just like the source RDS. Grant privileges to OGG user in the target database.

```sql
alter user oggadm quota unlimited on ADMIN;alter user oggadm quota unlimited on ADMIN_IDX;grant create session to oggadm;grant alter session to oggadm;grant CREATE CLUSTER to oggadm;grant CREATE INDEXTYPE to oggadm;grant CREATE OPERATOR to oggadm;grant CREATE PROCEDURE to oggadm;grant CREATE SEQUENCE to oggadm;grant CREATE TABLE to oggadm;grant CREATE TRIGGER to oggadm;grant CREATE TYPE to oggadm;grant select any dictionary to oggadm;grant create any table to oggadm;grant alter any table to oggadm;grant lock any table to oggadm;grant select any table to oggadm;grant insert any table to oggadm;grant update any table to oggadm;grant delete any table to oggadm;
EXEC DBMS_GOLDENGATE_AUTH.GRANT_ADMIN_PRIVILEGE(grantee=>'oggadm',privilege_type=>'apply',grant_select_privileges=>true, do_grants=>TRUE);
```

## OGG Extract and Replicat Configuration

We have a overview the OGG data flow process

![](/assets/img/ogg-aws/6.png)

Configure Extract parameter file

```
EXTRACT E1SETENV (ORACLE_SID=ORCL)SETENV (NLSLANG=AL32UTF8)USERID oggadm@SOURCE, PASSWORD XXXXXXEXTTRAIL dirdat/abIGNOREREPLICATESGETAPPLOPSTRANLOGOPTIONS EXCLUDEUSER OGGADMTABLE HR.TEST;
```

Add a checkpoint table for source DB

```bash
GGSCI> dblogin userid oggadm@sourceGGSCI> add checkpointtable
```

Turn on supplement logging for DB table

```bash
GGSCI> add trandata hr.test
```

Enable the Extract process

```bash
GGSCI> add extract e1 tranlog, INTERATED tranlog, begin nowGGSCI> add extrail dirdat/ab extract e1, MEGABYTES 100M
```

Register Extract

```bash
GGSCI> register EXTRACT e1, DATABASE
```

Start Extract

```bash
GGSCI> start e1
```

Configure Replicat parameter file

```
REPLICAT R1SETENV (ORACLE_SID=ORCL)SETENV (NLSLANG=AL32UTF8)USERID oggadm@TARGET, password XXXXXXASSUMETARGETDEFS MAP HR.TEST, TARGET HR.TEST; 
```

Add a checkpoint table for target DB

```bash
GGSCI> dblogin userid oggadm@targetGGSCI> add checkpointtable oggadm.oggchkpt
```

Enable Replicat

```bash
GGSCI> add replicat r1 EXTTRAIL dirdat/ab CHECKPOINTTABLE oggadm.oggchkpt
```

Start Replicat

```bash
GGSCI> start r1
```

## Conclusion

By now we have completed the configuration of OGG for RDS architecture and you can check the source and target DBs data has the same data.