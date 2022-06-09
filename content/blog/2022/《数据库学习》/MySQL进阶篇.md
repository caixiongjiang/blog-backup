---
title: "MySQL进阶篇 学习笔记"
date: 2022-06-08T18:18:05+08:00
lastmod: 2022-06-08T18:20:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/title_img.jpg"
description: "MySQL进阶篇主要学习存储引擎，索引，SQL优化，视图，锁，InnoDB引擎底层原理"
tags:
- MySQL
categories:
- 数据库学习
comment : true
---

## MySQL学习目标

* 面试题demo：
  * 什么是事物，以及事务的四大特性？
  * 事务的隔离级别有哪些？MySQL默认是哪几个？
  * 内连接与左外连接的区别是什么？
  * 常用的存储引擎？InnoDB和MyISAM的区别？
  * MySQL默认InnoDB引擎的索引是什么数据结构
  * 如何查看MySQL的执行计划？
  * 索引失效的情况有哪些？
  * 什么是回表查询？
  * 什么是MVCC？
  * MySQL主从复制的原理是什么？
  * 主从复制之后的读写分离如何实现？
  * 数据库的分库分表如何实现？
* 学习目标：
  * 入门阶段：下载和安装mysql，学习并使用SQL语言
  * 进阶阶段：学习事务，存储引擎，索引，SQL优化，锁
  * 高级阶段：学习日志管理，主从复制，读写分离，分库分表（主要用于集群）
* 注：本笔记基于b站最新的2022的mysql免费视频：[https://www.bilibili.com/video/BV1Kr4y1i7ru?spm_id_from=333.999.0.0](https://www.bilibili.com/video/BV1Kr4y1i7ru?spm_id_from=333.999.0.0)



## MySQL进阶篇

### 存储引擎

#### MySQL体系结构

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img40.jpg)

MySQl服务器分为4层：连接层；服务层；引擎层；存储层。

#### 存储引擎简介

存储引擎就是存储数据，建立索引，更新/查询数据等技术的实现方式。存储引擎是基于表的，而不是基于库的，所以存储引擎也可被称为表类型。

* 1.在创建表是，可以指定存储引擎

  ```sql
  create table 表名(
  		字段1 字段1类型 [comment 字段1注释],
    	......
    	字段n 字段n类型 [comment 字段n注释]
  )engine = innodb [comment 表注释];
  ```

* 查看当前数据库支持的存储引擎

  ```sql
  show engines;
  ```

#### 存储引擎特点

* InnoDB

  * 介绍：InnoDB是一种兼顾搞可靠性和高性能的通用存储引擎，在MySQL5.5之后，InnoDB是默认的MySQL存储引擎。

  * 特点：

    DML操作遵循ACID模型，支持`事务`；

    `行级锁`，提高并发访问性能；

    支持`外键`FOREIGN KEY约束，保证数据的完整性和正确性；

  * 文件

    xxx.ibd:xxx代表的是表名，**InnoDB引擎的每张表都会对应这样一个表空间文件**，存储该表的表结构（frm，sdi）、数据和索引。

    参数：innodb_file_per_table

  * 存储结构：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img41.jpg)

* MyISAM

  * 介绍：MyISAM是MySQL早期的默认存储引擎。

  * 特点：

    不支持事务，不支持外键；

    支持表锁，不支持行锁；

    访问速度快。

  * 文件

    xxx.sdi:存储表结构信息

    xxx.MYD:存储数据

    xxx.MYI:存储索引

* Memory

  * 介绍：Memory引擎的表数据时存储在内存中的，由于受到硬件问题，或断电问题的影响，只能将这些表作为临时表或缓存使用。

  * 特点：

    内存存放

    hash索引（默认）

  * 文件

    xxx.sdi：存储表结构信息

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img42.jpg)

#### 存储引擎选择

* InnoDB:是MySQL的默认存储引擎，支持事务，外键。如果应用对**事务的完整性有比较高的要求，在并发条件下要求数据的一致性**，数据操作除了插入和查询之外，还包含很多的更新，删除操作，那么InnoDB存储引擎是比较合适的选择。
* MyISAM：如果应用是以读取和插入操作为主，只有很少的更新和删除操作，**并且对事务的完整性，并发性要求不是很高**，那么选择这个存储引擎是非常合适的。（MongoDB取代）
* MEMORY：将所有数据保存在内存中，**访问速度快，常用于临时表及缓存。MEMORY的缺陷就是对表的大小有限制，太大的表无法缓存在内存中，而且无法保障数据的安全性**。（Redis取代）

### 索引

#### 索引概述

索引（index）是帮助MySQL高效获取数据的数据结构（`有序`）。在数据之外，数据库系统还维护满足特定查找算法的数据结构，这种数据结构以某种方式`指向数据`！

* demo演示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img43.jpg)

*``注:上述二叉树索引结构只是一个示意图，并不是真实的索引结构。``*

* 优缺点

  优点：提高数据检索的效率，降低数据的IO成本；通过索引对数据进行排序，降低CPU的消耗。

  缺点：索引列也需要占空间；索引大大提高了查询效率，但也降低了更新表的速度，对表进行增删改时，需要维护索引的结构，效率更低。

#### 索引结构

MySQL的索引是在存储引擎层实现的，不同的存储引擎有不同的结构，主要包含下面几种：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img44.jpg)

我们平常所说的索引，如果没有特别指明，都是指B+树结构组织的索引！

* 二叉树缺点：顺序插入时，会形成一个链表，查询性能大大降低。大数据量情况下，层级较深，检索速度慢。

* 红黑树：大数据量情况下，层级较深，检索速度慢。

* B-Tree（多路平衡查找树）

  以一颗最大`度数`为5的b-tree为例（每个节点最多存储4个key，5个指针）：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img45.jpg)

  B-Tree的数据结构的构建可视化网站：[https://www.cs.usfca.edu/~galles/visualization/about.html](https://www.cs.usfca.edu/~galles/visualization/about.html)

  **B树构建的总体思路：如果满了就遵循中间节点向上裂变！**

* 经典B+Tree

  以一颗最大度数为4（4阶）的b+tree为例：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img46.jpg)

  **所有的元素都会出现在叶子节点；非叶子结点起到索引的作用，叶子结点用来存放数据；叶子结点形成了一个单向链表。**

* MySQL对B+Tree的优化

  在原来B+Tree的基础上，`增加了一个指向相邻叶子节点的链表指针`，形成了带有顺序指针的B+Tree，提高区间的性能。最终形成了如下的结构：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img47.jpg)

* Hash

  哈希索引就是采用一定的哈希算法，将键值换算成新的哈希值，映射到对应的槽位上，然后存储在hash表中。**如果两个键值，映射到相同的槽位，它们就产生了哈希冲突，可以通过链表来解决。**

  * 哈希索引的特点

    1.Hash索引只能用于对等比较（=，in），不支持范围查询。

    2.无法利用索引完成排序操作

    3.查询效率高，通常只需要一次检索就可以了，效率通常要高于B+tree索引

  * 在MySQL中，支持hash索引的是Memory引擎，而InnoDB中具有自适应hash功能，hash索引是存储引擎根据B+Tree索引在指定条件下自动构建的。

* 为什么InnoDB存储引擎选择B+Tree索引结构？不用其他的树结构？

  * 相对二叉树，层级更少，搜索效率高
  * **对于B-Tree，无论是叶子节点还是非叶子节点，都会保存数据，这样导致一页中存储的键值减少，指针跟着减少，要同样保存大量数据，只能增加树的高度，导致性能降低**
  * 相对于hash索引，B+Tree支持范围匹配及排序操作

#### 索引分类

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img48.jpg)

在InnoDB存储引擎中，根据索引的存储形式，又可以分为以下两种：

* 聚焦索引：将数据和索引放到了一块，索引结构的叶子节点保存了行数据（有且只有一个）

* 二级索引：将数据和索引分开存储，索引结构的叶子节点关联的是对应的主键（可以存在多个）

* 聚集索引选取规则：

  * 如果存在主键，主键索引就是聚集索引
  * 如果不存在主键，将使用第一个唯一索引作为聚集索引
  * 如果表没有主键吗，也没有合适的唯一索引，则InnoDB会自动生成一个rowid作为隐藏的聚集索引

* demo：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img49.jpg)

  通过上图讲解一下下面这句SQL语句的查询过程：

  ```sql
  select * from user where name = 'Arm';
  ```

  * 通过name字段来查询名字为’Arm‘的数据：走二级索引，根据英文字典的排序方法定位到Arm，拿到对应的id值。
  * 拿到id值后回到聚集索引中定位到10的位置，拿到行数据（回表查询）

* 回表查询的概念：先走二级索引找到对应的主键值，在根据主键值再到聚集索引当中，拿到这一行的行数据。

#### 索引语法

* 创建索引：

  ```sql
  create [unique|fulltext] index index_name on 表名(index_col_name,...);#...代表一个索引可以关联多个字段
  ```

* 查看索引：

  ```sql
  show index from 表名;
  ```

* 删除索引：

  ```sql
  drop index index_name on 表名;
  ```

* 按照下列的需求，完成索引的创建

  表的结构如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img50.jpg)

  ```
  1.name字段为姓名字段，该字段的值可能会重复，为该字段创建索引
  2.phone手机号字段的值，是非空，且唯一的，为该字段创建唯一索引
  3.为profession、age、status创建联合索引
  4.为email建立合适的索引来提升查询效率
  ```

  SQL语句如下：

  ```sql
  -- 查询指定表的索引(默认为按行展示，按列展示要加\G)
  show index from tb_user\G;
  -- 实现需求,索引的结构在InnoDB下都默认为B+Tree
  create index idx_user_name on tb_user(name);
  create unique index idx_user_phone on tb_user(phone);
  create index idx_user_pro_age_sta on tb_user(profession,age,status);#联合索引
  create index idx_user_email on tb_user(email);
  -- 删除索引
  drop index idx_user_email on tb_user;
  ```

#### SQL性能分析

对SQL语句进行优化，需要定位的是执行频率较高的SQL语句，所以在性能优化之前，需要先查看各种SQL语句的执行频次。

* SQL执行频率

  MySQL客户端连接成功后，通过show [session|global] status命令可以提供服务器状态信息。通过如下指令，可以查看当前数据库的INSERT、UPDATE、DELETE、SELECT的访问频次：

  ```sql
  show global status like 'Com_______';#global代表查看全局的状态信息，session指的是查看当前会话的查询信息。
  ```

* 慢查询日志

  慢查询日志记录了所有执行时间超过指定参数（long_query_time，单位：秒，默认为10秒）的所有SQL语句的日志。

  ```sql
  -- 查看慢查询日志的开关
  show variables like 'slow_query_log';
  ```

  MySQL的慢查询日志默认没有开启，需要在MySQL的配置文件（/etc/my.cnf）中配置如下信息：

  ```yaml
  # 开启MySQL慢日志查询开关
  slow_query_log=1
  # 设置慢日志的时间为2秒，SQL语句执行时间超过2秒，就会视为慢查询，记录慢查询日志
  long_query_time=2
  ```

  配置完毕后，通过以下指令重新启动MySQL服务器进行测试，查看慢日志文件中记录的信息 /var/lib/mysql/localhost-slow.log （Linux）。

  ```shell
  systemctl restart mysqld
  ```

* profile详情

  show profiles能够在做SQL优化的时候帮助我们了解时间的耗费。通过having_profiling参数能够看到当前MySQL是否支持profile操作：

  ```sql
  select @@have_profiling;
  ```

  默认profiling是关闭的，可以通过set语句在session｜global级别开启profiling：

  ```sql
  -- 查询开关
  select @@profiling;
  -- 打开开关
  set profiling=1;
  ```

  执行一系列的业务SQL的操作，然后通过如下指令查看指令的执行耗时：

  ```sql
  # 查看每一条SQL的基本耗时情况(当前会话中)
  show profiles;
  
  # 查看指定query_id的SQL语句各个阶段的耗时情况
  show profile for query query_id;
  
  # 查看指定query_id的SQL语句的CPU的使用情况
  show profile cpu for query query_id;
  ```

* explain执行计划

  **开发人员通常通过explain来判断SQL语句的性能！**EXPLAIN或者DESC命令获取SELECT语句的信息，包括在SELECT语句执行过程中如何连接和连接的顺序。

  语法：

  ```sql
  # 直接在select语句之间加上关键字explain/desc
  explain select 字段列表 from 表名 where 条件;
  ```

  explain执行计划各字段含义：

  * id：select查询的序列号，表示查询中执行select子句或者是**操作表的顺序**（id相同，执行顺序从上到下；id不同，**值越大，越先执行**）。
  * select_type：表示SELECT的类型，常见的值有SIMPLE（简单表，即不使用表连接或者子查询）、PRIMARY（主查询，即外层的查询）、UNION（UNION中的第二个或者后面的查询语句）、SUBOUERY（SELECT/WHERE之后包含了子查询）等。
  * `type`：表示连接类型，性能由好到差的连接类型为NULL，system，const（根据主键和唯一索引一般会出现const），eq_ref，ref（使用非唯一性的索引进行查询），range，index（遍历整个索引树），all（全表扫描）。
  * `possible_key`：显示可能应用在这张表上的索引，一个或者多个。
  * `key`：实际用到的索引，如果为NULL，则没有使用索引。
  * `key_len`：表示索引中使用的字节数，该值为索引字段最大可能长度，并非实际使用长度，在不损失精确性的前提下，长度越短越好。
  * rows：MySQL认为必须要执行查询的行数，在InnoDB引擎的表中，是一个估计值，可能并不总是准确的。
  * filtered：表示返回结果的行数占需读取行数的百分比，filtered的值越大越好。

#### 索引使用

* 验证索引效率

  在未建立索引之前，执行如下SQL语句，查看SQL的耗时。

  ```sql
  select * from tb_sku where sn = '100000003145001';
  ```

  针对字段创建索引

  ```sql
  create index idx_sku_sn on tb_sku(sn);
  ```

  然后执行相同的SQL语句，再次查看SQL的耗时

  结果从`21s`优化到了`0.01s`！

* 最左前缀法则

  如果索引了多列（联合索引），要遵守最左前缀法则。最左前缀法则指的是查询从索引最左列开始，并且不跳过索引中的列。如果跳跃某一列，``索引将部分失效（后面的字段索引失效）`。

  以 tb_user 表为例，我们先来查看一下之前 tb_user 表所创建的索引。

  ##### ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img51.jpg)

  在 tb_user 表中，有一个联合索引，这个联合索引涉及到三个字段，顺序分别为:profession， age，status。

  对于最左前缀法则指的是，查询时，最左变的列，也就是profession必须存在，否则索引全部失效。 而且中间不能跳过某一列，否则该列后面的字段索引将失效。 接下来，我们来演示几组案例，看一下 具体的执行计划:

  ```sql
  explain select * from tb_user where profession = '软件工程' and age = 31 and status = '0';
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img52.jpg)

  ```sql
  explain select * from tb_user where profession = '软件工程' and age = 31;
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img53.jpg)

  ```sql
  explain select * from tb_user where profession = '软件工程';
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img54.jpg)

  以上的这三组测试中，我们发现只要联合索引最左边的字段 profession存在，索引就会生效，只不 过索引的长度不同。 而且由以上三组测试，我们也可以推测出profession字段索引长度为47、age 字段索引长度为2、status字段索引长度为5。

  ```sql
  explain select * from tb_user where age = 31 and status = '0';
  explain select * from tb_user where status = '0';
  ```

  而通过上面的这两组测试，索引并未生效，原因是因为不满足最左前缀法则，联合索引 最左边的列profession不存在。

  ```sql
  explain select * from tb_user where profession = '软件工程' and status = '0';
  ```

  上述的SQL查询时，存在profession字段，最左边的列是存在的，索引满足最左前缀法则的基本条 件。但是查询时，跳过了age这个列，所以后面的列索引是不会使用的，也就是索引部分生效，所以索 引的长度就是47。

* 范围查询

  联合索引中，出现范围查询(>,<)，范围查询右侧的列索引失效。

  ```sql
  explain select * from tb_user where profession = '软件工程' and age > 30 and status = '0';
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img55.jpg)

  当范围查询使用> 或 < 时，走联合索引了，但是索引的长度为49，就说明范围查询右边的status字 段是没有走索引的。

  ```sql
  explain select * from tb_user where profession = '软件工程' and age >= 30 and status = '0';
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img56.jpg)

  当范围查询使用>= 或 <= 时，走联合索引了，但是索引的长度为54，就说明所有的字段都是走索引 的。

  **所以，在业务允许的情况下，尽可能的使用类似于 >= 或 <= 这类的范围查询，而避免使用 > 或 < 。**

* 索引失效情况

  * 索引列运算

    不要在索引列上进行函数运算操作，`索引将失效`。

  * 字符串不加引号

    字符串类型字段使用时，不加引号，会造成隐式类型转换，`索引将失效`。

  * 模糊查询

    **如果仅仅是尾部模糊匹配，索引不会失效。如果是头部模糊匹配，`索引失效`。**

    ```sql
    explain select * from tb_user where profession like '软件%'; 
    -- 只要是头部匹配，索引都会失效。
    explain select * from tb_user where profession like '%工程'; explain select * from tb_user where profession like '%工%';
    ```

  * or连接条件

    用or分割开的条件， 如果or前的条件中的列有索引，而后面的列中没有索引，那么涉及的索引都不会被用到。`只有两侧都有索引的时候，查询时索引才会生效`。

  * 数据分布影响

    如果MySQL评估使用索引比全表更慢，则不使用索引。

    ```sql
    -- 返回整张表，不走索引
    select * from tb_user where phone >= '17799990000';
    -- 返回4条结果，走索引
    select * from tb_user where phone >= '17799990020';
    ```

    MySQL在查询时，会评估使用索引的效率与走全表扫描的效率，如果走全表扫描更快，则放弃 索引，走全表扫描。 **因为索引是用来索引少量数据的，如果通过索引查询返回大批量的数据，则还不 如走全表扫描来的快，此时索引就会失效。**

    is null 和 is not null的判定：

    条件使用is null和is not null是否走索引不是固定的，取决于当前字段的数据分布情况。`如果过滤出来的数据为少量数据则走索引，否则过滤出来的数据为大量数据，还不如走全表扫描来的快，所以就不走索引`。

* SQL提示

  SQL提示，是优化数据库的一个重要手段，简单来说，就是在SQL语句中加入一些人为的提示来达到优 化操作的目的。

  use index：

  ```sql
  -- 多个索引存在时，人为告诉数据库使用哪个索引（只是建议，mysql内部还会再次进行评估）
  explain select * from tb_user use index(idx_user_pro)where profession = '软件工程'; 
  ```

  ignore index：

  ```sql
  -- 多个索引存在时，人为告诉数据库不使用哪个索引
  explain select * from tb_user ignore index(idx_user_pro)where profession = '软件工程'; 
  ```

  force index：

  ```sql
  -- 多个索引存在时，人为告诉数据库必须使用这个索引
  explain select * from tb_user force index(idx_user_pro)where profession = '软件工程'; 
  ```

* 覆盖索引

  **尽量使用覆盖索引，减少select *。** 那么什么是覆盖索引呢? 覆盖索引是指查询使用了索引，并且`需要返回的列，在该索引中已经全部能够找到 `。

  接下来，我们来看一组SQL的执行计划，看看执行计划的差别，然后再来具体做一个解析。

  ```sql
  explain select id, profession from tb_user where profession = '软件工程' and age = 31 and status = '0' ;
  
  explain select id,profession,age, status from tb_user where profession = '软件工程' and age = 31 and status = '0' ;
  
  explain select id,profession,age, status, name from tb_user where profession = '软 件工程' and age = 31 and status = '0' ;
  
  explain select * from tb_user where profession = '软件工程' and age = 31 and status = '0';
  ```

  执行结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img57.jpg)

  | Extra                   | 含义                                                         |
  | ----------------------- | ------------------------------------------------------------ |
  | using index condition   | 查找使用了索引，但是需要回表查询数据                         |
  | using where;using index | 查找使用了索引，但需要的数据在索引列中能找到，所以不需要回表查询数据 |

  注：*针对主键建立的索引为聚集索引，针对其他字段建立的索引为二级索引/辅助索引；先走辅助索引再走聚集索引就称为回表查询。`而不需要回表查询，一次辅助索引就完成查询就称为覆盖索引`*

  如图，三个查询分别使用了：聚集索引（根据主键查询），覆盖索引（只使用辅助索引），辅助索引&回表查询

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img58.jpg)

* 思考题：

  一张表, 有四个字段(id, username, password, status), 由于数据量大, 需要对

  以下SQL语句进行优化, 该如何进行才是最优方案:

  ```sql
  select id,username,password from tb_user where username =
  
  'itcast';
  ```

  方案：对username和password两个字段建立联合索引，这样可以比username的单列索引少使用一次回表查询，而id为主键，不需要回表。

* 前缀索引

  当字段类型为字符串(varchar，text，longtext等)时，有时候需要索引很长的字符串，这会让 索引变得很大，查询时，浪费大量的磁盘IO，影响查询效率。**此时可以只将字符串的一部分前缀，建立索引，这样可以大大节约索引空间，从而提高索引效率。**

  * 语法：

  ```sql
  create index idx_xxxx on table_name(column(n));
  # n代表取出前n个字符作为索引
  ```

  * 前缀长度：

    可以根据索引的选择性来决定，而选择性是指`不重复的索引值(基数)和数据表的记录总数的比值`，索引选择性越高则查询效率越高， 唯一索引的选择性是1，这是最好的索引选择性，性能也是最好的。

  ```sql
  select count(distinct email) / count(*) from tb_user;
  select count(distinct substring(email,1,5)) / count(*) from tb_user;
  ```

  * 前缀索引查询流程

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img59.jpg)

* 单列索引和联合索引

  单列索引：即一个索引只包含单个列。

  联合索引：即一个索引包含了多个列。

  执行一句SQL语句，看看执行情况：

  ```sql
  explain select id,phone,name from tb_user where phone = '17799990010' and name = '韩信';
  ```

  phone、name上都是有单列索引的，但是最终mysql只会选择一个索引，也就是说，只能走一个字段的索引，此时是会回表查询的。

  紧接着，我们再来创建一个phone和name字段的联合索引来查询一下执行计划。

  ```sql
  create unique index idx_user_phone_name on tb_user(phone,name);
  -- 指定联合索引走查询
  explain select id,phone,name from tb_user use index(idx_user_phone_name) where phone = '17799990010' and name  = '韩信';
  ```

  查询时，就走了联合索引，而在联合索引中包含 phone、name的信息，在叶子节点下挂的是对应的主键id，所以查询是无需回表查询的。

  `在业务场景中，如果存在多个查询条件，考虑针对于查询字段建立索引时，建议建立联合索引，而非单列索引。`

  注：*多条件查询时，MySQL优化器会评估哪个字段的索引效率更高，会选择该索引完成本次查询。*

  联合索引的查询结构(联合索引使用得当，即使用覆盖索引可以避免回表查询)，需要注意的是使用联合索引时左边的字段必须存在（最左前缀法则）：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img60.jpg)

#### 索引设计原则

1.针对于数据量较大，且查询比较频繁的表建立索引。

2.针对于常作为查询条件(where)、排序(order by)、分组(group by)操作的字段建立索引。

3.`尽量选择区分度高的列作为索引`，尽量建立唯一索引，区分度越高，使用索引的效率越高。

4.`如果是字符串类型的字段，字段的长度较长`，可以针对于字段的特点，建立前缀索引。

5.尽量使用联合索引，减少单列索引，查询时，联合索引很多时候可以覆盖索引，节省存储空间，避免回表，提高查询效率。

6.要控制索引的数量，索引并不是多多益善，索引越多，维护索引结构的代价也就越大，会影响增删改的效率。

7.`如果索引列不能存储NULL值，请在创建表时使用NOT NULL约束它`。当优化器知道每列是否包含 NULL值时，它可以更好地确定哪个索引最有效地用于查询。

### SQL优化

#### 插入数据

* insert优化：

  * 批量插入

    ```sql
    -- 每次不要超过1000条
    insert into tb_test values(1,'Tom'),(2,'Cat'),(3,'Jerry');
    ```

  * 手动提交事务

    ```sql
    start transaction;
    insert into tb_test values(1,'Tom'),(2,'Cat'),(3,'Jerry');
    insert into tb_test values(4,'Tom'),(5,'Cat'),(6,'Jerry');
    insert into tb_test values(7,'Tom'),(8,'Cat'),(9,'Jerry');
    commit;
    ```

  * 主键顺序插入：主键顺序插入性能更高。

* 大批量数据插入

  如果一次性需要插入大批量的数据，使用insert语句插入性能较低，此时可以使用MySQL数据库提供的load指令进行插入，操作如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img61.jpg)

  ```sql
  -- 客户端连接服务端时，加上参数 -–local-infile 
  mysql –-local-infile -u root -p
  -- 设置全局参数local_infile为1，开启从本地加载文件导入数据的开关 
  set global local_infile = 1;
  -- 执行load指令将准备好的数据，加载到表结构中
  load data local infile '/root/sql1.log' into table tb_user fields terminated by ',' lines terminated by '\n';
  ```

  *主键顺序插入高于乱序插入*

#### 主键优化

* 数据组织方式

  在InnoDB存储引擎中，表数据都是根据主键顺序组织存放的，这种存储方式的表称为`索引组织表`(index organized table IOT)。每一个主键对应一个行数据。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img62.jpg)

  在InnoDB引擎中，数据行是记录在逻辑结构 page 页中的，而每一个页的大小是固定的，默认16K。 那也就意味着， 一个页中所存储的行也是有限的，`如果插入的数据行row在该页存储不小，将会存储 到下一个页中，页与页之间会通过指针连接。`

* 表结构插入数据的流程

  * 页分裂

    页可以为空，也可以填充一半，也可以填充100%。每个页包含了2-N行数据(如果一行数据过大，会行溢出)，根据主键排列。

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img63.jpg)

    主键乱序插入的情况：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img64.jpg)

  * 页合并

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img65.jpg)

    MERGE_THRESHOLD:合并页的阈值，可以自己设置，在创建表或者创建索引时指定。

* 主键的设计原则

  * 满足业务需求的情况下，尽量降低主键的长度。（降低二级索引的空间成本，每个二级索引的叶子节点挂的就是主键）
  * 插入数据时，尽量选择顺序插入，选择使用AUTO_INCREMENT自增主键。
  * 尽量不要使用[UUID](https://www.csdn.net/tags/MtjaggysNDkwMDYtYmxvZwO0O0OO0O0O.html)做主键或者其他自然主键，如身份证号。（uuid和身份证号长度都会比较长）
  * 业务操作时，避免对主键的修改。

#### order by优化

* Using filesort：通过表的索引或全表扫描，读取满足条件的数据行，然后在排序缓冲区sort buffer中完成排序操作，`所有不是通过索引直接返回排序结果的排序都叫 FileSort 排序`。

* Using index：通过有序索引顺序扫描直接返回有序数据，这种情况即为 using index，不需要 额外排序，操作效率高。

  对于以上的两种排序方式，Using index的性能高，而Using filesort的性能低，我们在优化排序 操作时，尽量要优化为 Using index。

* Demo:表结构如下

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img66.jpg)

  ```sql
  # 没有创建索引时，根据age，phone进行排序
  explain select id, age, phone from tb_user order by age, phone\G;
  ```

  结果如下，效率较低：

  ```
             id: 1
    select_type: SIMPLE
          table: tb_user
     partitions: NULL
           type: ALL
  possible_keys: NULL
            key: NULL
        key_len: NULL
            ref: NULL
           rows: 24
       filtered: 100.00
          Extra: Using filesort
  ```

  为字段创建索引：

  ```sql
  # 创建联合索引
  create index idx_user_age_phone_aa on tb_user(age,phone);
  # 创建索引后，根据age，phone进行升序排序
  explain select id, age, phone from tb_user order by age,phone;
  # 创建索引后，根据age，phone进行降序排序
  explain select id, age, phone from tb_user order by age desc,phone desc;
  ```

  这样搜索就会走索引。

  但如果是完成下列操作

  ```sql
  # 按照年龄升序，年龄相同时，按照phone降序
  explain select id, age, phone from tb_user order by age asc, phone desc\G;
  ```

  结果为：

  ```sql
           id: 1
    select_type: SIMPLE
          table: tb_user
     partitions: NULL
           type: index
  possible_keys: NULL
            key: idx_user_age_phone_aa
        key_len: 48
            ref: NULL
           rows: 24
       filtered: 100.00
          Extra: Using index; Using filesort
         # 这里即出现Using index又出现Using filesort是因为默认排序为升序，在age相同时，需要重新对phone进行排序，所以出现filesort
  ```

  解决这个问题需要再创建一个索引：

  ```sql
  create index idx_user_age_pho_ad on tb_user(age asc, phone desc);
  ```

  这样上述语句也纯走索引。该索引的叶子节点的结构如下图所示：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img67.jpg)

  *注：如果同为降序，只需要反向走第一个索引就可以达到目的，先降序后升序，只需要反向走第二个索引也可以达到目的。`但这些的前提一定是你使用了覆盖索引`*

* order by使用原则：

  * 根据排序字段建立合适的索引，多字段排序时，也遵循最左前缀法则。
  * 尽量使用覆盖索引。
  * 多字段排序, 一个升序一个降序，此时需要注意联合索引在创建时的规则(ASC/DESC)。
  * 如果不可避免的出现filesort，大数据量排序时，可以适当增大排序缓冲区大小 sort_buffer_size(默认256k)。

#### group by优化

```sql
-- 执行分组操作，根据profession字段分组（没有建立索引，使用临时表）
explain select profession , count(*) from tb_user group by profession ; # extra：Using temperory
-- 创建索引
create index idx_user_pro_age_sta on tb_user(profession , age , status);
-- 再次执行分组操作
explain select profession , count(*) from tb_user group by profession; # extra:Using index
-- 根据年龄分组
explain select age,count(*) from tb_user group by age; # extra:Using index; Using temporary
```

* group by使用原则：
  * 在分组操作时，可以通过索引来提高效率。
  * 分组操作时，索引的使用也是满足最左前缀法则的。

#### limit优化

在数据量比较大时，如果进行limit分页查询，在查询时，越往后，分页查询效率越低。

通过测试我们会看到，越往后，分页查询效率越低，这就是分页查询的问题所在。 因为，当在进行分页查询时，`如果执行 limit 2000000,10 ，此时需要MySQL排序前2000010记录`，仅仅返回 2000000 - 2000010 的记录，其他记录丢弃，查询排序的代价非常大 。

优化思路：一般分页查询时，通过创建`覆盖索引` 能够比较好地提高性能，可以通过覆盖索引加子查询形式进行优化。

```sql
explain select * from tb_sku t , (select id from tb_sku order by id limit 2000000,10) a where t.id = a.id;
```

#### count优化

执行下列语句

```sql
-- 查询总数据量
explain select count(*) from tb_user;
```

* MyISAM引擎把一个表的总行数存在了磁盘上，因此执行`count(*)`的时候直接返回这个数，效率非常高(没有where条件的时候)。
* InnoDB引擎就比较麻烦了，它执行`count(*)`的时候，需要把数据一行一行从引擎里面读出来，然后累积计数。

**优化思路：自己计数。**

* count的几种用法

  * count() 是一个聚合函数，对于返回的结果集，一行行地判断，`如果count函数的参数不是NULL，累计值就加 1`，否则不加，最后返回累计值。

  * 用法：count(*)，count(主键)，count(字段)，count(1)

  * 性能分析

    | count用法   | 含义                                                         |
    | ----------- | ------------------------------------------------------------ |
    | count(主键) | InnoDB 引擎会遍历整张表，把每一行的 主键id 值都取出来，返回给服务层。 服务层拿到主键后，直接按行进行累加(主键不可能为null) |
    | count(字段) | 没有not null 约束 : InnoDB 引擎会遍历整张表把每一行的字段值都取出 来，返回给服务层，服务层判断是否为null，不为null，计数累加。<br/> 有not null 约束:InnoDB 引擎会遍历整张表把每一行的字段值都取出来，返 回给服务层，直接按行进行累加。 |
    | count(1)    | InnoDB 引擎遍历整张表，但不取值。服务层对于返回的每一行，放一个数字“1” 进去，直接按行进行累加。 |
    | count(*)    | InnoDB引擎并不会把全部字段取出来，而是专门做了优化，不取值，服务层直接 按行进行累加。 |

     **按照效率排序的话，count(字段) < count(主键 id) < count(1) ≈ `count(*)`， 所以尽量使用 `count(*)`。**

#### update优化

我们主要需要注意一下update语句执行时的注意事项。

```sql
update course set name = 'javaEE' where id = 1;
```

当我们在执行删除的SQL语句时，`会锁定id为1这一行的数据(行锁)`，然后事务提交之后，行锁释放。

但是当我们在执行如下SQL时。

```sql
update course set name = 'SpringBoot' where name = 'PHP';
```

当我们开启多个事务，在执行上述的SQL时，`name字段没有索引，我们发现行锁升级为了表锁。` 导致该update语句的性能大大降低。

**InnoDB的行锁是针对索引加的锁，不是针对记录加的锁 ,并且该索引不能失效，否则会从行锁升级为表锁。所以我们使用update更新要对有索引的字段进行更改，否则并发性能就会降低。**

### 视图/存储过程/触发器

#### 视图

视图(View)是一种虚拟存在的表。视图中的数据并不在数据库中实际存在，行和列数据来自定义视 图的查询中使用的表，并且是在使用视图时动态生成的。

通俗的讲，视图只保存了查询的SQL逻辑，不保存查询结果。所以我们在创建视图的时候，主要的工作就落在创建这条SQL查询语句上。

* 创建

  ```sql
  create or replace view 视图名称[(列表名称)] as select语句 [with[cascaded|local]check option]
  ```

* 查询

  ```sql
  -- 查看创建视图语句
  show create view 视图名称;
  -- 查看视图数据 ...表示条件
  select * from 视图名称...;
  ```

* 修改

  ```sql
  -- 方式1
  create or replace view 视图名称[(列表名称)] as select语句 [with[cascaded|local]check option]
  -- 方式2
  alter view 视图名称[(列表名称)] as select语句 [with[cascaded|local]check option]
  ```

* 删除

  ```sql
  drop view [if exists] 视图名称[,视图名称]...;
  ```

* 检查选项

  当使用WITH CHECK OPTION子句创建视图时，MySQL会`通过视图检查正在更改的每个行`，例如插入，更新，删除，以使其符合视图的定义。 MySQL允许基于另一个视图创建视图，它还会`检查依赖视图中的规则以保持一致性`。为了确定检查的范围，mysql提供了两个选项: CASCADED 和 LOCAL ，默认值为 CASCADED 。

  * CASCADED代表级联，用一个示意图代表如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img68.jpg)

  加了检查选项，在插入时会检查当前视图是否满足规则，而且会把规则一并带给自己的`依赖视图`，也就是将视图2的规则一并导入视图1！**注意在视图上进行增删改数据其实是在基表中进行增删改。**

  * LOCAL代表检查选项只在当前层有效，不会向上传递。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img69.jpg)

* 检查选项总结：

  * 检查选项分为两种，一个为CASCADED，一个为LOCAL。
  * 对视图插入数据时，都要向上检查依赖视图是否有检查选项。
  * **CASCADED会向自己的依赖视图传递检查选项，而LOCAL则只在当前视图起作用，不会向上传递。**

* 视图的更新

  要使视图可更新，`视图中的行与基础表的行之间必须存在一对一的关系`，如果视图包含以下任何一项不可更新：

  * 聚合函数或窗口函数(SUM()、 MIN()、 MAX()、 COUNT()等) 
  * DISTINCT
  * GROUP BY
  * HAVING
  * UNION 或者 UNION ALL

* 作用

  * 简单

    视图不仅可以简化用户对数据的理解，也可以简化他们的操作。那些被经常使用的查询可以被定义为视图(类似于封装的思想)，从而使得用户不必为以后的操作每次指定全部的条件。

  * 安全

    数据库可以授权，但不能授权到数据库特定行和特定的列上。通过视图用户只能查询和修改他们所能见到的数据。

  * 数据独立

    视图可以帮助用户屏蔽真实表结构变化带来的影响。

* 根据如下需求，定义视图

  1.为了保证数据库表的安全性，开发人员在操作tb_user表时，只能看到的用户的基本字段，屏蔽手机号和邮箱两个字段。

  2.查询每个学生所选修的课程(三张表联查)，这个功能在很多的业务中都有使用到，为了简化操作，定义一个视图。

  ```sql
  -- 案例
  -- 1.为了保证数据库表的安全性，开发人员在操作tb_user表时，只能看到的用户的基本字段，屏蔽手机号和邮箱两个字段。
  create view tb_user_view as select id, name, profession, age, gender,status,createtime from tb_user;
  
  select * from tb_user_view;
  -- 2.查询每个学生所选修的课程(三张表联查)，这个功能在很多的业务中都有使用到，为了简化操作，定义一个视图。
  create view tb_stu_course_view as select s.name student_name, s.no student_no, c.name course_name from student s, course c, student_course sc where s.id = sc.studentid and sc.courseid = c.id;
  ```

#### 存储过程

存储过程是事先经过编译并存储在数据库中的一段 SQL 语句的集合，调用存储过程可以简化应用开发人员的很多工作，减少数据在数据库和应用服务器之间的传输，对于提高数据处理的效率是有好处的。(学过编程的这部分应该比较好理解)

存储过程思想上很简单，就是数据库 SQL 语言层面的代码封装与重用。

* 特点

  * 封装，复用
  * 可以接收参数，也可以返回数据
  * 减少网络交互，效率提升

* 创建

  ```sql
  create procedure 存储过程名称([参数列表])
  begin
  			--SQL语句
  end;
  ```

  **注意：在命令中，执行创建存储过程的SQL时，需要通过关键字delimiter指定SQL语句的结束符。**

* 调用

  ```sql
  CALL 名称([参数]);
  ```

* 查看

  ```sql
  -- 查询指定数据库的存储过程及状态信息
  select * from information_schema.routines where routine_schema = '数据库名称';
  -- 查询某个存储过程的定义
  show create procedure 存储过程名称;
  ```

* 删除

  ```sql
  drop procedure [if exists] 存储过程名称;
  ```

* 变量

  （1）系统变量：系统变量是MySQL服务器提供，不是用户定义的，属于服务器层面。分为全局变量(GLOBAL)、会话变量(SESSION)。

  * 查看系统变量

    ```sql
    -- 查看所有系统变量
    show [session|global] variables;
    -- 可以通过like模糊匹配方式查找变量
    show [session|global] variables like '...';
    -- 查看置顶变量的值
    select @@[session|global.]系统变量名;
    ```

  * 设置系统变量

    ```sql
    set [session|global] 系统变量名 = 值;
    set @@[session|global.]系统变量名 = 值;
    ```

    **注意：如果没有指定SESSION/GLOBAL，默认是SESSION，会话变量。mysql服务重新启动之后，所设置的全局参数会失效，要想不失效，可以在/etc/my.cnf中配置。**

  （2）用户自定义变量：用户定义变量是用户根据需要自己定义的变量，用户变量不用提前声明，在用的时候直接用 `"@变量名" `使用就可以。其作用域为当前连接（当前会话）。

  * 赋值

    ```sql
    -- 推荐使用:=进行赋值，因为判断符号也是=
    set @var_name1 = expr [,@var_name2 = expr]...;
    set @var_name1 := expr [,@var_name2 := expr]...;
    ```

    ```sql
    select @var_name1 := expr [,@var_name2 := expr]...;
    -- 将表中查询出来的数据赋给自定义的变量
    select 字段名 into @var_name from 表名;
    ```

  * 使用

    ```sql
    select @var_name;
    ```

    注意：用户定义的变量无需对其进行声明或者初始化，只不过获取到的值为NULL。

  （3）局部变量：局部变量是根据需要定义的在局部生效的变量，访问之前，`需要DECLARE声明`。可用作存储过程内的局部变量和输入参数，`局部变量的范围是在其内声明的BEGIN ... END块`。

  * 声明：

    ```sql
    declare 变量名 变量类型[default ...];
    ```

    变量类型就是数据库字段类型：INT、BEGINT、CHAR、VARCHAR、DATE、TIME等。

  * 赋值

    ```sql
    set 变量名 = 值;
    set 变量名 := 值;
    -- 将表中查询出来的数据赋给局部变量
    select 字段名 into 变量名 from 表名...;
    ```

* if

  语法：

  ```sql
  IF 条件1 THEN 
  			.....
  ELSEIF 条件2 THEN -- 可选
  			.....
  ELSE						 -- 可选 						  
  			.....
  END IF;
  ```

* 参数

  | 类型  | 含义                                         | 备注                                                         |
  | ----- | -------------------------------------------- | ------------------------------------------------------------ |
  | IN    | 该类参数作为输入，也就是需要调用时传入值     | 默认                                                         |
  | OUT   | 该类参数作为输出，也就是该参数可以作为返回值 |                                                              |
  | INOUT | 既可以作为输入参数，也可以作为输出参数       | 传入的参数和返回的参数变量和类型相同（对传入的参数进行修改） |

  用法：

  ```sql
  create procedure 存储过程名称([IN/OUT/INOUT 参数名 参数类型])
  BEGIN
  		--SQL语句
  END;
  ```

* case

  case结构及作用，和我们在基础篇中所讲解的流程控制函数很类似(c++中的switch case)。有两种语法格式:

  * 语法1（值判断）:

  ```sql
  -- 含义: 当case_value的值为 when_value1时，执行statement_list1，当值为 when_value2时， 执行statement_list2， 否则就执行statement_list
  CASE case_value
      WHEN  when_value1  THEN  statement_list1
     [ WHEN  when_value2  THEN  statement_list2] ...
     [ ELSE  statement_list ]
  END CASE;
  ```

  * 语法2（条件判断）:

  ```sql
  -- 含义: 当条件search_condition1成立时，执行statement_list1，当条件search_condition2成 立时，执行statement_list2，否则就执行statement_list
  CASE
    WHEN  search_condition1  THEN  statement_list1
    [WHEN  search_condition2  THEN  statement_list2] ...
    [ELSE  statement_list]
  END CASE;
  ```

* while

  while 循环是有条件的循环控制语句。满足条件后，再执行循环体中的SQL语句。具体语法为:

  ```sql
  -- 先判定条件，如果条件为true，则执行逻辑，否则，不执行逻辑 
  WHILE 条件 DO
  		SQL逻辑... 
  END WHILE;
  ```

* repeat 

  repeat是有条件的循环控制语句, 当满足until声明的条件的时候，则退出循环 。具体语法为:

  ```sql
  -- 先执行一次逻辑，然后判定UNTIL条件是否满足，如果满足，则退出。如果不满足，则继续下一次循环 
  REPEAT
  		SQL逻辑...
  		UNTIL 条件 
  END REPEAT;
  ```

* loop

  LOOP 实现简单的循环，如果不在SQL逻辑中增加退出循环的条件，可以用其来实现简单的死循环。LOOP可以配合一下两个语句使用:

  *  LEAVE :配合循环使用，退出循环。（类似于c++中的break）

  *  ITERATE:必须用在循环中，作用是跳过当前循环剩下的语句，直接进入下一次循环。(类似于c++中的continue)

  ```sql
  [begin_label:]LOOP 
  			SQL逻辑...
  END  LOOP  [end_label];
  ```

  ```sql
  LEAVE label; -- 退出指定标记的循环体 
  ITERATE label; -- 直接进入下一次循环
  ```

  上述语法中出现的 begin_label，end_label，label 指的都是我们所自定义的标记。

* 游标

  游标(CURSOR)是用来`存储查询结果集的数据类型`, 在存储过程和函数中可以使用游标对结果集进行循环的处理。游标的使用包括游标的声明、OPEN、FETCH 和 CLOSE，其语法分别如下。

  * 声明游标：

  ```sql
  DECLARE 游标名称 CURSOR FOR 查询语句;
  ```

  * 打开游标：

  ```sql
  OPEN 游标名称;
  ```

  * 获取游标记录：

  ```sql
  FETCH 游标名称 INTO 变量 [, 变量 ];
  ```

  * 关闭游标：

  ```sql
  CLOSE 游标名称;
  ```

  * 案例：

    根据传入的参数uage，来查询用户表tb_user中，所有的用户年龄小于等于uage的用户姓名 (name)和专业(profession)，并将用户的姓名和专业插入到所创建的一张新表 (id,name,profession)中。

  ```sql
  -- 游标
  -- A.声明游标
  -- B.创建表结构
  -- C.开启游标
  -- D.获取游标中的记录
  -- E.插入数据到新表中
  -- F.关闭游标
  
  create procedure p11(in uage int)
  begin
      -- 注意游标的声明要放在变量声明之后
      declare uname varchar(100);
      declare upro varchar(100);
      declare u_cursor cursor for select name, profession from tb_user where age <= uage;
      -- 声明条件处理程序 (如果满足sql状态码为02000触发退出操作) 02000代表没有数据
      declare exit handler for SQLSTATE '02000' close u_cursor;
      
      drop table if exists tb_user_pro;
      create table if not exists tb_user_pro(
          id int primary key auto_increment,
          name varchar(100),
          profession varchar(100)
      );
  
      open u_cursor;
      while true do
          fetch u_cursor into uname, upro;
          insert into tb_user_pro values (null, uname, upro);
      end while;
      close u_cursor;
  end;
  -- 调用函数
  call p11(40);
  ```

* 条件处理程序

  条件处理程序(Handler)可以用来定义在流程控制结构执行过程中遇到问题时相应的处理步骤。具体语法为:

```sql
DECLARE handler_action HANDLER FOR condition_value [,condition_value]... statement;

handler_action
			CONTINUE: 继续执行当前程序 
			EXIT: 终止执行当前程序
condition_value
			SQLSTATE sqlstate_value: 状态码，如 02000
			SQLWARNING: 所有以01开头的SQLSTATE代码的简写
			NOT FOUND: 所有以02开头的SQLSTATE代码的简写
			SQLEXCEPTION: 所有没有被SQLWARNING 或 NOT FOUND捕获的SQLSTATE代码的简写
```

#### 存储函数

存储函数是有返回值的存储过程，存储函数的参数只能是IN类型的。具体语法如下:

```sql
CREATE FUNCTION 存储函数名称 ([ 参数列表 ]) RETURNS type [characteristic ...]
BEGIN
-- SQL语句
    RETURN ...;
END;

characteristic说明: 
		DETERMINISTIC:相同的输入参数总是产生相同的结果
		NO SQL :不包含 SQL 语句。
		READS SQL DATA:包含读取数据的语句，但不包含写入数据的语句。
```

*在mysql8.0版本中binlog默认是开启的，一旦开启了，mysql就要求在定义存储过程时，需要指定 characteristic特性，否则就会报错。*

#### 触发器

触发器是与表有关的数据库对象，**指在insert/update/delete之前(BEFORE)或之后(AFTER)，触发并执行触发器中定义的SQL语句集合**。触发器的这种特性可以协助应用在数据库端确保数据的完整性, 日志记录 , 数据校验等操作 。

`使用别名OLD和NEW来引用触发器中发生变化的记录内容`，这与其他的数据库是相似的。现在触发器还只支持行级触发，不支持语句级触发。

行级触发器：执行一条update语句，影响几行数据就触发几次。

语句级触发器：执行一条update语句，无论影响几行都只触发一次。

| 触发器类型     | NEW和OLD                                             |
| -------------- | ---------------------------------------------------- |
| INSERT型触发器 | NEW表示将要或者已经增加的数据                        |
| UPDATE型触发器 | OLD表示修改之前的数据，NEW表示将要或已经修改后的数据 |
| DELETE型触发器 | OLD表示将要或者已经删除的数据                        |

* 语法

  * 创建：

  ```sql
  CREATE TRIGGER trigger_name 
  BEFORE/AFTER INSERT/UPDATE/DELETE
  ON tbl_name FOR EACH ROW -- 行级触发器 
  BEGIN
      trigger_stmt;
  END;
  ```

  * 查看

  ```sql
  SHOW TRIGGERS;
  ```

  * 删除

  ```sql
  DROP TRIGGER [schema_name.]trigger_name; 
  -- 如果没有指定 schema_name，默认为当前数据库 。
  ```

* 案例：

  通过触发器记录tb_user表的数据变更日志，将变更日志插入到日志表user_logs中，包含增加，删除，修改；

  ```sql
  -- 通过触发器记录tb_user表的数据变更日志，将变更日志插入到日志表user_logs中，包含增加，删除，修改；
  -- 准备工作 : 日志表 user_logs
  create table user_logs(
    id int(11) not null auto_increment,
    operation varchar(20) not null comment '操作类型, insert/update/delete',
    operate_time datetime not null comment '操作时间',
    operate_id int(11) not null comment '操作的ID',
    operate_params varchar(500) comment '操作参数',
    primary key(`id`)
  )engine=innodb default charset=utf8;
  
  -- 插入数据触发器
  create trigger tb_user_insert_trigger
      after insert on tb_user for each row
  begin
      insert into user_logs(id, operation, operate_time, operate_id, operate_params) values
          (null, 'insert', now(), new.id, concat('插入的数据内容为：id=', new.id,',name=', new.name,
              ', phone=', new.phone, ',profession=', new.profession));
  end;
  
  -- 查看
  show triggers;
  
  -- 删除
  drop trigger tb_user_insert_trigger;
  
  -- 插入数据到tb_user表
  insert into tb_user(id, name, phone, email, profession, age, gender, status, createtime)
  VALUES (25,'二皇子','18809091212','erhuangzi@163.com','软件工程',23,'1','1',now());
  
  
  
  -- 修改数据的触发器
  create trigger tb_user_update_trigger
      after update on tb_user for each row
  begin
      insert into user_logs(id, operation, operate_time, operate_id, operate_params) values
          (null, 'update', now(), new.id, concat('更新之前的的数据内容为：id=', old.id,',name=', old.name,
              ', phone=', old.phone, ',profession=', old.profession, ' | 更新之后的的数据内容为：id=', new.id,',name=', new.name,
              ', phone=', new.phone, ',profession=', new.profession));
  end;
  
  -- 查看
  show triggers;
  -- 更新数据
  update tb_user set age = 32 where id = 23;
  
  update tb_user set profession = '会计' where id <= 5;
  -- 删除数据的触发器
  create trigger tb_user_delete_trigger
      after delete on tb_user for each row
  begin
      insert into user_logs(id, operation, operate_time, operate_id, operate_params) values
          (null, 'delete', now(), old.id, concat('删除之前的的数据内容为：id=', old.id,',name=', old.name,
              ', phone=', old.phone, ',profession=', old.profession));
  end;
  
  -- 删除数据
  delete from tb_user where id = 25;
  ```

  

### 锁

#### 概述

锁是计算机协调多个进程或线程并发访问某一资源的机制。在数据库中，除传统的计算资源(CPU、 RAM、I/O)的争用以外，数据也是一种供许多用户共享的资源。如何保证数据并发访问的一致性、有效性是所有数据库必须解决的一个问题，锁冲突也是影响数据库并发访问性能的一个重要因素。从这个角度来说，锁对数据库而言显得尤其重要，也更加复杂。

MySQL中的锁，按照锁的粒度分，分为以下三类:

* 全局锁:锁定数据库中的所有表。

* 表级锁:每次操作锁住整张表。

* 行级锁:每次操作锁住对应的行数据。

#### 全局锁

全局锁就是对整个数据库实例加锁，加锁后整个实例就处于只读状态，后续的DML的写语句，DDL语句，已经更新操作的事务提交语句都将被阻塞。

其典型的使用场景是做全库的逻辑备份，对所有的表进行锁定，从而获取一致性视图，保证数据的完整性。

数据库在备份时，在全局表被锁定之后，只能进行DQL操作，不能进行DML、DDL操作(会进入阻塞状态)，备份完之后就会解锁。

* 演示：

```sql
-- 加上全局锁
flush tables with read lock;
-- 执行数据备份 用户名，密码，备份存储的sql文件
mysqldump -uroot -p1234 itcast>itcast.sql
-- 解锁
unlock tables;
```

* 特点：

数据库中加全局锁，是一个比较重的操作，存在以下问题：

1.如果在主库上备份，那么在备份期间都不能执行更新，业务基本上就得停摆。 

2.如果在从库上备份，那么在备份期间从库不能执行主库同步过来的二进制日志(binlog)，会导 致主从延迟。

在InnoDB引擎中，我们可以在备份时加上参数 --single-transaction参数来完成不加锁的一致性数据备份。

```sql
-- 底层通过快照读实现
mysqldump --single-transaction -uroot -p123456 itcast>itcast.sql
```

#### 表级锁

表级锁，每次操作锁住整张表。锁定粒度大，发生锁冲突的概率最高，并发度最低。应用在MyISAM、 InnoDB、BDB等存储引擎中。

对于表级锁，主要分为以下三类:

1.表锁

2.元数据锁(meta data lock，MDL)

3.意向锁

* 表锁

  对于表锁，可以分为两类：1.表`共享`读锁 2.表`独占`写锁

  语法：

  ```sql
  # 加锁
  lock tables 表名... read/write;
  # 释放锁
  unlock tables/客户端断开连接;
  ```

  * 表共享读锁的演示：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img70.jpg)

  当前客户端能读，其他客户端也能读；但任何客户端都不可以进行DDL/DML语句。**读锁不会阻塞读，但会阻塞`其他客户端`的写,自己的客户端写会报错。**

  * 表独占写锁演示：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img71.jpg)

  **客户端自己能进行读写，但其他客户端读写处于阻塞状态。**

* 元数据锁

  MDL加锁过程是系统自动控制，无需显式使用，在访问一张表的时候会自动加上。MDL锁主要作用是维 护表元数据的数据一致性，`在表上有活动事务的时候，不可以对元数据进行写入操作`。 **为了避免DML与DDL冲突，保证读写的正确性。**

  这里的元数据，大家可以简单理解为就是一张表的表结构。 也就是说，某一张表涉及到未提交的事务 时，是不能够修改这张表的表结构的。

  在MySQL5.5中引入了MDL，当对一张表进行增删改查的时候，加MDL读锁(共享);当对表结构进行变更操作的时候，加MDL写锁(排他)。

  | 对应SQL                                         | 锁类型                                  | 说明                                               |
  | ----------------------------------------------- | --------------------------------------- | -------------------------------------------------- |
  | lock tables xxx read / write                    | SHARED_READ_ONLY / SHARED_NO_READ_WRITE |                                                    |
  | select 、select ... lock in share mode          | SHARED_READ                             | 与SHARED_READ、 SHARED_WRITE兼容，与 EXCLUSIVE互斥 |
  | insert 、update、 delete、select ... for update | SHARED_WRITE                            | 与SHARED_READ、 SHARED_WRITE兼容，与 EXCLUSIVE互斥 |
  | alter table ...                                 | EXCLUSIVE（排他）                       | 与其他的MDL都互斥                                  |

  查看元数据锁：

  ```sql
  select object_type,object_schema,object_name,lock_type,lock_duration from performance_schema.metadata_locks;
  ```

* 意向锁

  为了避免DML在执行时，加的行锁与表锁的冲突，在InnoDB中引入了意向锁，使得表锁不用检查每行数据是否加锁，使用意向锁来减少表锁的检查。

  意向锁总共有两种：

  1.意向共享锁（IS）：由语句select.. lock in share mode添加。与表锁共享锁 (read)兼容，与表锁排他锁(write)互斥。

  2.意向排他锁（IX）：由insert、update、delete、select... for update 添加。与表锁共享锁(read)及排他锁(write)都互斥，意向锁之间不会互斥。

  

  示例过程：

  1.线程A和B对一张表进行修改，A开启事务，对第三行数据进行修改，此时第三行数据会加上行锁，并加上一个意向锁。

  2.B要对该表加一个表锁，会先检查是否与已有的意向锁兼容（减少了遍历检查行锁的过程）。如果不兼容，会进入阻塞状态，直到左边的事务提交，释放行锁和意向锁，阻塞才会结束，拿到该表的表锁。

  

  可以通过以下SQL，查看意向锁及行锁的加锁情况：

  ```sql
  select object_schema,object_name,index_name,lock_type,lock_mode,lock_data from performance_schema.data_locks;
  ```

  

#### 行级锁

行级锁，每次操作锁住对应的行数据。锁定粒度最小，发生锁冲突的概率最低，并发度最高。应用在 InnoDB存储引擎中。

InnoDB的数据是基于索引组织的，`行锁是通过对索引上的索引项加锁来实现的`，而不是对记录加的锁。对于行级锁，主要分为以下三类:

* 行锁(Record Lock):锁定单个行记录的锁，防止其他事务对此行进行update和delete。在 RC、RR隔离级别下都支持。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img72.jpg)

  * InnoDB实现了以下两种类型的行锁:

    共享锁(S):允许一个事务去读一行，阻止其他事务获得相同数据集的排它锁。

    排他锁(X):允许获取排他锁的事务更新数据，阻止其他事务获得相同数据集的共享锁和排他锁。

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img75.jpg)

    **常见增删改查SQL的行锁类型：**

    | SQL                           | 行锁类型   |                   说明                   |
    | :---------------------------- | ---------- | :--------------------------------------: |
    | INSERT ...                    | 排他锁     |                 自动加锁                 |
    | UPDATE ...                    | 排他锁     |                 自动加锁                 |
    | DELETE ...                    | 排他锁     |                 自动加锁                 |
    | SELECT(正常)                  | 不加任何锁 |                                          |
    | SELECT ... LOCK IN SHARE MODE | 共享锁     | 需要手动在SELECT之后加LOCK IN SHARE MODE |
    | SELECT ... FOR UPDATE         | 排他锁     |     需要手动在SELECT之后加FOR UPDATE     |

  * 行锁-演示

    默认情况下，InnoDB在 REPEATABLE READ事务隔离级别运行，InnoDB使用 next-key 锁进行搜索和索引扫描，以防止幻读。

    1.针对唯一索引进行检索时，对已存在的记录进行等值匹配时，将会自动优化为行锁。

    2.InnoDB的行锁是针对于索引加的锁，不通过索引条件检索数据，那么InnoDB将对表中的所有记录加锁，此时就会升级为表锁。

* 间隙锁(Gap Lock):锁定索引记录间隙(不含该记录)，确保索引记录间隙不变，防止其他事务在这个间隙进行insert，产生幻读。在RR隔离级别下都支持。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img73.jpg)

* 临键锁(Next-Key Lock):行锁和间隙锁组合，同时锁住数据，并锁住数据前面的间隙Gap。 在RR隔离级别下支持。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img74.jpg)

* 临键锁和间隙锁的区别（需要理解）：`如上图如果是范围查询6到16之间的数据，为了防止幻读，会将6到12以及12到16加上间隙锁，字段为12的这一行加上行锁;如果是查询值为12的数据，且该字段索引为非唯一索引，数据可能重复，那么为了防止重复数据插入幻读，原本的两种锁现在便只需要间隙锁了，因为重复的数据只可能在12的左右加入。`

  * 间隙锁/临建锁-演示

    默认情况下，InnoDB在REPEATABLE READ事务隔离级别运行，InnoDB使用 next-key 锁进行搜索和索引扫描，以防止幻读。

    1.索引上的等值查询(唯一索引)，给不存在的记录加锁时, 优化为间隙锁 。

    2.索引上的等值查询(非唯一普通索引)，向右遍历时最后一个值不满足查询需求时，next-key lock 退化为间隙锁。这里主要是为了`防止插入重复的数据时产生幻读现象 `。

    3.索引上的范围查询(唯一索引)--会访问到不满足条件的第一个值为止。

    *注意：间隙锁唯一目的是为了防止其他事务插入间隙。间隙锁可以共存，一个事务采用的间隙锁不会阻止另一个事务在同一间隙上采用间隙锁。*

### InnoDB引擎

#### 逻辑存储结构

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img76.jpg)

* 表空间（ibd文件），一个MySQL实例可以对应多个表空间，用于存储记录，索引等数据。
* 段，分为数据段(Leaf node segment)、索引段(Non-leaf node segment)、回滚段 (Rollback segment)，InnoDB是索引组织表，`数据段就是B+树的叶子节点，索引段即为B+树的非叶子节点`。段用来管理多个Extent(区)。
* 区，表空间的单元结构，每个区的大小为1M。 默认情况下， InnoDB存储引擎页大小为16K， 即一个区中一共有64个连续的页。
* **页，是InnoDB 存储引擎磁盘管理的最小单元**，每个页的大小默认为 16KB。为了保证页的连续性， InnoDB 存储引擎每次从磁盘申请 4-5 个区。
* 行，InnoDB 存储引擎数据是按行进行存放的。
  * Try_id:最后一次事务的id
  * Roll_pointer:每次对某条引记录进行改动时，都会把旧的版本写入到undo日志中，然后这个 隐藏列就相当于一个指针，可以通过它来找到该记录修改前的信息。

#### 架构(理解即可)

MySQL5.5 版本开始，默认使用InnoDB存储引擎，它擅长事务处理，具有崩溃恢复特性，在日常开发 中使用非常广泛。下面是InnoDB架构图，左侧为内存结构，右侧为磁盘结构。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img77.jpg)

* 内存结构

  1). Buffer Pool

  缓冲池 Buffer Pool，是主内存中的一个区域，里面可以缓存磁盘上经常操作的真实数据，在执行增删改查操作时，先操作缓冲池中的数据(若缓冲池没有数据，则从磁盘加载并缓存)，然后再以一定频率刷新到磁盘，从而减少磁盘IO，加快处理速度。

  缓冲池以Page页为单位，底层采用链表数据结构管理Page。根据状态，将Page分为三种类型:

   • free page:空闲page，未被使用。
   • clean page:被使用page，数据没有被修改过。
   • dirty page:脏页，被使用page，数据被修改过，也中数据与磁盘的数据产生了不一致。

  

  2). Change Buffer

  Change Buffer，更改缓冲区(针对于`非唯一二级索引页(普通索引)`)，在执行DML语句时，如果这些数据Page 没有在Buffer Pool中，不会直接操作磁盘，而会将数据变更存在更改缓冲区 Change Buffer 中，在未来数据被读取时，再将数据合并恢复到Buffer Pool中，再将合并后的数据刷新到磁盘中。

  `Change Buffer的意义是什么？`

  与聚集索引不同，二级索引通常是非唯一的，并且以相对随机的顺序插入二级索引。同样，删除和更新可能会影响索引树中不相邻的二级索引页，如果每一次都操作磁盘，会造成大量的磁盘IO。有了ChangeBuffer之后，我们可以`在缓冲池中进行合并处理，减少磁盘IO`。

  

  3). Adaptive Hash Index

  自适应hash索引，用于优化对Buffer Pool数据的查询。MySQL的innoDB引擎中不支持hash索引，但是给我们提供了一个功能就是这个自适应hash索引。因为前面我们讲到过，hash索引在进行等值匹配时，一般性能是要高于B+树的，因为hash索引一般只需要一次IO即可，而B+树，可能需要几次匹配，所以hash索引的效率要高，但是hash索引又不适合做范围查询、模糊匹配等。

  InnoDB存储引擎会监控对表上各索引页的查询，`如果观察到在特定的条件下hash索引可以提升速度， 则建立hash索引，称之为自适应hash索引。`

  **自适应哈希索引，无需人工干预，是系统根据情况自动完成。**

  参数: adaptive_hash_index(默认是开启的)

  

  4). Log Buffer

  Log Buffer:日志缓冲区，用来保存要写入到磁盘中的log日志数据(redo log 、undo log)， 默认大小为 16MB，日志缓冲区的日志会定期刷新到磁盘中。如果需要更新、插入或删除许多行的事务，增加日志缓冲区的大小可以节省磁盘 I/O。

  参数：

  innodb_log_buffer_size:缓冲区大小 

  innodb_flush_log_at_trx_commit:日志刷新到磁盘时机，取值主要包含以下三个:

  * 1: 日志在每次事务提交时写入并刷新到磁盘，默认值。 
  * 0: 每秒将日志写入并刷新到磁盘一次。
  * 2: 日志在每次事务提交后写入，并每秒刷新到磁盘一次。

* 磁盘结构

  1). System Tablespace

  系统表空间是更改缓冲区的存储区域。如果表是在系统表空间而不是每个表文件或通用表空间中创建 的，它也可能包含表和索引数据。(在MySQL5.x版本中还包含InnoDB数据字典、undolog等)

  

  2). File-Per-Table Tablespaces

  如果开启了innodb_file_per_table开关 ，则**每个表的文件表空间包含单个InnoDB表的数据和索引（表空间文件）**，并存储在文件系统上的单个数据文件中。

  开关参数:innodb_file_per_table ，该参数默认开启。

  

  3). General Tablespaces
   通用表空间，需要通过 CREATE TABLESPACE 语法创建通用表空间，在创建表时，可以指定该表空间。
   A. 创建表空间

  ```sql
  CREATE TABLESPACE ts_name ADD DATAFILE 'file_name' ENGINE = engine_name;
  ```

  B. 创建表时指定表空间

  ```sql
  CREATE TABLE xxx ... TABLESPACE ts_name;
  ```

  

  4). Undo Tablespaces 撤销表空间，MySQL实例在初始化时会自动创建两个默认的undo表空间(初始大小16M)，用于存储undo log日志。

  

  5). Temporary Tablespaces
   InnoDB 使用会话临时表空间和全局临时表空间。存储用户创建的临时表等数据。

  

  6). Doublewrite Buffer Files

  双写缓冲区，innoDB引擎将数据页从Buffer Pool刷新到磁盘前，先将数据页写入双写缓冲区文件 中，便于系统异常时恢复数据。

  

  7). Redo Log

  重做日志，是用来实现事务的持久性。该日志文件由两部分组成:重做日志缓冲(redo log buffer)以及重做日志文件(redo log),前者是在内存中，后者在磁盘中。当事务提交之后会把所 有修改信息都会存到该日志中, 用于在刷新脏页到磁盘时,发生错误时, 进行数据恢复使用。

  

* 后台线程

  1). Master Thread 

  核心后台线程，负责调度其他线程，还负责将缓冲池中的数据异步刷新到磁盘中, 保持数据的一致性，还包括脏页的刷新、合并插入缓存、undo页的回收 。

  

  2). IO Thread
  在InnoDB存储引擎中大量使用了AIO(异步IO)来处理IO请求, 这样可以极大地提高数据库的性能，而IO Thread主要负责这些IO请求的回调。

  | 线程类型             | 默认个数 | 职责                         |
  | -------------------- | :------: | ---------------------------- |
  | Read thread          |    4     | 负责读操作                   |
  | Write thread         |    4     | 负责写操作                   |
  | Log thread           |    1     | 负责将日志缓冲区刷新到磁盘   |
  | Insert buffer thread |    1     | 负责将写缓冲区内容刷新到磁盘 |

  我们可以通过以下的这条指令，查看到InnoDB的状态信息，其中就包含IO Thread信息。

  ```sql
  show engine innodb status;
  ```

  

  3). Purge Thread
  主要用于回收事务已经提交了的undo log，在事务提交之后，undo log可能不用了，就用它来回收。

  

  4). Page Cleaner Thread
  协助 Master Thread 刷新脏页到磁盘的线程，它可以减轻 Master Thread 的工作压力，减少阻塞。

#### 事务原理

1). 事务

事务 是一组操作的集合，它是一个不可分割的工作单位，事务会把所有的操作作为一个整体一起向系 统提交或撤销操作请求，即这些操作要么同时成功，要么同时失败。

2). 特性

• 原子性(Atomicity):事务是不可分割的最小操作单元，要么全部成功，要么全部失败。

• 一致性(Consistency):事务完成时，必须使所有的数据都保持一致状态。

• 隔离性(Isolation):数据库系统提供的隔离机制，保证事务在不受外部并发操作影响的独立环 境下运行。

• 持久性(Durability):事务一旦提交或回滚，它对数据库中的数据的改变就是永久的。

那实际上，我们研究事务的原理，就是研究MySQL的InnoDB引擎是如何保证事务的这四大特性的。

而对于这四大特性，实际上分为两个部分。 其中的原子性、一致性、持久化，实际上是由InnoDB中的 两份日志来保证的，一份是redo log日志，一份是undo log日志。 而持久性是通过数据库的锁， 加上MVCC来保证的。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img78.jpg)

* redo log

重做日志，记录的是事务提交时数据页的物理修改，是用来实现事务的**持久性**。

该日志文件由两部分组成:`重做日志缓冲(redo log buffer)以及重做日志文件(redo log file),前者是在内存中，后者在磁盘中。`当事务提交之后会把所有修改信息都存到该日志文件中, 用于在`刷新脏页到磁盘,发生错误时, 进行数据恢复使用`。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img79.jpg)

如上图，一个事务进行了一系列增删改操作。首先，系统会检查缓冲池看看有没有要修改的目标数据，如果没有则从磁盘向缓冲池中刷入要修改和删除的数据，最后redo log buffer会保存数据页的变化并刷入磁盘当中。`如果最后脏页刷入磁盘发生错误时，则可通过磁盘中数据页的变化来恢复数据。`这其实就是`预写日志（WAL，Write-Ahead logging）`的机制。

* undo log

回滚日志，用于记录数据被修改前的信息 , 作用包含两个 : 提供回滚(保证事务的**原子性**) 和 MVCC(多版本并发控制) 。

undo log和redo log记录**物理日志（记录数据的变化）**不一样，它是`逻辑日志(记录执行的SQL操作)`。可以认**为当delete一条记录时，undo log中会记录一条对应的insert记录，反之亦然，当update一条记录时，它记录一条对应相反的 update记录**。当执行rollback时，就可以从undo log中的逻辑记录读取到相应的内容并进行回滚。

Undo log销毁:undo log在事务执行时产生，事务提交时，并不会立即删除undo log，因为这些日志可能还用于MVCC。

Undo log存储:undo log采用段的方式进行管理和记录，存放在前面介绍的 rollback segment 回滚段中，内部包含1024个undo log segment。

#### MVCC(重要)

* 当前读：

  读取的是记录的最新版本，读取时还要保证其他并发事务不能修改当前记录，会`对读取的记录进行加锁`。对于我们日常的操作，**如:select ... lock in share mode(共享锁)，select ... for update、update、insert、delete(排他锁)都是一种当前读。**

* 快照读：

  简单的select(不加锁)就是快照读，快照读，读取的是`记录数据的可见版本`，有可能是历史数据，不加锁，是非阻塞读。
   • Read Committed（提交读）:每次select，都生成一个快照读。
   • Repeatable Read（可重复读）:`开启事务后第一个select语句才是快照读的地方`。**例如，在默认隔离级别时，开启事务，在第一次普通select读时会产生一个快照，后面每次查询都查的是快照上的数据。**

   • Serializable（可串行化读）:快照读会退化为当前读，每一次读取都会加锁。

* MVCC

  全称 Multi-Version Concurrency Control，多版本并发控制。**指维护一个数据的多个版本， 使得读写操作没有冲突，快照读为MySQL实现MVCC提供了一个非阻塞读功能。**MVCC的具体实现，还需要依赖于数据库记录中的三个隐式字段、undo log日志、readView。

* MVCC-实现原理

  * 隐式字段

  |  id  | age  | name |
  | :--: | :--: | :--: |
  |  1   |  1   | Tom  |
  |  3   |  3   | Cat  |

  当我们创建了上面的这张表，我们在查看表结构的时候，就可以显式的看到这三个字段。 实际上除了这三个字段以外，InnoDB还会自动的给我们添加三个隐藏字段及其含义分别是:

  | 隐藏字段    | 含义                                                         |
  | ----------- | ------------------------------------------------------------ |
  | DB_TRX_ID   | 最近修改事务ID，记录插入这条记录或最后一次修改该记录的事务ID。 |
  | DB_ROLL_PTR | 回滚指针，指向这条记录的上一个版本，用于配合undo log，指向上一个版本。 |
  | DB_ROW_ID   | 隐藏主键，如果表结构没有指定主键，将会生成该隐藏字段         |

  * undo log

  回滚日志，在insert、update、delete的时候产生的便于数据回滚的日志。

  当insert的时候，产生的undo log日志只在回滚时需要，在事务提交后，可被立即删除。

  而update、delete的时候，产生的undo log日志不仅在回滚时需要，在快照读时也需要，在事务提交后不会立即被删除。

  * undo log版本链

  以以下四个事务进行演示：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img80.jpg)

  当事务4执行第一条修改语句时，也会记录undo log日志，记录数据变更之前的样子; 然后更新记录，并且记录本次操作的事务ID，回滚指针，回滚指针用来指定如果发生回滚，回滚到哪一个版本。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img81.jpg)

  *注：undo log记录的是逻辑日志，这里只是为了演示方便，才会进行数据的记录，实际上记录的是用于回滚的相反的SQL语句。*

  **不同事务或相同事务对同一条记录进行修改，会导致该记录的undolog生成一条记录版本链表，链表的头部是最新的旧记录，链表尾部是最早的旧记录。**

  * readview

  ReadView(读视图)是`快照读SQL执行时MVCC提取数据的依据`，记录并维护系统当前**活跃的事务 (未提交的)id**。

  ReadView中包含了四个核心字段:

  | 字段           | 含义                                               |
  | -------------- | -------------------------------------------------- |
  | m_ids          | 当前活跃的事务ID集合                               |
  | min_trx_id     | 最小活跃事务ID                                     |
  | max_trx_id     | 预分配事务ID，当前最大事务ID+1(因为事务ID是自增的) |
  | creator_trx_id | ReadView创建者的事务ID                             |

  *版本链数据访问规则*

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img82.jpg)

  *规则比较复杂，不需要记*

  

  **不同的隔离级别，生成ReadView的时机不同:**

  **READ COMMITTED :在事务中每一次执行快照读时生成ReadView。**

  **REPEATABLE READ:仅在事务中第一次执行快照读时生成ReadView，后续复用该ReadView。**

  * 通过案例理解readview

  在RC隔离级别下，在事务中每一次执行快照读时都生成ReadView

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img83.jpg)

  它生成的记录版本链如下：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img81.jpg)

  按照规则，找到的是0x00002那条历史数据，对应事务2提交后的版本数据。

  总结：**在RC（提交读）隔离级别下，快照读产生于当前快照读之前的最后一次事务提交后的数据。**

  

  在RR隔离级别，仅在事务中第一次执行快照读时生成ReadView，后续服用该ReadView。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img84.jpg)

  **总结：在RR（可重复读）级别下，无论哪条快照读都和当前事务第一条select语句的快照读一致。而第一条select语句的快照读为当前语句之前最后一个提交事务后的版本数据。**

* 总体原理实现结构图

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img85.jpg)



### MySQL管理

#### 系统数据库

Mysql数据库安装完成后，自带了一下四个数据库，具体作用如下:

| 数据库             | 含义                                                         |
| ------------------ | ------------------------------------------------------------ |
| mysql              | 存储MySQL服务器正常运行所需要的各种信息 (时区、主从、用户、权限等) |
| information_schema | 提供了访问数据库元数据的各种表和视图，包含数据库、表、字段类型及访问权限等 |
| performance_schema | 为MySQL服务器运行时状态提供了一个底层监控功能，主要用于收集数据库服务器性能参数 |
| sys                | 包含了一系列方便 DBA 和开发人员利用performance_schema 性能数据库进行性能调优和诊断的视图 |

#### 常用工具

* mysql

该mysql不是指mysql服务，而是指mysql的客户端工具。

```sql
语法 :
    mysql   [options]   [database]
选项 :
    -u, --user=name      #指定用户名
    -p, --password[=name]#指定密码
    -h, --host=name			 #指定服务器IP或域名
    -P, --port=port			 #指定连接端口
    -e, --execute=name 	 #执行SQL语句并退出
```

-e选项可以在Mysql客户端执行SQL语句，而`不用连接到MySQL数据库`再执行，对于一些批处理脚本， 这种方式尤其方便。

```shell
# 示例
mysql -h192.168.200.202 -uroot -p123456 itcast -e "select * from stu"
```

* mysqladmin

mysqladmin 是一个执行管理操作的客户端程序。可以用它来**检查服务器的配置和当前状态、创建并删除数据库等**。

```shell
# 通过帮助文档查看选项:
mysqladmin --help
```

```sql
语法:
mysqladmin [options] command ...
选项:
		-u, --user=name      #指定用户名
    -p, --password[=name]#指定密码
    -h, --host=name			 #指定服务器IP或域名
    -P, --port=port			 #指定连接端口
```

示例：

```sql
mysqladmin -uroot –p1234 drop 'test01';
mysqladmin -uroot –p1234 version;
```

* mysqlbinlog

由于服务器生成的二进制日志文件以二进制格式保存，所以如果想要检查这些文本的文本格式，就会使用到mysqlbinlog日志管理工具。

```shell
语法 :
    mysqlbinlog [options]  log-files1 log-files2 ...
选项 :
		-d, --database=name     指定数据库名称，只列出指定的数据库相关操作。
		-o, --offset=#					忽略掉日志中的前n行命令。
		-r,--result-file=name		将输出的文本格式日志输出到指定文件。
		-s, --short-form				显示简单格式， 省略掉一些信息。
		--start-datatime=date1 --stop-datetime=date2  指定日期间隔内的所有日志。
		--start-position=pos1 --stop-position=pos2    指定位置间隔内的所有日志。
```

* mysqlshow

mysqlshow 客户端对象查找工具，用来很快地查找存在哪些数据库、数据库中的表、表中的列或者索引。

```shell
语法 :
    mysqlshow [options] [db_name [table_name [col_name]]]
选项 :
		--count 显示数据库及表的统计信息(数据库，表 均可以不指定) 
		-i 			显示指定数据库或者指定表的状态信息
示例:
		#查询test库中每个表中的字段书，及行数 
		mysqlshow -uroot -p2143 test --count
		#查询test库中book表的详细情况
		mysqlshow -uroot -p2143 test book --count
```

* mysqldump

mysqldump 客户端工具用来备份数据库或在不同数据库之间进行数据迁移。备份内容包含创建表，及插入表的SQL语句。

```shell
语法 :
		mysqldump [options] db_name [tables]
		mysqldump [options] --database/-B db1 [db2 db3...]
		mysqldump [options] --all-databases/-A
连接选项 :
    -u, --user=name         指定用户名
    -p, --password[=name]		指定密码
    -h, --host=name					指定服务器ip或域名
    -P, --port=#						指定连接端口
输出选项:
    --add-drop-database     在每个数据库创建语句前加上 drop database 语句
		--add-drop-table				在每个表创建语句前加上 drop table 语句 , 默认开启 ;不开启 (--skip-add-drop-table)
    -n, --no-create-db			不包含数据库的创建语句
    -t, --no-create-info 		不包含数据表的创建语句
    -d --no-data						不包含数据
		-T, --tab=name 					自动生成两个文件:一个.sql文件，创建表结构的语句;一个.txt文件，数据文件
```

* mysqlimport/source

1). mysqlimport

mysqlimport 是客户端数据导入工具，用来导入mysqldump 加 -T 参数后导出的文本文件。

```shell
语法 :
    mysqlimport [options]  db_name  textfile1  [textfile2...]
示例 :
    mysqlimport -uroot -p2143 test /tmp/city.txt
```

2). source 如果需要导入sql文件,可以使用mysql中的source 指令 :

```shell
语法 :
    source /root/xxxxx.sql
```
