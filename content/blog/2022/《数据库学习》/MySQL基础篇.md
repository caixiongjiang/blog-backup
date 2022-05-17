---
title: "MySQL基础篇 学习笔记"
date: 2022-05-17T18:18:05+08:00
lastmod: 2022-05-17T18:20:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/title_img.jpg"
description: "MySQL基础篇主要学习基础的SQL语句，函数，约束，多表查询以及事务的内容！"
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

## MySQL基础篇

### MySQL概述

#### 数据库相关概念

* 数据库，也就是存储数据的仓库，简称DataBase（DB）
* 数据库管理系统，也就是操纵和管理数据库的软件，DataBase ManageMent System（DBMS）
* SQL是操作`关系型数据库`的编程语言，定义了一套操作关系型数据库统一标准，Structure Query Language（SQL）
* 主流的关系型数据库管理系统：`Oracle`，`MySQL`，`SQL Server`,`SQLite3`(嵌入式微型数据库)

#### 下载&安装MySQL（mac版本）

* 版本：MySQL Community Server 8.0.29（社区免费版）

* 下载：[https://www.mysql.com/](https://www.mysql.com/)

  * mac版本下安装完之后，可以打开系统偏好设置，就可以找到mysql

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img1.png)

  * 需要设置`root`密码

* 启动和停止：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img2.png)

* 连接mysql（命令行工具）：

  * 打开`Terminal`，修改配置文件（Windows下为环境变量配置）：

    * 输入：

      ```shell
      sudo vim .bash_profile
      ```

    * 加入路径：

      ```
      PATH=$PATH:/usr/local/mysql/bin
      ```

    * 保存退出（:wq），并启用该文件(每次新建一个Terminal都需要重新启用)

      ```shell
      source .bash_profile
      ```

  * 连接数据库并输入`root`密码：

    ```shell
    mysql [-h 127.0.0.1] [-p 3306] -u root -p
    ```

    其中带`[]`的部分可以省略，默认连接自身电脑。然后输入密码，出现MySQL版本就成功了！

### 关系型数据库

关系型数据库是建立在`关系模型`的基础上，由多张相互连接的二维表组成的数据库：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img3.jpg)

特性：

* 1.使用表存储，格式统一，便于维护
* 2.使用SQL语言操作，标准统一

#### MySQL数据库数据模型

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img4.jpg)

### SQL

#### SQL通用语法

* SQL语句可以单行或者多行书写，以分号结尾
* SQL语句可以使用空格/缩进来增强语句的可读性
* MySQL数据库的SQL语句不区分大小写，关键字建议使用大写
* 注释：
  * 单行注释：`--注释内容` 或 `# 注释内容`（MySQL特有）
  * 多行注释：`/* 注释内容 */`

#### SQL分类

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img5.jpg)

#### DDL语句

* DDL-数据库操作

  * 查询

    * 查询所有数据库

      ```sql
      show databases;
      ```

    * 查询当前所处于的数据库

      ```sql
      select databases();
      ```

  * 创建

    ```sql
    creat databases [if not exists] 数据库名 [default charset 字符集] [collate 排序规则]
    ```

    `if not exists`代表如果不存在才创建数据库，存在则不创建。

    如果向创建`utf-8`字符集的数据库需要指定的参数为`utf8mb4`：因为如果为`utf8`只能支持最多3个字节大小的数据。

  * 删除

    ```sql
    drop databases [if exists] 数据库名
    ```

  * 使用

    ```sql
    use 数据库名
    ```

    代表切换到当前数据库进行使用操作

* DDL-表操作-查询

  * 查询当前表的所有数据库的所有表(必须切换到当前的数据库中)

    ```sql
    show tables;
    ```

  * 查询表结构

    ```sql
    desc 表名;
    ```

  * 查询指定表的建表语句

    ```sql
    show create table 表名;
    ```

* DDL-表操作-创建

  * 语法：

  ```sql
  create table 表名(
  		字段1 字段1类型[comment 字段1注释],
    	字段2 字段2类型[comment 字段2注释],
  		字段3 字段3类型[comment 字段3注释],
  		...
    	字段n 字段n类型[comment 字段n注释]
  )[comment 表注释];
  ```

  注意：*[...]为可选参数，最后一个字段后面没有逗号*

  * Demo:

    ```sql
    create table tb_user(
    		id int comment "编号",
      	name varchar(50) comment "姓名",
    		age int comment "年龄",
      	gender varchar(1) comment "性别"
    )comment "用户表";
    ```

    

  * 创建成功之后，使用`show create table tb_user;`之后显示真实的表创建语句：

    ```sql
    CREATE TABLE `tb_user` (
      `id` int DEFAULT NULL COMMENT '编号',
      `name` varchar(50) DEFAULT NULL COMMENT '姓名',
      `age` int DEFAULT NULL COMMENT '年龄',
      `gender` varchar(1) DEFAULT NULL COMMENT '性别'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户表'; 
    ```

    **其中`DEFAULT`代表默认值，`ENGINE`代表数据库引擎，`COLLATE`代表排序规则。**

* DDL-表操作-数据类型

  MySQL中的数据类型有很多，主要分为3类：数值类型，字符串类型，日期时间类型。

  * 数值类型：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img6.jpg)

    *其中`123.456`的`精度`为6，`标度`为2*

    *如果使用无符号的数可以加一个unsigned：`age TINYINT UNSIGNED`*

    *使用`double`需要指定参数：`socore double(4, 1)`:第一个参数代表整个数值的长度（包括小数位），第二个参数代表小数位数*

  * 字符类型：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img7.jpg)

    *二进制数据：视频，音频，软件安装包 vs 长文本数据：文字信息；但在实际的开发中一般不存储二进制数据，性能不高，也不够方便*

    *char(10) vs varchar(10):两者在字符超过10个字节都会报错。然而前者在字符串只有1个字节时，也会占用十个字节的空间，为占用的字符其他空间会用空格补位；后者使用几个字符就占几个字符的空间*

    **`char`性能高，`varchar`相对于`char`性能较低：因为`varchar`在使用时需要根据内容计算占用的空间**

    demo：用户名的类型一般用varchar（不定长）；性别的类型一般使用char（不是男就是女）。

  * 日期类型：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img8.jpg)

  * Demo（根据需求创建表）:

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img9.jpg)

    ```sql
    create table emp(
        id int comment "编号",
        worknumber varchar(10) comment "工号",
        name varchar(10) comment "姓名",
        gender char(1) comment "性别",
        age tinyint unsigned comment "年龄",
        idcard char(18) comment "身份证号",
        entrydate date comment "入职时间"
    ) comment "员工表";
    ```

* DDL-表操作-修改

  * 添加字段

    ```sql
    alter table 表名 add 字段名 类型（长度）[comment 注释][约束];
    ```

  * 修改字段

    * 修改数据类型

      ```sql
      alter table 表名 modify 字段名 新数据类型(长度);
      ```

    * 修改字段名和数据类型

      ```sql
      alter table 表名 change 旧字段名 新字段名 类型(长度) [comment 注释][约束];
      ```

  * 删除字段

    ```sql
    alter table 表名 drop 字段名;
    ```

  * 修改表名

    ```sql
    alter table 表名 rename to 新表名;
    ```

* DDL-表操作-删除

  * 删除表

    ```sql
    drop table [if exists] 表名;
    ```

  * 删除指定表，并重新创建该表

    ```sql
    truncate table 表名;
    ```

    **注意这个操作会将表格中的数据清空，只留下表格的字段及基本信息！**

* DDL语句总结

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img10.jpg)

#### MySQL图形化界面

市面上主流的工具有：`Sqlyog`，`Navicat`，`DataGrip`，本次使用`DataGrip`。

#### DML语句

DML语句也叫做数据操作语言，用来对数据库表中的数据记录进行增删改操作。

添加数据（`INSERT`）；修改数据（`UPDATE`）；删除数据（`DELETE`）。

* DML-添加数据

  * 给指定字段添加数据

    ```sql
    insert into 表名 (字段1, 字段2, ...) values(值1, 值2,...);
    ```

  * 给**全部字段**添加数据

    ```sql
    insert into 表名 values (值1, 值2,...);
    ```

  * 批量添加数据

    ```sql
    insert into 表名 (字段1, 字段2, ...) values(值1, 值2,...), values(值1, 值2,...), values(值1, 值2,...);
    ```

    ```sql
    insert into 表名 values (值1, 值2,...), values(值1, 值2,...), values(值1, 值2,...);
    ```

  注意:

  * *插入数据时，指定字段顺序需要和值的顺序是一一对应的。*
  * *字符串和日期型数据应该包含在引号中。*
  * *插入数据的大小，应该在字段的规定的范围内。*

* DML-修改数据

  ```sql
  update 表名 set 字段名1 = 值1, 字段名2 = 值2,...[where 条件];
  ```

  **注意：修改语句的条件可以有，也可以没有，如果没有条件，则会修改整张表的所有数据。**

* DML-删除数据

  ```sql
  delete from 表名 [where 条件];
  ```

#### DQL语句

DQL是数据查询语言，用来查询数据库中表的记录。查询关键字为`SELECT`。

* DQL-语法（单表查询）

  ```sql
  select 
  			字段列表
  from
  			表名列表
  where 
  			条件列表
  group by
  			分组字段列表
  having
  			分组后条件列表
  order by
  			排序字段列表
  limit
  			分页参数
  ```

* DQL-基本查询

  * 查询返回多个字段

    ```sql
    select 字段1,字段2,字段3... from 表名;
    ```

    ```sql
    select * from 表名;
    ```

  * 设置别名

    ```sql
    select 字段1 [as 别名1],字段2 [as 别名2]... from 表名;
    ```

  * 去除重复记录

    ```sql
    select distinct 字段列表 from 表名;
    ```

* DQL-条件查询

  * 语法：

    ```sql
    select 字段列表 from 表名 where 条件列表;
    ```

  * 条件：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img11.jpg)

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img12.jpg)

* DQL-聚合函数

  * 定义：将一列数据作为一个整体，进行纵向计算。

  * 常见聚合函数(**作用于某一列数据**)：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img13.jpg)

  * 语法：

    ```sql
    select 聚合函数(字段列表) from 表名
    ```

    注意：null值不参与所有聚合函数运算

* DQL-分组查询

  * 语法：

    ```sql
    select 字段列表 from 表名 [where 条件] group by 分组字段名 [having 分组后过滤条件];
    ```

  * `where`和`having`区别：

    * 执行时机不同：where是分组之前进行过滤，不满足where条件，不参与分组；而having是分组之后对结果进行过滤。
    * 判断条件不同：where不能对聚合函数进行判断，而having可以。

  * 注意：

    * 执行顺序：`where` > `聚合函数` > `having`。
    * **分组之后，查询的字段一般为聚合函数和分组字段，查询其他字段无任何意义。**

* DQL-排序查询

  * 语法：

    ```sql
    select 字段列表 from 表名 order by 字段1 排列方式1, 字段2 排列方式2;
    ```

  * 排序方式

    * ASC：升序（默认值）
    * DESC：降序

    **注意：如果是多字段排序，当第一个字段值相同时，才会根据第二个字段进行排序**

* DQL-分页查询

  * 语法：

    ```sql
    select 字段列表 from 表名 limit 起始索引, 查询记录数;
    ```

  注意：

  * 起始索引从0开始，起始索引 = （查询页码 - 1）* 每页显示记录数
  * 分页查询是数据库的方言，不同的数据库有不同的实现，MySQL中是`LIMIT`。
  * 如果查询的是第一页数据，起始索引可以省略，直接简写为limit 10。

* DQL案例：

  按照需求完成DQL语句的编写：

  ```sql
  -- ---------------------DQL语句案例练习------------------------
  -- 1.查询年龄为20，21，22，23岁的女性员工信息。
  select * from emp where age in(20, 21, 22, 23) and gender = '女';
  -- 2.查询性别为男，并且年龄在20-40岁（含）以内的姓名为三个字的员工。
  select * from emp where gender = '男' and age between 20 and 40 and name like '___';
  -- 3.统计员工表中，年龄小于60岁的，男性员工和女性员工的人数。
  select gender, count(id) '数量' from emp where age < 60 group by gender;
  -- 4.查询所有年龄小于等于35岁员工的姓名和年龄，并对查询结果按年龄升序排列，如果年龄相同按入职时间降序排序。
  select name, age from emp where age <= 35 order by age asc, entrydate desc;
  -- 5.查询性别为男，且年龄在20-40岁（含），对查询的结果按年龄升序排序，年龄相同按入职时间升序排序,最后取前5个员工信息。
  select * from emp where gender = '男' and age between 20 and 40 order by age asc, entrydate asc limit 0,5;
  ```

* DQL-执行顺序

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img14.jpg)

  * Demo:

    ```sql
    -- 查询年龄大于15的员工的姓名、年龄、并根据年龄进行升序排序
    select name, age from emp where age > 15 order by age asc;
    -- 执行顺序：
    -- from emp ==> where age >15 ==> select name ==> order by age asc
    ```

#### DCL语句

DCL是数据控制语言，用来管理数据库用户、控制数据库的访问权限。

* DCL-管理用户

  * 查询用户

    ```sql
    use mysql;
    select * from user;
    ```

    *在MySQL数据库中用户的信息和用户具有的权限信息都存放在系统数据库中的`mysql`的`user`表中的。*

  * 创建用户

    ```sql
    create user '用户名'@'主机名' identified by '密码';
    ```

  * 修改用户密码

    ```sql
    alter user '用户名'@'主机名' identified with mysql_native_password by '新密码';
    ```

  * 删除用户

    ```sql
    drop user '用户名'@'主机名';
    ```

  注意：

  * 主机名可以使用%通配。
  * 这类SQL开发人员操作比较少，主要是DBA（数据库管理员）使用。

* DCL-权限控制

  * mysql中常用的权限：

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img15.jpg)

  * 查询权限

    ```sql
    show grants for '用户名'@'主机名';
    ```

  * 授予权限

    ```sql
    grant 权限列表 on 数据库名.表名 to '用户名'@'主机名';
    ```

  * 撤销权限

    ```sql
    revoke 权限列表 on 数据库名.表名 from '用户名'@'主机名';
    ```

  注意：

  * 多个权限之间，使用逗号分隔
  * 授权时，数据库名和表名可以使用"*"进行通配，代表所有。

### 函数

`函数`是指一段可以直接被另一段程序调用的程序或者代码。

* 使用：

  ```sql
  select 函数(参数);
  ```

#### 字符串函数

MySQL中常用的字符串函数：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img16.jpg)

* 注意：其中`substring`函数与`java中的substring`以及`c++中的substr`有区别：**它的索引是从1开始计数的**。

* 案例练习：

  * 需求：由于业务需求变更，统一为5位数，目前不足5位数的全部在前面补0.比如：1号员工的工号应该为00001。

  * 实现：

    ```sql
    update emp set workno = lpad(workno, 5, 0);
    ```

#### 数值函数

MySQL中常用的数值函数：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img17.jpg)

* 案例练习：

  * 需求：通过数据库的函数，生成一个六位数的随机验证码。

  * 实现：

    ```sql
    select lpad(floor(rand() * 1000000), 6, 0) '六位数的随机验证码';
    ```

#### 日期函数

MySQL中常用的日期函数：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img18.jpg)

* 案例练习：

  * 需求：查询所有员工的入职天数，并根据入职天数倒序排序

  * 实现：

    ```sql
    select name, datediff(curdate(), entrydate) '入职天数' from emp order by 入职天数 desc;
    ```

#### 流程函数

MySQL中常用的流程函数：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img19.jpg)

* 案例练习1:

  * 需求：需求：查询emp表的员工姓名和工作地址（上海/北京-----> 一线城市 其他---------> 二线城市）

  * 实现：

    ```sql
    select name, (case workaddress when '北京' then '一线城市' when '上海' then '一线城市' else '二线城市' end) '工作地址' from emp;
    ```

* 案例练习2:

  * 需求：统计各班级各个学员的成绩，展示的规则如下：如果 >= 85，展示优秀， >= 60 展示及格，否则展示不及格。

  * 实现：

    ```sql
    -- 创建表结构
    create table score(
        id int comment 'ID',
        name varchar(20) comment '姓名',
        math int comment '数学',
        english int comment '英语',
        chinese int comment '语文'
    ) comment '学员成绩表';
    
    -- 插入数据
    insert into score values (1, 'Tom', 67, 88, 95),
                             (2, 'Rose', 23, 66, 90),
                             (3,'Jack', 56, 98, 76);
    
    -- 按照规则查询数据
    select id,
           name,
           (case when math >= 85 then '优秀' when math >= 60 then '及格' else '不及格' end) '数学',
           (case when english >= 85 then '优秀' when english >= 60 then '及格' else '不及格' end) '英语',
           (case when chinese >= 85 then '优秀' when chinese >= 60 then '及格' else '不及格' end) '语文'
    from score;
    ```

### 约束

#### 约束概述

1.概念：约束是作用于表中字段上的规则，用于限制存储在表格中的数据。

2.目的：保证数据库中数据的正确性，有效性，完整性。

3.分类：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img20.jpg)

**注意：约束是作用于表中的字段上的，可以在创建表/修改表的时候添加约束。**

#### 约束演示

根据需求，完成表结构的创建：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img21.jpg)

```sql
create table user(
    id int primary key auto_increment comment '主键',
    name varchar(10) not null unique comment '姓名',
    age int check ( age > 0 and age <= 120 ) comment '年龄',
    status char(1) default '1' comment '状态',
    gender char(1) comment '性别'
) comment '用户表';
```

#### 外键约束

* 概念：外键是用来让两张表的数据之间建立连接，从而保证数据的一致性和完整性。

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img22.jpg)

  上述两张表中`子表中的dept_id`为`外键`，外键关联的表称为父表。*注意：上述两张表在数据库层面并未建立外键连接。*

* 语法

  * 添加外键（创建表时）：

    ```sql
    create table 表名(
    			字段名 数据类型,
      		...
    			[constraint] [外键名称] foreign key(外键字段名) references 主表(主列表名)
    );
    ```

  * 表结构创建好之后（额外添加）：

    ```sql
    alter table 表名 add constraint 外键名称 foreign key(外键字段名) references 主表(主表字段名);
    ```

  * 删除外键

    ```sql
    alter table 表名 drop foreign key 外键名称;
    ```

* 删除/更新行为

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img23.jpg)

  ```sql
  alter table 表名 add constraint 外键名称 foreign key(外键字段名) references 主表(主表字段名) on update 行为 on delete 行为;
  ```

### 多表查询

#### 多表关系

* 概述

  项目开发中，在进行数据库表结构设计时，会根据业务需求及业务模块之间的关系，分析并设计表结构，由于业务之间相互关联，所以各个表结构之间也存在各种联系，基本上分为三种：

  * 一对多（多对一）：

    * 案例：部门和员工的关系
    * 关系：一个部门对用多个员工，一个员工对应一个部门
    * 实现：**在多的一方建立外键，指向一的一方的主键**

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img24.jpg)

  * 多对多

    * 案例：学生和课程的关系
    * 关系：一个学生可以选修多门课程，一门课程也可以供多个学生选择
    * 实现：**建立第三张中间表，中间表至少含有两个外键，分别关联两方主键**

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img25.jpg)

  * 一对一

    * 案例：用户和用户详情的关系
    * 关系：一对一关系，多用于单表拆分，将一张表的基础字段放在一张表中，其他详情字段放在另一张表中，以提升操作效率。
    * 实现：**在任意的一方加入外键，关联另外一方的主键，并且设置外键为唯一的（UNIQUE），设置外键为唯一值是为了防止其成为一对多的类型**

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img26.jpg)

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img27.jpg)

#### 多表查询

* 多表查询概述：指从多张表中查询数据

* 笛卡尔积：笛卡尔积是指在数学中，两个集合A集合和B集合的所有组合情况。（在多表查询时，需要消除无效的笛卡尔积）

  ```sql
  -- 多表查询 -- 笛卡尔积
  select * from employee, dept;
  
  -- 去除无效的笛卡尔积
  select * from employee, dept where employee.dept_id = dept.id;
  ```

* 多表查询分类

  * 连接查询：

    内连接：相当于查询A，B交集部分数据

    外连接：

    ​		左外连接：查询左表所有数据，以及两张表交集部分数据

    ​		右外连接：查询右表所有数据，以及两张表交集部分数据

    自连接：当前表与自身连接查询，自连接必须使用表的别名![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img28.jpg)

    *如上图所示，内外连接的情况都可以用这张图进行表示*

  * 子查询

#### 内连接

内连接查询的是两张表交集的部分！

内连接查询语法：

* 隐式内连接：

  ```sql
  select 字段列表 from 表1, 表2 where 条件...;
  ```

* 显式内连接：

  ```sql
  select 字段列表 from 表1 [inner] join 表2 on 连接条件...;
  ```

* Demo:

  ```sql
  -- 内连接演示
  -- 1.查询每一个员工的姓名，及关联的部门的名称（隐式内连接实现）
  -- 表结构：employee, dept
  -- 连接条件：employee.dept_id = dept.id
  select employee.name, dept.name from employee, dept where employee.dept_id = dept.id;
  
  -- 2.查询每一个员工的姓名，及关联的部门的名称（显示内连接实现）... inner join ... on ...
  select e.name, d.name from employee e inner join dept d on e.dept_id = d.id;
  select e.name, d.name from employee e join dept d on e.dept_id = d.id;
  ```

  **隐式连接好理解好书写，语法简单，担心的点较少。但是显式连接可以减少字段的扫描，有更快的执行速度。这种速度优势在3张或更多表连接时比较明显。**

#### 外连接

外连接查询语法：

* 左外连接

  ```sql
  -- 相当于查询表1（左表）的所有数据包含表1和表2交集部分的数据
  select 字段列表 from 表1 left[outer] join 表2 on 条件...;
  ```

* 右外连接

  ```sql
  -- 相当于查询表2（右表）的所有数据包含表1和表2交集部分的数据
  select 字段列表 from 表1 right[outer] join 表2 on 条件...;
  ```

* Demo:

  ```sql
  -- 外连接演示
  -- 1.查询employee的所有数据，和对应的部门信息（左外连接）
  -- 表结构：employee, dept
  -- 连接条件：employee.dept_id = dept.id
  select e.*, d.name from employee e left join dept d on e.dept_id = d.id;
  
  
  -- 2.查询dept表中的所有数据，和对应员工信息（右外连接）
  select e.*, d.name from employee e right join dept d on e.dept_id = d.id;
  ```

  结果：

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img29.jpg)

  ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img30.jpg)

  **第一张图为左外连接查询，第二张图为右外连接查询。可以明显地看出区别：左外连接是以左表的所有信息为基础加上与之关联的右表信息；而右外连接是以右表的所有信息为基础加上与之关联的左表信息！**

#### 自连接

自连接查询语法：

```sql
select 字段列表 from 表A 别名A join 表A 别名B on 条件...;
```

**自连接查询可以是内连接查询也可以是外连接查询。**

Demo:查询一张表中员工及其所属领导的名字,思路如下

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img31.jpg)

```sql
-- 自连接
-- 1.查询员工及其所属领导的名字
-- 表结构：employee
-- 思路：将原表分为A表和B表两个副本，分别代表员工表和领导表，连接条件为 A.managerid = B.id（内连接的形式）
select a.name '员工', b.name '他的领导' from employee a, employee b where a.managerid = b.id;

-- 2.查询所有员工employee及其领导的名字employee，如果没有领导，也要查询出来（外连接的形式）
select a.name '员工', b.name '他的领导' from employee a left join employee b on a.managerid = b.id;
```

**如上所示，如果没有领导也要查询出来，就代表要包含所有员工的信息，也就是A.name，所以要使用左外连接！**

#### 联合查询-union, union all

对于`union`查询，就是把多次查询的结果合并起来，形成一个**新的查询结果集**。

* 语法：

  ```sql
  select 字段列表 from 表A...
  union [all]
  select 字段列表 from 表B...;
  ```

* Demo:

  ```sql
  -- 联合查询
  -- 1.将薪资低于10000的员工，和年龄大于50的员工全部查询出来
  
  -- 分开查询
  select * from employee where salary < 10000;
  select * from employee where age > 50;
  -- 联合查询(不去重)
  select * from employee where salary < 10000
  union all
  select * from employee where age > 50;
  -- 联合查询（去重）
  select * from employee where salary < 10000
  union
  select * from employee where age > 50;
  select * from employee where age > 50 or salary < 10000;
  ```

  联合查询的结果相当于把分开查询的结果直接合并起来一起展示。

  这个联合查询和使用筛选条件为or查询的结果的区别为：

  * **联合查询在同一个人同时符合薪资低于10000和年龄大于50的两个条件时分开显示两条记录，而使用or条件查询时上述情况只显示一条记录。**

  * 将all关键字去掉之后的结果，就为去重的结果
  * Union all只能对**查询结果结构相同（相同的列数和类型）**进行合并

#### 子查询

* 概念：SQL语句中嵌套SELECT语句，称为`嵌套查询`,又称`子查询`。

  ```sql
  select * from t1 where colum1 = (select column1 from t2);
  ```

  **子查询外部的语句可以是INSERT/UPDATE/DELETE/SELECT的任何一个。**

* 根据子查询结果不同，分为：

  * 标量子查询（子查询结果为单个值）

    常用的操作符：=  <>  >  >=  <  <=

    ```sql
    -- 标量子查询
    -- 1.查询销售部所有的员工信息
    -- a.查询销售部的部门id
    -- b.根据销售部的部门id查询员工信息
    select * from employee where dept_id = (select id from dept where name = '销售部');
    
    -- 2.查询在"方东白"入职之后的员工信息
    -- a.查询"方东白"的入职日期
    -- b.查询该日期之后的所有员工信息
    select * from employee where entrydate > (select entrydate from employee where name = '方东白');
    ```

  * 列子查询（子查询结果为一列）

    常用的操作符：IN 、NOT IN 、ANY、 SOME、ALL

    ![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img32.jpg)

    ```sql
    -- 列子查询
    -- 1.查询"销售部"和"市场部"的所有员工信息
    -- a.查询销售部的和市场部的部门id
    -- b.根据部门id查询员工信息
    select * from employee where dept_id in (select id from dept where name = '销售部' or '市场部');
    
    -- 2.查询比财务部所有人工资都高的员工信息
    -- a.根据财务部查询部门id
    -- b.根据部门id查询所有财务部人员工资
    -- c.比财务部所有人员工资都高的员工信息
    select * from employee where salary > all (select salary from employee where dept_id = (select id from dept where name = '财务部'));
    
    -- 3.查询比研发部其中任意一人工资高的员工信息
    -- a.根据研发部查询部门id
    -- b.根据部门id查询所有研发部人员工资
    -- c.比研发部任意一人员工资高的员工信息
    select * from employee where salary > any (select salary from employee where dept_id = (select id from dept where name = '研发部'));
    ```

  * 行子查询（子查询结果为一行）

    常用的操作符：=  、<> 、IN 、NOT IN 

    ```sql
    -- 行子查询
    -- 1.查询"张无忌"的薪资及直属领导相同的员工信息
    -- a.查询张无忌的薪资和领导
    -- b.查询张无忌的薪资及直属领导相同的员工信息
    select * from employee where (salary, managerid) = (select salary, managerid from employee where name = '张无忌') and name <> '张无忌';
    ```

  * 表子查询（子查询结果为多行多列）

    常用操作符：IN

    ```sql
    -- 表子查询
    -- 1.查询与鹿杖客，宋远桥的职位和薪资相同的员工信息
    -- a.查询鹿杖客和宋远桥的职位和薪资
    -- b.查询与它们相同的员工信息
    select * from employee where (salary, job) in (select salary, job from employee where name = '鹿杖客' or name = '宋远桥');
    
    -- 2.查询入职日期是"2001-01-01"之后的员工信息，及部门信息
    -- a.查询入职日期在"2001-01-01"之后的员工信息
    -- b.这部分员工对应的部门信息
    select e.*, d.* from (select * from employee where entrydate > '2001-01-01') e left join dept d on e.dept_id = d.id;
    ```

* 根据子查询位置，分为：WHERE之后、FROM之后、SELECT之后。

#### 多表查询案例

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img33.jpg)

```sql
-- ----------------------------- 多表查询案例 ------------------------
create table salgrade(
    grade int,
    losal int,
    hisal int
) comment '薪资等级表';

insert into salgrade values (1, 0, 3000),
                            (2, 3001, 5000),
                            (3, 5001, 8000),
                            (4, 8001, 10000),
                            (5, 10001, 15000),
                            (6, 15001, 20000),
                            (7, 20001, 25000),
                            (8, 25001, 30000);

-- 1.查询员工的姓名、年龄、职位、部门信息（隐式内连接）
select e.name, e.age, e.job, d.name '部门' from employee e, dept d where e.dept_id = d.id;

-- 2.查询年龄小于30岁的员工的姓名，年龄，职位，部门信息（显式内连接）
select e.name, e.age, e.job, d.name '部门' from employee e inner join dept d on e.dept_id = d.id where age < 30;

-- 3.查询拥有员工的部门ID和部门名称
-- 需求就是查询有员工的部门，也就是求两个表的交集部分，采用内连接
-- 还需要对查询的结果进行去重
select distinct d.* from employee e, dept d where e.dept_id = d.id;

-- 4.查询所有年龄大于40岁的员工，及其归属的部门名称；如果员工没有分配部门，也需要展示出来
-- 如果员工没有分配部门，也需要展示出来，代表这是一个外连接。
select e.*, d.name '部门' from employee e left join dept d on d.id = e.dept_id where age > 40;

-- 5.查询所有员工的工资等级
-- 表：employee, salgrade
-- 连接条件：salgrade.losal <= employee.salary <= salgrade.hisal
select e.name, s.grade from employee e, salgrade s where e.salary >= s.losal and e.salary <= s.hisal;
select e.name, s.grade from employee e, salgrade s where e.salary between s.losal and s.hisal;

-- 6.查询"研发部"所有员工的信息及工资等级
-- a.查询研发部的id
-- b.根据id查询员工信息和工资等级表进行连接
-- 查询所有员工信息说明是外连接
select e.*, s.grade
from employee e,
     dept d,
     salgrade s
where e.dept_id = d.id
  and (e.salary between s.losal and s.hisal)
  and d.name = '研发部';

select e.*, s.grade
from employee e
         left join salgrade s on e.salary between s.losal and s.hisal
where dept_id = (select id from dept where name = '研发部');

-- 7.查询研发部与员工的平均工资
select avg(e.salary) from employee e, dept d where e.dept_id = d.id and d.name = '研发部';

-- 8.查询工资比"灭绝"高的员工信息
-- a.查询"灭绝"的薪资
-- b.查询比她工资高的员工数据
select * from employee where salary > (select salary from employee where name = '灭绝');

-- 9.查询比高于平均薪资的员工数据
select * from employee where salary > (select avg(salary) from employee);

-- 10. 查询低于本部门平均工资的员工信息
-- a.查询本部门平均薪资
-- b.低于此薪资的员工信息
select * from employee e2 where salary < (select avg(e.salary) from employee e where e.dept_id = e2.dept_id);

-- 11.查询所有的部门信息，并统计部门的员工数量
select d.id, d.name, (select count(id) from employee e where e.dept_id = d.id) '员工数量' from dept d;

-- 12.查询所有学生的选课情况，展示出学生名称，学号，课程名称
-- 表结构：student，course，student_course
-- 连接条件：student.id = student_course.id, course.id = student_course.courseid

select s.name, s.number, c.name from student s, student_course sc, course c where s.id = sc.studentid and c.id = sc.courseid;
```

### 事务

#### 事务简介

`事务`是一组操作的集合，它是一个不可分割的工作单位，事务会将所有的操作作为一个整体一起向系统提交或撤销操作请求，即这些操作`要么同时成功，要么同时失败`。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img34.jpg)

**默认MySQL的事物是自动提交的，也就是说，当执行一条DML语句，MySQL会立即执行隐式的提交事务。**

#### 事务操作

方式一：

* 查看/设置事务的提交方式

  ```sql
  select @@autocommit;
  set @@autocommit = 0;
  ```

* 提交事务

  ```sql
  commit;
  ```

* 回滚事务

  ```sql
  rollback;
  ```

方式二：

* 开启事务

  ```sql
  start transaction 或 begin;
  ```

* 提交事务

  ```sql
  commit;
  ```

* 回滚事务

  ```sql
  rollback;
  ```

Demo:

```sql
-- --------------------------事务操作-----------------------
-- 数据准备
create table account(
    id int auto_increment primary key comment '主键ID',
    name varchar(10) comment '姓名',
    money int comment '余额'
) comment '账户表';
insert into account(id, name, money) values (null, '张三', 2000), (null, '李四', 2000);


-- 恢复数据
update account set money = 2000 where name = '张三' or name = '李四';


-- 查看数据库事务提交方式
select @@autocommit; #结果为1，代表事务是自动提交；1代表手动提交
-- 设置为手动提交
set @@autocommit = 0;


-- 转账操作（张三给李四转账1000）
-- 1.查询张三账户余额
select * from account where name = '张三';

-- 2.如果账户余额大于1000，将张三账户余额-1000
update account set money = money - 1000 where name = '张三';

-- 3.李四账户余额+1000
update account set money = money + 1000 where name = '李四';

-- 提交事务
commit;

-- 回滚事务
rollback;


-- 方式二
-- 开启事务
start transaction;
-- 1.查询张三账户余额
select * from account where name = '张三';

-- 2.如果账户余额大于1000，将张三账户余额-1000
update account set money = money - 1000 where name = '张三';

程序执行报错...
-- 3.李四账户余额+1000
update account set money = money + 1000 where name = '李四';

-- 提交事务
commit;

-- 回滚事务
rollback;
```



#### 事务的四大特性

* 原子性：事务是不可分割的最小操作单元，要么全部成功，要么全部失败。
* 一致性：事务完成时，必须使所有数据都保持一致的状态。
* 隔离性：数据库系统提供的隔离机制，保证事务在不受外界并发操作影响的独立环境下运行。（**A事务和B事务不会相互影响**）
* 持久性：事务一旦提交或回滚，它对数据库中的数据的改变就是永久的。

#### 并发事务引发的问题

A事务和B事务在同时操作一个数据库表所引发的问题：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img35.jpg)

`脏读`具体表现：事务A读取到事务B还没有提交（commit）的数据！

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img36.jpg)

`不可重复读`具体表现：同样的sql语句在同一个事务中读到的内容却不同！换句话说，`可重复读`的事务隔离级别就是同一条sql语句在同一事务中读到的内容一定相同！

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img37.jpg)

`幻读`具体表现：幻读现象其实就是设置了`不可重复读`的隔离级别之后，同一条sql语句在同一事务中运行结果相同，而插入时又产生了冲突，就好像他又存在了一样！

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img38.jpg)



#### 事务的隔离级别

为了解决并发事务带来的问题，产生了一个新的概念：`事务的隔离级别`。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mysql/img39.jpg)

上图中第三种为**MySQL的默认事务隔离级别**（Oracle把第二种为作为默认的事物隔离级别）,**最后一种隔离级别代表`串行`,也就是事务A没有结束，事务B的操作就会阻塞，同时只能有一个事务在运行。**

从上往下事务隔离级别越来越高，效率是越来越低的。

```sql
-- 查看事务隔离级别
select @@transaction_isolation;

-- 设置事务隔离级别
-- session代表只对当前会话窗口有效
-- global代表对所有的会话窗口有效
set [session|global] transaction isolation level {read uncommitted | read committed | repeatable read | serializable};
```

