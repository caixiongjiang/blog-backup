---
title: "Python的编程艺术"
date: 2024-07-30T18:18:05+08:00
lastmod: 2024-07-30T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/python_advance/title.jpg"
description: "学习一些较为常用的编程设计思想"
tags:
- python
categories:
- Python语法
series:
- 《Python进阶》
comment : true
---

## Python的编程艺术


### 策略模式

当前我们需要写一个处理文件的类：
```python
from abc import abstractmethod, ABC


class ProcessStrategy(ABC):
    # 约束子类必须实现process_file方法
    @abstractmethod
    def process_file(self, filepath):
        pass


class ExcelProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Processing excel file")


class CsvProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Process csv file")

class TxtProcessStrategy(ProcessStrategy):
    def process_file(self, filepath):
        print("Process txt file")


class FileProcessor:
    def __init__(self, file_path: str, strategy: ProcessStrategy) -> None:
        self.file_path = file_path
        self.strategy = strategy

    def process_file(self) -> None:
        self.strategy.process_file(self.file_path)
```

在上述代码中，`@abstractmethod`是一个装饰器，使用该装饰器的类方法，在子类继承其类的时候必须实现该方法。`ABC`的全称是`abstract base class`(抽象类)，当我们定义仅作为基类而不应该被实例化的类时，就会用到`ABC`。当前`ABC`和`@abstractmethod`使得基类不能被实例化。然后再FileProcessor中传入了策略类，这样避免了反复更改FileProcessor类的代码。

### EAFP：先斩后奏的编程哲学

EAFP全名叫`it's easier to ask for forgiveness than permission`，请求原谅比请求许可更容易。

来看一个例子：
```python
profile1 = {"name": "Tom", "age": 33}

def print_profile(profile):
    name = profile["name"]
    age = profile["age"]
    print(f"This is {name}, {age} years old.")

print_profile(profile1)


profile2 = {"name": "Jerry"}
print_profile(profile2)
```

可以根据逻辑明显看到，在`print_profile(profile2)`这句代码会报错。

假设使用`请求许可`的方式规避：
```python
def print_profile(profile):
    if "name" in profile and "age" in profile:
        name = profile["name"]
        age = profile["age"]
        print(f"This is {name}, {age} years old.")
    else:
        print("Missing keys!")
```

使用`请求原谅`的方式进行规避：
```python
def print_profile(profile):
    try:
        name = profile["name"]
        age = profile["age"]
        print(f"This is {name}, {age} years old.")
    except KeyError as e:
        print(f"Missing {e} keys!")
```

通常来说，在工程代码上，请求原谅的写法比请求许可是更好的：
* 请求许可中，需要先进行读取两次对象来确认是否存在问题，请求原谅则只需要读取一次。
* 可读性更强，可以让别人迅速知道代码在规避什么问题

再看一个读文件的例子：
```python
import os

def read_file(filepath: str) -> None:
    if os.path.exists(filepath):
        with open(filepath) as f:
            print(f.read())
    else:
        print("File not exists!")
```

它的请求原谅版本：
```python
def read_file(filepath: str) -> None:
    try:
        with open(filepath) as f:
            print(f.read())
    except FileNotFoundError as e:
        print(e)
    else:
        with f:
            print(f.read())
# 这里的else代表try的语句没有出现报错时，会执行的语句
```

### Python中的工厂模式

工厂模式是一种创建型设计模式，它提供了一种创建对象的方式，而无需指定具体的类。工厂模式通过定义一个共同的接口来创建对象，让子类决定实例化哪一个类，核心是将**创建对象和使用对象的过程分开**。

来看一个例子：
```python
class DatabaseConnection:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"


def client():
    main_db = DatabaseConnection("localhost", 3306, "root", "password123")
    analytics_db = DatabaseConnection("192.168.1.1", 5432, "admin", "securepass")
    cache_db = DatabaseConnection("10.0.0.1", 27017, "cacheuser", "cachepass")

    print(main_db.connect())
    print(analytics_db.connect())
    print(cache_db.connect())

client()
```
上述代码的缺点是代码的重复使用和修改密码相关内容在大型项目中会相对耗时。我们根据工厂模式的特点，将创建和使用分开：
```python
class DatabaseConnection:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"

# 创建对象过程集成
def connection_factory(db_type):
    db_configs = {
        "main": {
            "host": "localhost",
            "port": 3306,
            "username": "root",
            "password": "password123"
        },
        "analytics": {
            "host": "192.168.1.1",
            "port": 5432,
            "username": "admin",
            "password": "securepass"
        },
        "cache": {
            "host": "10.0.0.1",
            "port": 27017,
            "username": "cacheuser",
            "password": "cachepass"
        }
    }

    return DatabaseConnection(**db_configs[db_type])

# 使用对象过程集成
def client():
    main_db = connection_factory("main")
    analytic_db = connection_factory("analytics")
    cache_db = connection_factory("cache")


    print(main_db.connect())
    print(analytic_db.connect())
    print(cache_db.connect())

client()
```
这样可以使代码更加清晰明了，当然db_config对象可以放在配置文件中，读取进来，这样更符合设计原则。

### Python中的建造者模式

建造者模式是一种创建型设计模式，它允许你**创建复杂对象的步骤分解**，使得同样的构建过程可以创建不同的表示。

当一个类的创建需要传入大量的参数时，我们就可以使用建造者模式，将复杂对象进行分解，来看一个例子：

```python
class DatabaseConnection:
    def __init__(self, host, port, username, password,
                 max_connections=None, timeout=None,
                 enable_ssl=False,
                 ssl_cert=None, connection_pool=None,
                 retry_attempts=None,
                 compression=None,
                 read_preference=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.max_connections = max_connections
        # validate timeout
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be a positive integer")
        self.timeout = timeout
        self.enable_ssl = enable_ssl
        self.ssl_cert = ssl_cert
        self.connection_pool = connection_pool
        self.retry_attempts = retry_attempts
        self.compression = compression
        self.read_preference = read_preference

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"


def client():
    connection = DatabaseConnection(
        "localhost", 5432, "postgres", "password",
        max_connections=10, timeout=10, enable_ssl=True,
        ssl_cert="/path/to/cert.pem", connection_pool=True,
        retry_attempts=3, compression=True, read_preference="primary"
    )
    print(connection.connect())
```
上述这段代码存在两个问题：一是构造函数里面除了构造的逻辑还包含了参数验证的逻辑，违背了责任单一原则，而是新建某个实例特别麻烦

所以我们可以构建一个中间builder类，将必须传入的参数传入builder类，做一个中间构造，每次设置入参返回self本身，设置build方法来构建初始的类，代码如下：
```python
class DatabaseConnection:
    def __init__(self, host, port, username, password,
                 max_connections=None, timeout=None,
                 enable_ssl=False,
                 ssl_cert=None, connection_pool=None,
                 retry_attempts=None,
                 compression=None,
                 read_preference=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.max_connections = max_connections
        # validate timeout
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be a positive integer")
        self.timeout = timeout
        self.enable_ssl = enable_ssl
        self.ssl_cert = ssl_cert
        self.connection_pool = connection_pool
        self.retry_attempts = retry_attempts
        self.compression = compression
        self.read_preference = read_preference

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"


class DatabaseConnectionBuilder:
    def __init__(self, host, port, username, password):
        self._config = {
            "host": host,
            "port": port,
            "username": username,
            "password": password
        }

    def set_max_connections(self, max_connections):
        self._config["max_connections"] = max_connections
        return self

    def set_timeout(self, timeout):
        if timeout <= 0:
            raise ValueError("Timeout must be a positive integer")
        self._config["timeout"] = timeout
        return self

    def set_enable_ssl(self, enable_ssl):
        self._config["enable_ssl"] = enable_ssl
        return self

    def set_ssl_cert(self, ssl_cert):
        self._config["ssl_cert"] = ssl_cert
        return self

    def set_connection_pool(self, connection_pool):
        self._config["connection_pool"] = connection_pool
        return self

    def set_retry_attempts(self, retry_attempts):
        self._config["retry_attempts"] = retry_attempts
        return self

    def set_compression(self, compression):
        self._config["compression"] = compression
        return self

    def set_read_preference(self, read_preference):
        self._config["read_preference"] = read_preference
        return self

    def build(self):
        return DatabaseConnection(**self._config)

def client():
    builder = DatabaseConnectionBuilder("localhost", 27017, "admin", "password")
    connection = (builder.set_max_connections(10).
                  set_timeout(45).
                  set_enable_ssl(True).
                  set_ssl_cert("/path/to/cert.pem").
                  set_connection_pool(True).
                  set_retry_attempts(3).
                  set_compression("snappy").
                  set_read_preference("secondaryPreferred").
                  build())
    print(connection.connect())

client()
```

### Python中的单例模式

单例模式是一种创建型设计模式，它**确保一个类只有一个实例**，并提供一个全局访问点。

比如一个服务工程代码中的日志模块，它通常需要整个代码都使用一个唯一的日志对象，也就是单例模式，看下面这个例子：
```python
class DatabaseConnection:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"



def client():
    db1 = DatabaseConnection("localhost", 3306, "root", "password")
    db2 = DatabaseConnection("localhost", 3306, "root", "password")
    print(db1 is db2)
    # 输出 False

client()
```

我们可以通过重写`__new__`方法来实现这个功能，代码如下：
```python
class DatabaseConnection:
    _instance = None

    # new是一个类方法，用于在创建实例时创建一个新对象，然后将对象交给__init__初始化
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        return f"Connected to {self.host}:{self.port} as {self.username}"



def client():
    db1 = DatabaseConnection("localhost", 3306, "root", "password")
    db2 = DatabaseConnection("localhost", 3306, "root", "password")
    print(db1 is db2)
    # True

client()
```
> 需要注意的是所有的设计模式都是一种思路，而不是一种固定的实现，上面的部分代码也只是几个符合条件的例子，切忌在自己的代码中进行生搬硬套。
