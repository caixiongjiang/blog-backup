---
title: "爬虫：使用selenium模拟12306购票"
date: 2022-04-28T10:07:05+08:00
lastmod: 2022-04-28T10:08:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%88%AC%E8%99%AB/selenium.png"
description: "本人学习爬虫的目的是为了了解网页结构，正则表达式以及一些爬虫工具的使用"
tags:
- python
- selenium
categories:
- 爬虫
comment : true
---

## 使用selenium模拟12306购票

### selenium介绍

* selenium是一种自动化测试工具，可以通过浏览器引擎来自动化帮人类做某些事。
* selenium也可以作为一种轻量化爬虫工具。优点是它能绕过网站自身加密后的源码，通过浏览器解析后来获取网页中的元素，也就是文本信息；缺点也很明显，就是它本身是通过浏览器去运行的，非常容易受浏览器访问时网络波动的影响，因此通常要设置睡眠短暂时间来等待浏览器的加载，整体效率就不高。
* 总体来说，selenium适合轻量化数据的爬虫。

### 本小程序原理介绍

* 通过python使用selenium自动操控浏览器引擎模拟人的购票动作。
* 通过第三方图像识别破解复杂验证码。（以前的12306登录有这个环节，现在取消了）
* 本小程序没有涉及UI界面的设计，只为学习爬虫工具，粗浅了解网页结构。

### 资源准备

* 1.谷歌浏览器（你也可以使用火狐浏览器）

* 2.下载谷歌浏览器驱动：
  * 查看浏览器的版本：点击浏览器右上角的`三个点按钮`，找到`帮助`中的`关于Google Chrome`，点击就可以看到自己的版本号了
  * 打开网址：[https://registry.npmmirror.com/binary.html?path=chromedriver/](https://registry.npmmirror.com/binary.html?path=chromedriver/),选择浏览器版本对应的镜像，根据操作系统来选择下载哪一个镜像
  * 下载解压后，将文件改为`chromedriver`，放入python解释器所在的文件夹，也就是你python环境配置的地方，这里我使用的`PyCharm`,所以把它放在了我的项目目录下的bin文件中。
  
* 3.下载`selenium`库（哪种ok就哪种）:

  ```shell
   pip install selenium
   pip install selenium -i 清华源
  ```

### 破解12306检测自动化控制

* 12306的滑窗验证会自动检测自动化测试，导致机器滑窗验证失败。

* 反识别自动化控制工具使用：

  * 如果你的chrome版本小于88，在启动浏览器的时候（此时没有加载任何内容），向页面潜入js代码，去掉webdriver：

    ```python
    from selenium.webdriver import Chrome
    web = Chrome()
      
    web.execute_cdp_cmd("Page.addScriptToEvaluateOnDocument", {
    	"source" : """
    	navigator.webdriver = undefined
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
      })  
    ```

  * 如果你的chrome版本大于88，引入option:

    ```python
    from selenium.webdriver import Chrome
    #导入驱动选项包
    from selenium.webdriver.chrome.options import Options
    #去掉自动化标志(滑块能监测到自动化工具)
    option = Options() #创建一个Options对象
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    option.add_argument('--disable-blink-features=AutomationControlled')
    ```

### 程序整体步骤

* 输入账号，密码，实现自动登录，并跳转到首页页面。
* 输入出发地和目的地，时间，是否为学生票，进行搜索。
* 输入要购买的票类型，爬取该种票类型的信息，提示输入要选择的序号，选择后并进行预订
* 选择学生票，座位类型（自动选择，前面已经输入），以及座位号，点击预订

### 程序源码

通过一个类来实现和几个类内方法来实现：

```python
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver import ActionChains
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.select import Select #引入select包，为了下拉列表使用
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class Login12306():
    # Chrome浏览器绕过webdriver检测，可以成功滑窗
    # 这种方式浏览器没监测到是自动测试工具
    def __init__(self, url, login_user, login_passwd):
        option = ChromeOptions()
        option.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.driver = webdriver.Chrome(options=option)
        self.driver.get(url)
        script = 'Object.defineProperty(navigator, "webdriver", {get:()=>undefined,});'
        self.driver.execute_script(script)
        self.driver.maximize_window() #最大化窗口
        self.login_user = login_user
        self.login_passwd = login_passwd
        self.init_url = url

    def Login(self):
        # 点击选择账号登录
        wait = WebDriverWait(self.driver, 60, 0.1)
        account_login = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.login-hd-code a')))#J-password
        account_login.click()

        # 输入用户名密码
        user_input = self.driver.find_element(By.XPATH, '//*[@id="J-userName"]')
        user_input.send_keys(self.login_user)
        time.sleep(1)
        passwd_input = self.driver.find_element(By.XPATH, '//*[@id="J-password"]')
        passwd_input.send_keys(self.login_passwd)
        time.sleep(1)
        login_btn = self.driver.find_element(By.XPATH, '//*[@id="J-login"]')
        login_btn.click()

        # 滑窗验证处理
        time.sleep(1)
        span = self.driver.find_element(By.XPATH, '//*[@id="nc_1_n1z"]')
        actions = ActionChains(self.driver)
        actions.click_and_hold(span)
        actions.drag_and_drop_by_offset(span, 350, 0)
        actions.perform()

        # 通过url变化与否判断是否登录成功
        time.sleep(2)
        print('current_url: ', self.driver.current_url)
        if self.driver.current_url == self.init_url:
            # 网页没跳转, 判断是否提示错误信息
            err_login = self.driver.find_element(By.CSS_SELECTOR, 'div.login-error')
            if err_login:
                if err_login.is_displayed():
                    print('Login Error!')
        else:
            try:
                # 登录成功后，关闭弹出的对话框
                modal = self.driver.find_element(By.CSS_SELECTOR, 'div.modal')
                confirm_btn = self.driver.find_element(By.CSS_SELECTOR, 'div.modal > div.modal-ft > a')
                confirm_btn.click()
                print("登录成功")
            except NoSuchElementException:
                print('NoSuchElementException')
            self.driver.find_element(By.XPATH, '//*[@id="J-index"]/a').click()

    def Purchase(self):
        #点击跳到首页
        departure = input("请输入出发地：")
        destination = input("请输入目的地：")
        date = input("请输入日期（年-月-日）：")
        isStudent = input("是否为学生票（输入Y代表是，其他代表不是）：")
        print("座位类型：特等座/商务座  一等座 二等座/二等包座 高级软卧 软卧/一等卧 动卧 硬卧/二等卧 软座 硬座 无座 其他")
        type = input("请输入座位类型：")
        if isStudent == "y" or isStudent == "Y":
            flag = True
        else:
            flag = False
        #time.sleep(5)
        self.driver.find_element(By.XPATH, '//*[@id="toolbar_Div"]/div[3]/div[2]/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[1]/label').click()
        self.driver.find_element(By.XPATH, '//*[@id="fromStationText"]').send_keys(departure, Keys.ENTER)
        #time.sleep(2)
        self.driver.find_element(By.XPATH, '//*[@id="toolbar_Div"]/div[3]/div[2]/div/div[1]/div/div[2]/div[1]/div[1]/div[1]/div[2]/label').click()
        self.driver.find_element(By.XPATH, '//*[@id="toStationText"]').send_keys(destination, Keys.ENTER)
        #time.sleep(2)
        #先将默认值清空,再输入
        self.driver.find_element(By.XPATH, '//*[@id="train_date"]').clear()
        self.driver.find_element(By.XPATH, '//*[@id="train_date"]').send_keys(date)
        #time.sleep(2)
        if flag:
            self.driver.find_element(By.XPATH, '//*[@id="isStudentDan"]').click()
        #time.sleep(2)
        self.driver.find_element(By.XPATH, '//*[@id="search_one"]').click()

        #转换到最后一个页面
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.find_element(By.XPATH, '//*[@id="qd_closeDefaultWarningWindowDialog_id"]').click()

        #点击到能预定的车次
        self.driver.find_element(By.XPATH, '//*[@id="sear-result"]/span/label[2]').click()
        # 解析车次信息
        WebDriverWait(self.driver, 1000).until(
            EC.presence_of_all_elements_located((By.XPATH, "//tbody[@id='queryLeftTable']/tr"))
        )
        tran_trs = self.driver.find_elements(By.XPATH, "//tbody[@id='queryLeftTable']/tr[not(@datatran)]")

        infos = [[] for y in range(40)] #定义一个二维数组，y的维度定为40，因为1天的车次一般不会超过40

        types = ["特等座/商务座", "一等座", "二等座/二等包座", "高级软卧", "软卧/一等卧", "动卧", "硬卧/二等卧", "软座", "硬座", "无座", "其他"]
        for i in range(len(types)):
            if types[i] == type:
                type_id = i + 7 #types[0]对应字段infos[index - 1][7]的数据
                break
        '''
        注意infos存储的是网页上所有车次的信息，没有筛选座位类型的结果
        '''
        index = 1 #网页上真实的次序
        for tran_tr in tran_trs:
            infos[index - 1] = tran_tr.text.replace('\n', ' ').split(' ')
            #去除多余字段
            if infos[index - 1][1] == "复":
                infos[index - 1].remove("复")
            if infos[index - 1][type_id] == "--" or infos[index - 1][type_id] == "无":
                index += 1
                continue
            #打印有此座位类型的余票信息
            print("=============================================================================================================================================================================")
            print("序号：", end = "")
            print(str(index), end="")
            print("  车次：" + infos[index - 1][0], end="")
            print("  出发站-到达站：" + infos[index - 1][1] + "-" + infos[index - 1][2], end="")
            print("  出发时间-到达时间：" + infos[index - 1][3] + "-" + infos[index - 1][4], end="")
            print("  历时：" + infos[index - 1][5] + " " + infos[index - 1][6], end="")
            print("  特等座/商务座：" + infos[index - 1][7], end="")
            print("  一等座：" + infos[index - 1][8], end="")
            print("  二等座/二等包座：" + infos[index - 1][9], end="")
            print("  高级软卧：" + infos[index - 1][10], end="")
            print("  软卧/一等卧：" + infos[index - 1][11], end="")
            print("  动卧：" + infos[index - 1][12], end="")
            print("  硬卧/二等卧：" + infos[index - 1][13], end="")
            print("  软座：" + infos[index - 1][14], end="")
            print("  硬座：" + infos[index - 1][15], end="")
            print("  无座：" + infos[index - 1][16], end="")
            print("  其他：" + infos[index - 1][17])
            index += 1

        while(1):
            select = input("请输入要选择的班次序号：")
            if int(select) > index:
                print("请输入正确的次序号！")
                continue
            break

        #点击预定
        btns = self.driver.find_elements(By.CLASS_NAME, 'btn72')
        for i in range(len(btns)):
            if i == int(select) - 1: #select从1开始，i从0开始
                btns[i].click()
                break
        time.sleep(1)
        #如果操作时间过长就要继续进行验证（虽然前面已经登录了还是要验证）

        #输入用户名和密码,点击登录
        # self.driver.find_element(By.XPATH, '//*[@id="J-userName"]').send_keys(self.login_user)
        # self.driver.find_element(By.XPATH, '//*[@id="J-password"]').send_keys(self.login_passwd)
        # self.driver.find_element(By.XPATH, '//*[@id="J-login"]').click()
        # time.sleep(0.5)

        #滑块验证
        # span = self.driver.find_element(By.XPATH, '//*[@id="nc_1_n1z"]')
        # actions = ActionChains(self.driver)
        # actions.click_and_hold(span)
        # actions.drag_and_drop_by_offset(span, 350, 0)
        # actions.perform()


        #选择票类型和乘客类型
        if flag: #(学生票)
            self.driver.find_element(By.XPATH, '//*[@id="normal_passenger_id"]/li[1]/label').click()
        self.driver.find_element(By.XPATH, '//*[@id="dialog_xsertcj_ok"]').click()

        #下拉框选择
        options_element = self.driver.find_element(By.XPATH, '//*[@id="seatType_1"]')
        options = Select(options_element)
        for i in range(len(options)):
            select.select_by_visible_text(type)

        #提交订单
        self.driver.find_element(By.XPATH, '//*[@id="submitOrder_id"]').click()
        print(" 窗 | A | B | C | 过道 | D | F | 窗 ")
        dest_el = self.driver.find_elements(By.CSS_SELECTOR, 'rect')
        while(1):
            dest = input("请选择你的座位序号：")
            if dest == 'A':
                dest_el[0].click()
            elif dest == 'B':
                dest_el[1].click()
            elif dest == 'C':
                dest_el[2].click()
            elif dest == 'D':
                dest_el[3].click()
            elif dest == 'F':
                dest_el[4].click()
            else:
                print("请输入正确的座位")
                continue
            break
        time.sleep(1)
        #最后确认
        self.driver.find_element(By.XPATH, '//*[@id="qr_submit_id"]').click()
        print("订票成功！")

def main():
    url = 'https://kyfw.12306.cn/otn/resources/login.html'
    login_user = input("Input Username: ")
    login_passwd = input("Input Password: ")
    global login  # 避免chrome浏览器驱动在程序执行完后自动关闭浏览器
    login = Login12306(url, login_user, login_passwd)
    login.Login()
    login.Purchase()



if __name__ == '__main__':
    main()
```



### 运行结果

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%88%AC%E8%99%AB/img1_1.png)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%88%AC%E8%99%AB/img1_2.png)

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E7%88%AC%E8%99%AB/img1_3.png)

### 程序不足

* 1.因为浏览器有时会不稳定，程序运行可能会出现问题（但概率较小，基本没出现过）

* 2.因为12306是不是会给你来一个，点击预订，会出现重新登录的情况（可能是内部设定的某种机制），有时又不会出现，所以我在代码中注释掉该部分。
* 3.12306网站购票本身比较容易崩溃，因为我们是借助浏览器访问，可能会出现页面崩溃的情况
* 4.这里我们只是实现了买票，并不涉及付款部分，所以这里的买票不是真正的买票

### Tips

* 如果你想对一个外行人展示，一个装逼小技巧🐶，无头浏览器（不会弹出浏览器）实现：

  ```python
  #准备好参数配置（无头浏览器）
  opt = Options()
  opt.add_argument("--headless")
  opt.add_argument("--disable-gpu") #不显示
  
  web = Chrome(options=opt)
  ```

  **这里跟上面的避开自动控制检测是一样的方式：都是在Options对象添加属性，骗过网站的检测。**

* **郑重声明：这里只是用作个人程序学习使用，并不用作任何商业用途！大家牢记一句话：爬虫爬虫，越爬越刑！**

* 本项目源码和其他方式（BeautifulSoup4以及正则表达式的使用）的基础爬虫学习demo已经上传到Github：
    * 仓库地址：[https://github.com/caixiongjiang/py_reptile](https://github.com/caixiongjiang/py_reptile),如果可以的话，动动你的小手在右上角帮我点个star哦！😯
    * 本demo源码地址：[https://github.com/caixiongjiang/py_reptile/blob/master/12306%E7%88%AC%E8%99%AB%E6%8A%A2%E7%A5%A8/12306.py](https://github.com/caixiongjiang/py_reptile/blob/master/12306%E7%88%AC%E8%99%AB%E6%8A%A2%E7%A5%A8/12306.py)