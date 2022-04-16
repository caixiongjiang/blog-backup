---
title: "Open cv模块学习"
date: 2022-03-04T18:07:05+08:00
lastmod: 2022-04-16T00:43:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/opencv.jpg"
description: "Open cv是python中专门用于图像处理的一个库，将python的基本操作写成笔记（持续更新中）"
tags:
- python
- OpenCV
categories:
- 《数字图像处理》学习笔记
series:
- 数字图像处理
comment : true
---


## 图像基本操作

### 数据读取-图像
* cv2.IMREAD_COLOR:彩色图像
* cv2.IMREAD_GRAYSCALE:灰度图像

```python
import cv2 as cv # opencv模块，读取图片的的格式是BGR
import matplotlib.pyplot as plt # 制做图表的模块
import numpy as np # 数据处理模块

img1 = cv.imread("cat.jpg") # 读取图片（彩色图片）
# 输出img1为一个三个的二维矩阵，dtype为 uint8（代表0～255之间的整数）

# 图像的显示
cv.imshow("image", img1) #参数：（图片标题，图片名）
# 等待时间，毫秒级，0表示按任意键终止图像显示
cv.watKey(0)
cv.destroyAllWindows()

# 可以将显示图像放到一个函数当中
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 获取图像矩阵的维度属性（彩色图像）
print(img1.shape) # (414,500,3) 代表3个414✖️500的二维矩阵 （GBR通道）

img2 = cv.imread("cat.jpg", cv.IMREAD_GRAYSCALE) # 使用灰度读取彩色图片
print(img2.shape) # (414,500) 代表一个414✖️500的二维矩阵 （灰度）

# 保存图像
cv.imwrite("mycat.png", img1) # 保存成功则返回一个True

print(type(img1)) # numpy.ndarray （底层格式）
print(img1.size) # 207000 (像素点的个数)
print(img1.dtype) # dtype("uint8") （数据类型）
```

### 数据读取-视频
* cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0，1
* 如果是视频文件，直接指定好路径就可以了

#### 视频读取和处理

```python
import cv2 as cv # opencv模块，读取图片的的格式是BGR
import matplotlib.pyplot as plt # 制做图表的模块
import numpy as np # 数据处理模块

vc = cv.VideoCapture("test.mp4") # 读取视频
# 检查是否正确打开
if vc.isOpened():
    # open代表是否读取成功，为bool值，frame为每一帧的图像的三维数组
    open, frame = vc.read() # vc.read()代表读取视频的每一帧（从第一帧开始读取）
else:
    open = False

# 用黑白播放读取的彩色视频
while open:
    ret, frame = vc.read()
    if frame is None: # 为空则break
        break
    if ret == True:
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 将彩色图像转化成灰度图
        cv.imshow("result", gray) # 
        if cv.waitKey(10) & OxFF == 27: # waitKey的参数代表等待的时间（数值越大，播放的速度越慢）  OxFF代表退出键为27
            break
vc.release()
cv.destroyAllWindows()
```
#### 截取部分图像数据

```python
img = cv.imread("cat.jpg")
cat = img[0:200, 0:200] # 截取前200✖️200的图片区域
cv_show("cat", cat) # 展示图片的函数（前面已经介绍）
```

#### 颜色通道提取

```python
b,g,r = cv.split(img)
print(b.shape)
print(g.shape)
print(r.shape)
# 结果都为 414✖️500
img = cv.merge((b,g,r)) # 重新合成
print(img.shape) # （414，500，3）

# 只保留R
cur_img = img.copy() # 图像的拷贝
cur_img[:,:,0] = 0 # B通道置0
cur_img[:,:,1] = 0 # G通道置0
cv_show("R", cur_img)

# 只保留G
cur_img = img.copy() # 图像的拷贝
cur_img[:,:,0] = 0 # B通道置0
cur_img[:,:,2] = 0 # R通道置0
cv_show("G", cur_img)

# 只保留B
cur_img = img.copy() # 图像的拷贝
cur_img[:,:,1] = 0 # G通道置0
cur_img[:,:,2] = 0 # R通道置0
cv_show("B", cur_img)
```

#### 边界填充

边界填充指的是对读取的图像的边界进行扩充，在很多图像处理的算法中都能用到！

```python
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50) # 指定上下左右填充的大小

    img = cv.imread("woman.jpg")
    replicate = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_WRAP)
    constant = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=1)

    plt.subplot(231), plt.imshow(img, "gray"), plt.title("ORIGINAL")
    plt.subplot(232), plt.imshow(replicate, "gray"), plt.title("REPLICATE")
    plt.subplot(233), plt.imshow(reflect, "gray"), plt.title("REFLECT")
    plt.subplot(234), plt.imshow(reflect101, "gray"), plt.title("REFLECT_101")
    plt.subplot(235), plt.imshow(wrap, "gray"), plt.title("WRAP")
    plt.subplot(236), plt.imshow(constant, "gray"), plt.title("CONSTANT")

    plt.show()
```

结果：
5种边界填充的结果：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/pic1.png)

* BORDER_REPLICATE:复制法，也就是复制最边缘的像素
* BORDER_REFLECT:反射法，对感兴趣的图像中的像素在两边进行复制   demo： fedcba｜abcdefgh｜hgfedcb
* BORDER_REFLECT_101:反射法，图像边缘为对称轴，对称   demo： gfedcb｜abcdefgh｜gfedcba
* BORDER_WRAP:外包装法 demo： cdefgh｜abcdefgh｜abcdefg
* BORDER_CONSTANT:常量法， 常数值填充

## 图像阈值

函数：
* ret, dst = cv2.threshold(src, thresh, maxval, type)
    * src: 输入图，**只能输入单通道图像，通常为灰度图。**
    * dst: 输出图
    * thresh: 阈值
    * maxval: 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    * type: 二值化操作的类型，包括以下5种类型：

    |类型|含义|
    |:--|:--|
    |cv2.THRESH_BINARY|超过(大于)阈值部分取maxval(最大值)，否则取0|
    |cv2.THRESH_BINARY_INV|THRESH_BINARY的反转|
    |cv2.THRESH_TRUNC|大于阈值部分设为阈值，否则不变|
    |cv2.THRESH_TOZERO|大于阈值部分不改变，否则为0|
    |cv2.THRESH_TOZERO_INV|THRESH_TOZERO的反转|

```python
import cv2 as cv # opencv模块，读取图片的的格式是BGR
import matplotlib.pyplot as plt # 制做图表的模块

img = cv.imread("woman.jpg")
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ["Original Image", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

```

结果：

使用图像阈值5种不同方式的处理结果：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/pic2.png)

## 图像平滑

与卷积相关，也就是对图像进行滤波。下面的介绍都以3×3的矩阵为例。

### 均值滤波和方框滤波
效果几乎相同，都是对3×3矩阵做平均卷积操作。

具体实现就是通过构建一个全是1的3×3矩阵和图像中的3×3矩阵进行求内积的操作

### 高斯滤波

我们都知道高斯函数图像是一个比较陡峭的钟的形状。也就是差值却接近0，其值越大，应用到图像中就是（3×3矩阵），周围像素点的值与中间像素点的值差值越小，它的权重就应该越大（发挥效果越好）。

**具体实现就是通过构建一个根据高斯函数构建的3×3权重矩阵和图像中的3×3矩阵求内积！**

### 中值滤波

将3×3矩阵的像素值进行排序，取中间值。注意，中值滤波的取的n×n矩阵中（n只能为奇数）。

```python
import cv2 as cv # opencv模块，读取图片的的格式是BGR
import matplotlib.pyplot as plt # 制做图表的模块

# 显示图像
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():

    img = cv.imread("color_woman_zaosheng.png")
    cv_show("img", img)
    b, g, r = cv.split(img)
    img_rgb = cv.merge([r, g, b])

    #均值滤波：进行平均卷积操作(就是在3×3的矩阵里做平均值操作)

    blur = cv.blur(img, (3, 3))
    cv_show("blur", blur)
    b, g, r = cv.split(blur)
    blur_rgb = cv.merge([r, g, b])

    #方框滤波
    #基本与均值滤波一样，可以选择归一化(最后一个参数)，防止越界
    box = cv.boxFilter(img, -1, (3, 3), normalize=True)
    cv_show("box", box)
    b, g, r = cv.split(box)
    box_rgb = cv.merge([r, g, b])

    # 发生越界就会将maxVal（255）作为结果
    box_wrong = cv.boxFilter(img, -1, (3, 3), normalize=False)
    cv_show("box_wrong", box_wrong)
    b, g, r = cv.split(box_wrong)
    box_wrong_rgb = cv.merge([r, g, b])

    #高斯滤波
    #高斯滤波的卷积和里的数值是满足高斯分布的，相当于更重视中间的
    aussian = cv.GaussianBlur(img, (5, 5), 1)
    cv_show("aussian", aussian)
    b, g, r = cv.split(aussian)
    aussian_rgb = cv.merge([r, g, b])

    #中值滤波
    #相当于用中值替代
    median = cv.medianBlur(img, 5)
    cv_show("mediam", median)
    b, g, r = cv.split(median)
    median_rgb = cv.merge([r, g, b])

    titles = ["Original Image", "blur", "box", "box_wrong", "Gaussian", "median"]
    images = [img_rgb, blur_rgb, box_rgb, box_wrong_rgb, aussian_rgb, median_rgb]

    #使用plt展示图像必须将图片通道转为rgb
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
```

结果：

5种滤波的结果：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/pic3.png)

中值滤波的效果最好！

*如果想不转通道用cv2输出对比图，可以使用直接打印图像的array数组(demo)：*

```python
res1 = np.hstack((blur, Gaussian, median))#横着拼接
res2 = np.vstack((blur, Gaussian, median))#竖着拼接
print(res1)
print(res2)
```

## 腐蚀操作和膨胀操作

* 腐蚀操作一般用于二值图像。其作用在于腐蚀图像的边界，比如很粗的文字可以通过腐蚀操作把它变细，同时可以将其边缘的毛刺直接去掉。

* 膨胀操作其实就是腐蚀操作的逆过程。

**使用腐蚀操作和膨胀操作可以去掉二值图像边缘的毛刺。**

```python
import cv2 as cv # opencv模块，读取图片的的格式是BGR
import matplotlib.pyplot as plt # 制做图表的模块
import numpy as np # 数据处理模块

# 显示图像
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = cv.imread("wenzi.jpg")
    #cv_show("原图",img)

    '''
    腐蚀操作
    cv2.erode(img, size, iterations)
    size代表用于腐蚀的矩阵大小
    ierations = n    代表迭代腐蚀n次
    '''
    kernel = np.ones((5, 5), np.uint8)
    erosion1 = cv.erode(img, kernel, iterations=1)
    erosion2 = cv.erode(img, kernel, iterations=2)
    erosion3 = cv.erode(img, kernel, iterations=3)
    ans1 = np.hstack((img, erosion1))
    ans2 = np.hstack((erosion2, erosion3))
    res1 = np.vstack((ans1, ans2))
    cv_show("腐蚀不同次数的结果", res1)

    '''
    膨胀操作
    cv2.erode(img, size, iterations)
    size代表用于膨胀的矩阵大小
    ierations = n    代表迭代膨胀n次
    '''
    img1 = cv.dilate(erosion1, kernel, iterations=1)
    res2 = np.hstack((img, img1))
    cv_show("腐蚀一次+膨胀一次的结果", res2)


if __name__ == '__main__':
    main()
```

结果：
* 1.从左上到右下分别是原图，迭代腐蚀1次，迭代腐蚀2次，迭代腐蚀3次的结果

不同迭代次数的腐蚀效果：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/pic4.jpg)

* 左边为原图，右边为腐蚀一次+膨胀一次的结果

腐蚀+膨胀的结果：![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/pic5.jpg)

可以看到虽然部分变得模糊了一点，但横字左边的毛刺基本都被去除了。如果对图像的清晰度有要求，可以调整size的大小，越小每次腐蚀的东西更少，图像细节保留的越多！

注意：*要实现图像去毛刺的效果，腐蚀的迭代次数一定要达到保证图像的基本轮廓，还要基本看不到边缘的毛刺，再进行膨胀操作，而且膨胀操作的size大小和迭代次数要和腐蚀的一样！*