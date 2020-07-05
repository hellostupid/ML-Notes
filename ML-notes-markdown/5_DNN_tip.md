> 本文的主要思路：针对training set和testing set上的performance分别提出针对性的解决方法 1、在training set上准确率不高：   new activation function：ReLU、Maxout   adaptive learning rate：Adagrad、RMSProp、Momentum、Adam 2、在testing set上准确率不高：Early Stopping、Regularization or Dropout

### Recipe

#### three steps of deep learning

做深度学习也遵循这三个步骤：
+ define a set of function，
+ goodness of function，找到loss function
+ pick the best function，找到使loss最小化的参数

overfitting是指模型在训练集上表现良好，但在测试集上表现却很差的现象。

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607122016217.png" alt="image-20200607122016217" style="zoom:50%;" />

#### Do not always blame Overfitting

在下图中，我们展示了一个20层和56层的network，在训练集和测试集上的error。黄色表示20层network，红色表示56层network。

由于模型在训练集上的表现，20层的network表现得比较好，有同学就认为这是overfitting，但其实这并不是overfitting问题，因为这个模型在训练集上的表现，也是20层的network表现好

之所以出现这个20层和56层network表现（都是20层network表现好），是由于模型的训练没有训练好

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607123330505.png" alt="image-20200607123330505" style="zoom:50%;" />

#### Different approaches for different problems.

在网络的训练过程中，我们要针对网络的不同问题提供不同的解决方法，主要有两个问题

+ 在training data上表现不好
+ 在testing data上表现不好

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607123926844.png" alt="image-20200607123926844" style="zoom:50%;" />

### Good Results on Training Data?

要在training data上获得好的结果，可以使用new activation function和adaptive learning rate

#### New Activation Function

##### deeper is better ？

在1980年代，network中主要使用sigmoid funcion作为激活函数，从下图中我们可以看出，使用sigmoid function并不能保证网络结构越深，训练结果越好

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607124417000.png" alt="image-20200607124417000" style="zoom:50%;" />

##### Vanishing Gradient Problem

出现上面这个问题的原因并不是overfitting，而是vanishing gradient（梯度消失）

当网络层数很深的时候，在靠近input layer的位置，常常会有很小的 gradient，学习速度也很慢；而在靠近output layer的地方，常常会有更大的gradient，学习速度也会很快，很快就到了converge（收敛）了

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607125209255.png" alt="image-20200607125209255" style="zoom:50%;" />

下面将叙述出现这个问题的原因。对于下图的中sigmoid function，输出范围为[0,1]，对于很大的输入，输出往往会被压缩成一个较小的值

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607125550128.png" alt="image-20200607125550128" style="zoom: 50%;" />

对于loss function对其中一个参数w的微分$\frac{\partial l}{\partial w}$，这里我们将其表达式写为$\Delta w$，可以表示当前参数w对结果loss的影响，当把这个参数w进行变化时，对loss的影响会有多大

如下图所示，如果我们输入一个很大的$\Delta w$，在经过sigmoid function运算之后，其值就缩小一次；当经过后面多层的压缩之后，$\Delta w$的值就变得越来越小;......；因此gradient在input layer附近的值会很大，但在output layer附近的值经过多次的压缩就变得很小了。

当缩小的gradient达到我们设置的那个临界值，即gradient接近于0，就发生了梯度消失问题

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607125910525.png" alt="image-20200607125910525" style="zoom:50%;" />

要解决这个问题，可以修改一下network中用到的激活函数

##### ReLU

我们选取ReLU函数的原因有以下几个：
+ 可以快速计算，不管是函数值还是对应的梯度
+ 结合了生物上的一些观察
+ 无穷多个不同bias的sigmoid function叠加的结果可以变成ReLU
+ <u>可以解决梯度消失问题</u>；

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607131402741.png" alt="image-20200607131402741" style="zoom:50%;" />

使用ReLU函数之后，代入具体的网络结构

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607131816149.png" alt="image-20200607131816149" style="zoom:50%;" />

对于input为0的值，network将不再计算其相对应的weight，而对于input不为0的值，就相当于一个线性函数$y=x$。这样做可以简化网络结构，网络结构中也就不存在gradient很小的neural

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607131920750.png" alt="image-20200607131920750" style="zoom:50%;" />

Q：但这时出现了一个新问题，ReLU函数在z=0这一点是不可导的，那么我们在根据loss function如何来计算gradient呢？

A：这里我们将输入z<0的数的gradient看作0，相当于从network中抹去了这部分神经元；对于z>=0的数，gradient=1

##### ReLU - variant

对于ReLU，当x<=0时，函数的输出值就为0了，网络中的参数也没办法更新。因此，就有学者提出了Leaky ReLU，当x<=0时，函数的输出值不是0，而是乘以一个系数0.01，这时的函数就称作**Leaky ReLU**

还有另外一种ReLU函数的变体，**Parametric ReLU**，前面乘上的系数也可以进行训练

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607143129117.png" alt="image-20200607143129117" style="zoom:50%;" />

##### Maxout

> ReLU is a special cases of Maxout

Maxout的主要思想是：让network自己去学习对应的activation function，可以学习出ReLU，也可以是其他的activation function

Maxout激活函数是对前几个神经元取最大值，再输出相应的最大值

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607144717796.png" alt="image-20200607144717796" style="zoom:50%;" />



**Maxout->ReLU**

在下图中，对于左图中的ReLU function
+ input为蓝色直线，表示$z=wx+b$，
+ output：当z<0时，ReLU也输出为0；当z>0时，ReLU的图像和z是一致的

  <img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607144943655.png" alt="image-20200607144943655" style="zoom:50%;" />

对于右图中的Maxout function，$z_1$对应的权重是$w,b$，而$z_2$对应的权重则是0，
+ input为$z_1,z_2$，
  + 蓝色直线表示$z_1=wx+b$，
  + 红色表示$z_2=0$
+ 这时neural的output为$max\{z_1,z_2\}$，输出则是和ReLU一致的（图中绿色直线）

**Maxout->more than ReLU**

maxout不仅可以学习ReLU，也可以学习其他的activation function

对于右图中的新的输入，$z_2$对应的权重则变成了$w',b'$，那么相应的input和output为
+ input，$z_1=wx+b,z_2=w'x+b'$，分别对应图中蓝色和绿色直线；
+ output为$max\{z_1,z_2\}$，表现为图中绿色直线

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607150829520.png" alt="image-20200607150829520" style="zoom:50%;" />

这时我们得到的activation function就是图中的绿色直线，是通过网络training出来的，训练的参数为$w,w',b,b'$，训练结束即可得出我们的activation function

**Summary**

这里我们先对maxout做一个总结，maxout是一个可学习的activation function，可以学习出任何分段线性凸函数（piecewise linear convex function），具体的分段数取决于在group中的元素个数

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607151653515.png" alt="image-20200607151653515" style="zoom:50%;" />

对于maxout函数的训练，如果是下图中的网络结构，我们假设已经知道$z_1^1,z_4^1,z_2^2,z_3^2$为对应的最大值，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607152054887.png" alt="image-20200607152054887" style="zoom:50%;" />

那么network可以再次被化简，neural也可以变少

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607152242384.png" alt="image-20200607152242384" style="zoom:50%;" />

#### Adaptive Learning Rate

##### Review

**Adagrad**



<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607152520774.png" alt="image-20200607152520774" style="zoom:50%;" />

**RMSprop**

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607153455066.png" alt="image-20200607153455066" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607153507651.png" alt="image-20200607153507651" style="zoom:50%;" />

gradient为0的点可以是local minimum，也可以是saddle point，还可以是在很平缓的plateau中的某些点

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607153541752.png" alt="image-20200607153541752" style="zoom:50%;" />

而在物理世界，物体本身是带有momentum的，再加上gradient的作用，就很可能可以跳出saddle point，继续训练

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607153719322.png" alt="image-20200607153719322" style="zoom:50%;" />

##### Momentum

图中蓝色箭头表示Movement（前进方向），红色箭头表示gradient的方向，绿色虚线表示上一次movement对本次的影响（惯性）

再$\theta^0$处，movement为$v^0=0$；对于在$\theta^1$处的前进方向，先计算出在$\theta^0$处的梯度$\Delta L(\theta^0)$，我们要移动的方向是由上一个时间点的gradient为$\Delta L(\theta^0)$和前进方向$v_0$决定的，即
$$
v^1=\lambda v^0-\eta \Delta L(\theta^0)=-\eta \Delta L(\theta^0)
$$
其中$\lambda$也是一个可以手动调整的参数

对于下一个时间点的移动方向$v^2$，是和当前时间节点的移动方向和梯度$v^1,\Delta L(\theta^1)$决定的，即
$$
\begin{aligned}
v^2&=\lambda v^1-\eta \Delta L(\theta^1)\\
&=-\lambda\eta \Delta L(\theta^0)-\eta \Delta L(\theta^1)
\end{aligned}
$$
<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607154125395.png" alt="image-20200607154125395" style="zoom:50%;" />

再回到之前的例子，红色箭头表示gradient的反方向，绿色表示momentum的方向，蓝色箭头表示受到gradient和momentum影响后的真实运动方向

初始点的momentum值为0；在下一个plateau上的点，虽然gradient的值很小很小，但由于受到上一个很大的momentum的影响，真实的movement还是向前的，步长也没有因为gradient的变小而变得很小；

如果我们现在走到了local minimum，此时gradient=0，此时由于momentum的影响，如果momentum的值足够大，还会继续向前运动

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607161327324.png" alt="image-20200607161327324" style="zoom:50%;" />

##### Adam

> RMSProp + Momentum
>
> Adam其实就是结合了RMSProp和Momentum思想的方法

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607162646245.png" alt="image-20200607162646245" style="zoom: 67%;" />

先将动量momentum和初始移动方向初始化为0，即$m_0=0,v_0=0$，$v_0$表示RMSProp中分母上的参数$\sigma$

计算在t时的梯度$g_t$，
$$
g_t=\Delta_\theta f_t(\theta_{\theta-1})
$$
根据上一个时间点的要走的方向$m_t$和gradient，因此t时的移动方向为$m_t$------Momentum
$$
m_t=\beta_1\cdot m_{t-1}+(1-\beta_1)\cdot g_t
$$
根据上一个时间点的移动方向$v_{t-1}$和gradient，则此时的真实移动方向$v_t$为-------RMSprop
$$
v_t=\beta_1\cdot v_{t-1}+(1-\beta_2)\cdot g_t
$$
该算法还进行了bias corrected，
$$
\hat m_t=\frac{m_t }{(1-\beta_1^t)},\quad
\hat v_t=\frac{v_t}{(1-\beta_2^t)}
$$
将进行了bias corrected的参数再输入公式，更新参数
$$
\theta_t=\theta_{t-1} -\alpha \cdot \hat m _t/\sqrt{\hat v _t}+\epsilon
$$

### Good Results on Testing Data?

#### Early Stopping

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607163018382.png" alt="image-20200607163018382" style="zoom:50%;" />

#### Regularization

##### L2 Regularization

正则化就引入了一个新的loss function，加上了一个新的正则项，这个正则项将所有需要training的参数都包括进来了，通常不包括bias

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607163135769.png" alt="image-20200607163135769" style="zoom:50%;" />

这个新的loss function再对w求偏微分，对参数进行更新
$$
w^{t+1}=(1-\eta\lambda)w^t-\eta \frac{\partial L}{\partial w}
$$
<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607163452933.png" alt="image-20200607163452933" style="zoom:50%;" />

与原来的参数更新公式相比，可以发现$w^t$前面多了一项$(1-\eta\lambda)$，通常这个$\eta,\lambda$都是很小的值，这里我们假设$(1-\eta\lambda)$是很接近于1的值，约等于0.99; regularization所做的事就是，在每次更新参数时，全都在前面乘上了一个小于1的数，在经过若干次的训练之后，$(1-\eta\lambda)w^t$的值就很接近0了

虽然$(1-\eta\lambda)w^t$的值每次都会变得越来越小，但参数更新的公式中，后面还有另外一项$\eta \frac{\partial L}{\partial w}$，会使得梯度的值不会变成0，达到平衡

##### L1 Regularization

既然L2可以作为正则项，L1也可以作为正则项，正则项为参数的绝对值相加。对这些参数求导，当$w_i$大于0时，gradient=1，当$w_i$小雨0时，gradient=-1，即为$sgn(w)$函数

L1的参数更新公式为
$$
w^{t+1}=w^t-\eta \frac{\partial L}{\partial w} - \eta\lambda\ \rm{sgn}(w^t)
$$
<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607171442670.png" alt="image-20200607171442670" style="zoom:50%;" />

与原来的参数更新公式相比较，可以发现后面多了一项$- \eta\lambda\ \rm{sgn}(w^t)$，表示参数在原来的基础上都要减去一个小于1的数

##### L1 vs L2 

参数更新公式分别如下，
$$
L1:\ w^{t+1}=w^t-\eta \frac{\partial L}{\partial w} - \eta\lambda\ \rm{sgn}(w^t)\\
L2:\ w^{t+1}=(1-\eta\lambda)w^t-\eta \frac{\partial L}{\partial w}
$$

+ L1参数更新公式每次都多**减去了一个小于1的固定值（constant）**
+ L2参数更新公式中每次都将**前一次的参数乘上一个小于1的值**

#### Dropout

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173403475.png" alt="image-20200607173403475" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173419426.png" alt="image-20200607173419426" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173455150.png" alt="image-20200607173455150" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173629804.png" alt="image-20200607173629804" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173828093.png" alt="image-20200607173828093" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173729715.png" alt="image-20200607173729715" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200607173943611.png" alt="image-20200607173943611" style="zoom:50%;" />

