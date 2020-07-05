#### Introduction

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610195909860.png" alt="image-20200610195909860" style="zoom:50%;" />

对于猫狗分类问题，如果只有一部分data有label，还有其他很大一部分data是unlabeled，那么我们可以认为unlabeled data对我们网络的训练是无用的吗？

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610200040869.png" alt="image-20200610200040869" style="zoom:50%;" />

Q：Why semi-supervised learning helps ?

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610200322008.png" alt="image-20200610200322008" style="zoom:50%;" />

A：如图所示，图中灰色圆点表示unlabeled data，其他圆点表示labeled data。如果没有unlabeled data，此时可以用一条竖直的线将猫狗进行分类，boundary为竖直的那条线；但unlabeled data的分布也可以告诉我们一些信息，对我们的训练也是有帮助的，有了unlabeled data，此时的boundary为斜直线

#### Semi-supervised Learning for Generative Model

##### Intuitive

不考虑unlabeled data，只有labeled data

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610201326618.png" alt="image-20200610201326618" style="zoom:50%;" />



如果把unlabeled data也考虑进来，此时的boundary 也发生了变化

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610201826012.png" alt="image-20200610201826012" style="zoom:50%;" />

##### Formulation

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610204004390.png" alt="image-20200610204004390" style="zoom:50%;" />

不同的maximum likelihood对比

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610204526467.png" alt="image-20200610204526467" style="zoom:50%;" />

#### Low-density Separation Assumption

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610214224919.png" alt="image-20200610214224919" style="zoom:50%;" />

##### Self-training

有labeled data和unlabeled data，重复以下过程：

+ 从labeled data中tarin了模型$f^*$；
+ 将$f^*$应用到unlabeled data，得到带label的数据，称为Pseudo-label
+ 从unlabeled data中移出这部分data，并加入labeled data；要移除哪部分data，要根据具体的限制条件而定
+ 有了更多的label data，就可以继续训练我们的模型，返回第一步

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610205046125.png" alt="image-20200610205046125" style="zoom:50%;" />

Q：这种训练方式对regression 有用吗 ？

W：不能，regression输出的是一个真实的值

**hard label vs soft label**

self-training用的是hard label；generative model用的是soft label

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610210239919.png" alt="image-20200610210239919" style="zoom:50%;" />

##### Entropy-based Regularization

如果输出的每个类别的概率是相近的，那么这个模型就比较bad；输出的类别差距很大，比如某个类别的概率为1，其他都是0；我们可以用$E(y^u)$来衡量
$$
E(y^u)=-\sum_{m=1}^5y_m^uln(y_m^u)
$$
对于第一个和第二个distribution，那么$E(y^u)=0$；

对于第三个distribution，那么$E(y^u)=-ln(\frac{1}{5})=ln5$

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610211113227.png" alt="image-20200610211113227" style="zoom:50%;" />

那么我们现在就可以重新设计loss function，用cross entropy来估计$y^r,\hat y ^r$之间的差距，即$C(y^r,\hat y^r)$，使用labeled data，还加上了一个regularization term
$$
L=\sum_{x^T}C(y^r,\hat y^r)+\lambda \sum_{x^u} E(y^u)
$$

##### Outlook: Semi-supervised SVM

对于unlabeled data，如果是SVM 二分类问题，可以把所有的unlabeled data都穷举为Class1或Class2，列举出所有可能的方案，再找出对应的boundary，计算loss，可以发现下图中黑色方框图具有最小的loss

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610213305033.png" alt="image-20200610213305033" style="zoom:50%;" />

#### Smoothness Assumption

##### Introduction

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610214254778.png" alt="image-20200610214254778" style="zoom:50%;" />

假设：如果x是similar的，那么他们的y也是一样的

这样的假设是非常不精确的，下面我们做出一个更加精确的假设：

+ x是分布不均匀的，有的地方很密集，有的地方很稀疏
+ $x^1,x^2$中间有个high density region，那么label $y^1,y^2$就很可能是一样的；但$x^2,x^3$中间没有high density region，其label相同的概率就非常小

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610214546977.png" alt="image-20200610214546977" style="zoom:50%;" />

对于下图中的数字，2之间是有过渡形态的，所以这两个图片是similar的；而2与3之间没有过渡形态，因此是不similar的

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610215523269.png" alt="image-20200610215523269"  />

比较直观的做法是先进行cluster，再进行label

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610221020376.png" alt="image-20200610221020376" style="zoom:50%;" />

##### Graph-based Approach

那么我们到底要怎么才能知道$x^1,x^2$到底在high density region是不是close呢 ？

我们可以把data point用图来表示，图的表示有时是比较nature，有时需要我们自己找出来point之间的联系

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610221342226.png" alt="image-20200610221342226" style="zoom:50%;" />

 **Graph Construction**

首先定义不同point之间的相似度$s(x^i,x^j)$，可以通过以下两个算法来添加edge：

+ KNN，对于图中红色的圆点，与其最相近的三个（k=3）neighbor相连接
+ e-Neighborhood，对于周围的neighbor，只有和他相似度大于1的才会被连接起来

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610221803745.png" alt="image-20200610221803745" style="zoom:50%;" />

edge并不是只有相连和不相连两种选择而已，也可以给edge一些weight，让这个weight和这两个point之间的相似度成正比

labeled data会影响他的邻居，如果这个point是class1，那么他周围的某些point也可能是class1

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610224513039.png" alt="image-20200610224513039" style="zoom:50%;" />

**Definition**

对于下图中的两幅图，如果从直观上看，我们可以认为左边的图更smooth

现在我们用数字来定量描述，S的定义如下
$$
S=\frac{1}{2}\sum_{i,j}w_{i,j}(y^i-y^j)^2
$$
根据公式我们可以算出左图的S=0.5，右图的S=3，值越小越smooth，越小越好

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610224714665.png" alt="image-20200610224714665" style="zoom:50%;" />

对原来的S进行改造一下，$S=y^TLy$

其中$L=D-W$，W为权重矩阵，D表示将weight每行的和放到对角线的位置

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610225336692.png" alt="image-20200610225336692" style="zoom:50%;" />

loss function其中一项就包括cross entropy计算的loss；smoothness的量S，前面再乘上一个可以调整的参数$\lambda$ ，$\lambda S$就表示一个regularization term

网络的整体目标是使loss function 取得最小值，即cross entropy项和smoothness都必须要达到最小值，和其他的网络一样，计算相应的gradient，做gradient descent即可

如果要计算smoothness不一定非要在output的地方，也可以是其他位置，比如hidden layer拿出来进行一些transform，或者直接拿hidden layer，都可以计算smoothness

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610225851786.png" alt="image-20200610225851786" style="zoom:50%;" />

#### Better Representation

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200610230634932.png" alt="image-20200610230634932" style="zoom:50%;" />