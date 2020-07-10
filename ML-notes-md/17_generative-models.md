> 本文主要叙述了集中generative models，包括PixelRNN、VAE，GAN（生成对抗网络）；还叙述了auto-encoder和VAE的区别，以及VAE进行改进的地方，通过高斯混合模型来对VAE的原理进行了详细介绍。

#### Overview

Generative Models可以分为三大类：pixelRNN、autoencoder、GAN；

现在machine可以对猫狗进行分类，在将来machine就可以通过自己的学习画出一只猫来。

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629141407487.png" alt="image-20200629141407487" style="zoom: 67%;" />

#### PixelRNN

##### Introduction

如果要生成一张图像，我们可以先生成这张图像的第一个红色pixel，再根据这个pixel生成下一个蓝色pixel，每个pixel可以用一个三维的vector来进行表示

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629141521860.png" alt="image-20200629141521860" style="zoom: 67%;" />

这是一个无监督学习的过程，并不需要对data进行标注

如果我们现在把下图中的狗下半身遮住，可以让machine学习出下半身的图片

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629141945697.png" alt="image-20200629141945697" style="zoom: 67%;" />

也可以用到声讯号

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629142331286.png" alt="image-20200629142331286" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629142353897.png" alt="image-20200629142353897" style="zoom:50%;" />

##### Practicing Generation Models: Pokémon Creation

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629142516414.png" alt="image-20200629142516414" style="zoom: 67%;" />

做这个实验时，有一些tips；

由于每个pixel都使用三维的vector来表示（RGB），只有三个channel的值相差特别大时，才会有颜色特别鲜明的图片，但学习结果并不能保证这一点，因此图片会会比较灰蒙蒙的，最后得出来的结果会不太好；

因此，每个pixel最好使用1-of-N encoding来表示，输入一个绿色的方块，vector中只有绿色方块对应的dimensions为1，其他都是0；但有$256\times256$种颜色，颜色种类非常繁多，可以先对类似的颜色做一个**clustering**，类似的颜色都用同一种颜色来表示，可以把颜色数量缩小到167种

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629142535096.png" alt="image-20200629142535096" style="zoom: 67%;" />

下面开始正式做实验

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629143804656.png" alt="image-20200629143804656" style="zoom: 67%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629143843226.png" alt="image-20200629143843226" style="zoom: 33%;" />

#### VAE

##### Auto-encoder

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629144022672.png" alt="image-20200629144022672" style="zoom: 67%;" />

##### VAE

VAE不像autoencoder那样，直接得出code，经过了一些变换，即$c_i=exp(\sigma_i)\times e_i+m_i$，也是来最小化reconstruction error

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629144250780.png" alt="image-20200629144250780" style="zoom: 67%;" />

minimize的目标还有
$$
\sum_{i=1}^3(exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)
$$

##### Pokémon Creation

input一个宝可梦图像，进行encoder、decoder，得到reconstruct之后的图像，其中code是10-dim的；

现在我们只选择其中的2个纬度出来，其他纬度的值都固定不变，对这个二维的坐标轴，放入不同的点，再输入对应的Decoder，观察其合成出来的image；如果使用不同维度的点，就可以观察到不同维度生成的image；根据不同的维度，我们就可以根据自己的需要，调整这些维度的值，从而得出我们想要的结果

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629150656140.png" alt="image-20200629150656140" style="zoom: 50%;" />

部分结果展示

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629151406932.png" alt="image-20200629151406932" style="zoom:50%;" />

##### Why VAE?

对于一张满月的图像，先进行encode，再进行decode，使得input和output之间的差值最小化；对半月的图像输入也如此

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629152107223.png" alt="image-20200629152107223" style="zoom: 67%;" />

对于encoder的其中一个输出$\sigma_i$，表示noise的variance，需要再输入exp函数进行计算，保证其值是大于0的，是machine自己学习的

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629153400892.png" alt="image-20200629153400892" style="zoom: 67%;" />

但只有noise是不够的，还需要加一些限制
$$
\sum_{i=1}^3(exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)
$$
在下图中，蓝色的线表示$exp(\sigma_i)$，红色的线表示$(1+\sigma_i)$，绿色的线表示两者的差值，可以发现在原点处差值是最小的，$\sigma _i$的值接近1，variance的值接近1

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629153918937.png" alt="image-20200629153918937" style="zoom:50%;" />

$(m_i)^2$为正则项regularization term，让结果不那么overfiting

在下图中，为一个高维的坐标（用一维进行了展示），曲线表示image是宝可梦的概率，可以在图中看到，对于一些很像宝可梦的image，$P(x)$值很高，但对于一些比较模棱两可的image，$P(x)$值就很低

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629154606838.png" alt="image-20200629154606838" style="zoom: 33%;" />

那么我们到底要怎么来评估这个probability distribution呢？答案是高斯混合模型

##### Gaussian Mixture Model

对于下图中的曲线图，蓝色表示多个高斯模型，黑色曲线表示这些模型通过一定的weight进行叠加之后的结果

现在有很多个gaussian model（1，2，3，4，5...），且都对应了自己的weight $P(m)$（蓝色方块），<span style="color: red">要先根据weight选对应的gaussian，再决定到底要从gaussian中sample哪些data</span> ;我们使用$P(m)$表示每个gaussian的weight，$P(x|m)$表示从对应的gaussian中选出data x的概率

x并不是代表着某个类别，而是用一个vector来进行表示，每个维度表示不同的特征，即 <u>Distributed representation is better than cluster</u>

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629155005437.png" alt="image-20200629155005437" style="zoom: 67%;" />

我们现在从正态分布中sample一个data z，z的每个dimension就表示某种attribute；

根据z我们就可以得出高斯分布mean和variance，即$\mu (z),\sigma(z)$，z有无穷多个可能，mean和variance也有无穷多个可能

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629160314318.png" alt="image-20200629160314318" style="zoom: 67%;" />

在最下方的图中，我们可以认为最中间的圆点被sample到的几率最大，其他圆点被sample到的几率就相对较小；<span style="color: red">每个z被sample到之后，根据某个function，计算出对应的mean和variance，都对应着不同的gaussian model</span>

这个function可以是一个neural network，input为z，output为$\mu(z),\sigma(z)$

由于现在是连续的z，$P(z)$的形式也发生了变化，
$$
P(x)=\mathop{\int }_zP(z)P(x|z){\rm d}z
$$

##### **Maximizing Likelihood**

现在我们需要找到z和$\mu(z),\sigma(z)$之间的关系，用network来进行计算的时候也需要一个评估手段，这里我们采用的是最大化$L=\sum_xlogP(x)$，即最大化我们已经看到的image x的likelihood；

那么我们现在就有一个NN的参数需要调整，来最大化likelihood L；

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629162138964.png" alt="image-20200629162138964" style="zoom: 67%;" />

现在我们还有另外一个distribution $q(z|x)$，和前一个是相反的，其中$z|x$表示给出x，再把x输入网络$NN'$，得到属于z的gaussian distribution，其mean和variance，即$\mu'(x),\sigma'(x)$

$P(x|z)$表示z决定了x的distribution，$q(z|x)$表示x决定了z的distribution

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629163947560.png" alt="image-20200629163947560" style="zoom:45%;" />

其中$P(z,x)=P(x)P(z|x)$

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629165559403.png" alt="image-20200629165559403" style="zoom:50%;" />

我们本来需要调整的参数是$P(x|z)$，使得$P(x)$取得最大值，从而使likelihood取得最大值；但现在的lower bound为$L_b$，是$log(P(x))$的最小值，其中$P(z)$是已知的，不知道的参数是$P(x|z),q(z|x)$，两项都需要调整

Q：这里为什么突然多了一项需要调整的参数？

A：这里并不知道lower bound和likelihood之间的关系到底是怎样的，有可能升高了lower bound，但likelihood反而下降了；引入q就可以解决这个问题

根据原式子，即$P(x)=\mathop{\int }_zP(z)P(x|z){\rm d}z$，只和$P(x|z)$有关，与$q(z|x)$是无关的，即下图中<span style="color: purple">方框</span>部分是不变的，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629170614939.png" alt="image-20200629170614939" style="zoom: 67%;" />

**如果我们现在固定$P(x|z)$，想通过$q(z|x)$来使$L_b$ 最大化，那么对应的KL divergence就会最小化，到一种很极端的情况，KL divergence会变为0；如果$L_b$继续上升，那么肯定会超出该区域，因此对应的$logP(x)$也会继续上升；KL divergence不断变小的过程，$q(z|x),P(z|x)$之间的差距也会不断缩小**

因此，这个过程不仅让likelihood变得越来越大，也会找到和$P(x|z)$接近的$q(z|x)$

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629171631388.png" alt="image-20200629171631388" style="zoom:50%;" />

对于$q(z|x)$，其中$z|x$表示给出x，再把x输入网络$NN'$，得到属于z的gaussian distribution，就可以知道z是从什么样的gaussian distribution得出来的

##### **Connection with Network**

现在我们的目标就是最小化$q(z|x),P(z)$之间的KL divergence，即使$q(z|x)$和normal distribution $P(z)$越接近越好，即minimize
$$
\sum_{i=1}^3(exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)
$$
还需要maximizing
$$
P(x)=\mathop{\int }_zP(z)P(x|z){\rm d}z
$$
<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629172034668.png" alt="image-20200629172034668" style="zoom:50%;" />

对于input的x，我们先输入网络$NN'$，得到获取z的gaussian distribution；sample得到z之后，再输入网络$NN$，得到获取x的gaussian distribution，使得新分布的mean和x越接近越好

##### KL散度

> 所谓KL散度，是指当某分布q(x)被用于近似p(x)时的信息损失。

计算公式为
$$
KL(p||q)=\sum p(x)log\frac{p(x)}{q(x)}
$$

##### Problems of VAE

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629174600157.png" alt="image-20200629174600157" style="zoom:50%;" />

#### GAN

##### The evolution of generation

v1：会generate一些奇怪的图片，然后会有一个第一代的discriminator，来辨别到底哪一个是real image；

V2： generater会根据上一次discriminater的结果进行调整，第二代生成的image就和真实的image更像了，再把image输入第二代discriminator，再进行比较；

.......

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629175524852.png" alt="image-20200629175524852" style="zoom:50%;" />

##### Discriminator

对于generator，可以看作是VAE里面的decoder，我们通常选择从某个distributionsample出来的vector，输入到第一代的generator，再生成相对应的image，有多少个vector，就生成多少个image；

把真实的image输入discriminator，与generator生成的image进行比较；并对这些data进行标注，real image标注为1，其余标注为0

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629201306815.png" alt="image-20200629201306815" style="zoom:50%;" />

##### Generator

首先随机sample一个vector，作为Generator的输入，generator会生成对应的image，再把这个image输入discriminator，输出是real image的概率，对于第一代（v1），generator还不成熟，因此概率只有0.87；generator接下来会调整自己的参数，使discriminator认为其生成的image是real image；

这generator+discriminator就相当于一个network，整体目标是使discriminator认为generator生成的image是real image，就可以根据这个指标来使用梯度下降算法，进行back propagation，来不断调整generator的参数；

<span style="color: red">注：在训练过程中应该固定discriminator的参数，来调整generator的参数</span>

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629201707553.png" alt="image-20200629201707553" style="zoom:50%;" />

##### Why GAN is hard to train?

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200629202903543.png" alt="image-20200629202903543" style="zoom:50%;" />

