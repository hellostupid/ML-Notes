考虑到宝可梦的两个属性（**Defense**、**SP Defense**），将输入的宝可梦进行属性分类（Water、Normal）

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605204445258.png" alt="image-20200605204445258" style="zoom:50%;" />

#### Gaussian Distribution

假设图中的点服从高斯分布，由于只考虑了两个属性，$\mu$为一个二维向量，则有高斯分布公式，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605202837571.png" alt="image-20200605202837571" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605204129951.png" alt="image-20200605204129951" style="zoom:50%;" />

#### Maximum Likelihood

第i个example的概率密度函数为$f_{\mu,\sum}(x^i)$，可得出最大似然函数$L(\mu,\sum)$的表达式，其中$\mu=\mu^*,\sum=\sum^*$时，函数L得到最大值

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605203041766.png" alt="image-20200605203041766" style="zoom:50%;" />

这里我们假设有两个类别，其参数和分布如下，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605210600803.png" alt="image-20200605210600803" style="zoom:50%;" />

输入样例x是Class 1:Water的概率为，
$$
P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x)}=\frac{P(x|C_1)P(C_1)}
{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}
$$
再代入相应的表达式，其中$P(x|C_1)=f_{\mu^1,\sum^1},P(x|C_2)=f_{\mu^2,\sum^2}$，即可算出x为C1的概率，如果算出这个概率大于0.5，我们就可以认为x的属性为C1

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605212347236.png" alt="image-20200605212347236" style="zoom:50%;" />

但这样子算出来的分类精确度很低，只有47%，就算加入其他的属性，精确度也只提高到了54%

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605214028329.png" alt="image-20200605214028329" style="zoom:50%;" />

#### Modifying Model

由于上面模型的精确度都不高，所以在此我们对模型进行了修改，模型的参数就只有三个，$\mu^1,\mu^2,\sum=\sum^1=\sum^2$

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605214245755.png" alt="image-20200605214245755" style="zoom:50%;" />

对于Water属性的宝可梦对应参数为$\mu^1$，Normal属性的宝可梦为$\mu^2$，共同属性为$\sum$，这时对应的最大似然函数为$L(\mu^1,\mu^2,\sum)$，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605214446675.png" alt="image-20200605214446675" style="zoom:50%;" />

修改后的模型准确率可以达到54%，如果加入更多的属性，准确率可以提高到73%

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605215021833.png" alt="image-20200605215021833" style="zoom:50%;" />

#### Three Steps

这里再回忆一下三个步骤：

（1）Function Set，计算分类为该类的概率$P(C_1|x)$，如果大于0.5，则认为类别为1，否则为类别2

（2）找出相对应的$\mu,\sum$，使得似然函数L取得最大值；

（3）得出使似然函数最大化的参数，$\mu^*,\sum^*$
$$
u^*=\frac{1}{79}\sum_{n=1}^{79}x^n,\quad\sum^*=\frac{1}{79}\sum_{n=1}^{79}(x^n-\mu^*)(x^n-\mu^*)^T
$$
<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605220705405.png" alt="image-20200605220705405" style="zoom:50%;" />

#### Probability Distribution

这里我们所使用的概率分布是

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605221617855.png" alt="image-20200605221617855" style="zoom:50%;" />

下面开始公式推导，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605221733838.png" alt="image-20200605221733838" style="zoom:50%;" />

得出了我们的Sigmoid函数，$\sigma(z)$的函数图像为s型，值域范围为[0,1]，将z化简，并代入$P(C_i)=\frac{N_i}{N_1+N_2}$ (i=1,2)，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605221911457.png" alt="image-20200605221911457" style="zoom:50%;" />

将$P(x|C_1),P(x|C_2)$的表达式代入$ln\frac{P(x|C_1)}{P(x|C_2)}$，

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605222200879.png" alt="image-20200605222200879" style="zoom:50%;" />

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605223207505.png" alt="image-20200605223207505" style="zoom:50%;" />

再进行进一步化简，带入$\sum^1=\sum^2=\sum$，我们可以得出z的简易表达式$z=w\cdot x + b$，可得出$P(C_1|x)=\sigma(z)=\sigma(w\cdot x+b)$。当得出$N_1,N_2,\mu^1,\mu^2,\sum$时，就可以计算出w和b的值。

<img src="https://gitee.com/scarleatt/image/raw/master/img/image-20200605223410530.png" alt="image-20200605223410530" style="zoom:50%;" />

