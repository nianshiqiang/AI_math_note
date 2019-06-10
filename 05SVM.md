# SVM

## 一：机器学习中的两类问题

### 1.1 回归问题

* 回归预测的是连续值

  ![5-1.png](https://i.loli.net/2019/05/20/5ce1f738333b016227.png)

* 线性回归问题的结果是一条直线（二维）、一个平面（三维）或超平面（高于三维）。
### 1.2 分类问题

* 分类问题预测的是离散值

  ![5-2.png](https://i.loli.net/2019/05/20/5ce1f8311aa0c58711.png)

## 二：SVM建模

### 2.1 线性分类器和最优线性分类器

* 样本空间中，切分超平面
  $$
  \mathbf{w^Tx}+b=0
  $$

  * 其中$\mathbf{w}$是平面的法向量，b为偏移量。

* ![5-3.png](https://i.loli.net/2019/05/20/5ce1fa04cf0c537814.png)

从上图中可以看出，我们可以通过调节w和b，得到无数个可以用于切分的超平面，那么哪一个超平面是最优的呢？接下来的问题便是，什么样的超平面是最优的？我们可以这样来理解，鲁棒性（robust）越好的平面越优。直白来说，就是当有新的样本进入时，这个超平面也能够进行正确的切分。按照这个思路来讲，那么当这个超平面处于两类样本中间时是最好的。因为偏向任何一类样本，都更加容易导致该类新样本分类错误。

### 2.2 点到超平面的距离

* 超平面$\{\mathbf{x}|\mathbf{w^Tx}+b=0\}$
* 空间中任意一点p到该点的距离

$$
r = \frac{|\mathbf{w^Tp}+b|}{||\mathbf{w}||_2}
$$

推导：

![5-4.jpg](https://i.loli.net/2019/05/20/5ce2050f5314285176.jpg)

### 2.3 SVM目标函数

* 几何间隔：(找到到超平面距离最小的那个点，然后计算出这个距离)
  $$
  M=\min _{i} r_{i}=\min _{i} \frac{\left|\mathbf{w}^{T} \mathbf{x}_{i}+b\right|}{\|\mathbf{w}\|_{2}}
  $$

* SVM目标函数
  $$
  \begin{array}{c}{\max _{\mathbf{w}, b} M} \\ {\max _{\mathbf{w}, b}\left\{\min _{i} \frac{\left|\mathbf{w}^{T} \mathbf{x}_{i}+b\right|}{\|\mathbf{w}\|_{2}}\right\}}\end{array}
  $$

  * 目标是使得最小的距离最大化。每给定一个w,b，就可以计算出所有样本到超平面的距离，这时候就可以找到一个最小距离，然后就可以通过穷举的方法优化w和b，让最小的距离最大化。
  * ![5-5.png](https://i.loli.net/2019/05/20/5ce21235f224b72239.png)

### 2.4 简化目标函数

* 分子部分$\mathbf{w^Tx_i}+b$可以改写为：
  $$
  y_i(\mathbf{w^Tx_i}+b)
  $$

  * $y_i$是人为定义的$y_i \in \{1,-1\}$
  * 如果$\mathbf{w^Tx_i}+b>0$,则$y_i=1$;
  * 如果$\mathbf{w^Tx_i}+b<0$,则$y_i=-1$;

- 目标函数由${\max _{\mathbf{w}, b}\left\{\min _{i} \frac{\left|\mathbf{w}^{T} \mathbf{x}_{i}+b\right|}{\|\mathbf{w}\|_{2}}\right\}}$变为：
  $$
  {\max _{\mathbf{w}, b}\left\{\min _{i} \frac{ y_i(\mathbf{w}^{T} \mathbf{x}_{i}+b)}{\|\mathbf{w}\|_{2}}\right\}}
  $$
  

* 上述目标函数中，大括号里边是对i进行遍历，与w无关，因此目标函数可以写成：

$$
\max _{\mathbf{w}, b}\left\{\frac{1}{\|\mathbf{w}\|_{2}} \min _{i} y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i}+b\right)\right\}
$$

* 进行scaling，$\mathbf{w} \rightarrow k\mathbf{w}$和$b \rightarrow kb$，将这两个参数分别缩放后，我们可以知道点到平面的距离不变。（分子分母同时提出一个k)

* 因此，我们可以设计距离最近的一个点，使得$y_i(\mathbf{w}^{T} \mathbf{x}_{i}+b)=1$

* 同时有
  $$
  y_i(\mathbf{w}^{T} \mathbf{x}_{i}+b) \ge1 \quad i=1,2,3...N
  $$

* 因此目标函数变为

* $$
  \max _{\mathbf{w}, b}\left\{\frac{1}{\|\mathbf{w}\|_{2}} *1\right\} \rightarrow \max _{\mathbf{w}, b}\left\{\frac{1}{\|\mathbf{w}\|_{2}}\right\} \rightarrow \min _{\mathbf{w}, b}\{{\frac{1}{2}|\mathbf{w}\|_{2}^{2}}\}\\s.t. \quad y_i(\mathbf{w}^{T} \mathbf{x}_{i}+b) \ge1 \quad i=1,2,3...N
  $$

* 约束条件的解释：对于距离最近的点，取等号；其余点均取大于号

* ![5-6.png](https://i.loli.net/2019/05/20/5ce21b1c3d7bf88856.png)


## 三：SVM求解

 