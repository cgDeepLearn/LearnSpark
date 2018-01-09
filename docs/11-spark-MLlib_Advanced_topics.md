# Advanced Topics
三种线性方法的优化方法

## Limited-memory BFGS (L-BFGS)有限记忆BFGS

 L-BFGS是拟牛顿方法家族里的一个优化算法，解决min w∈R d f(w)形式的优化问题。L-BFGS方法以二次方程来逼近目标函数来构造Hessian矩阵，不考虑目标函数的二阶偏导数。Hessian矩阵由先前的迭代评估逼近，所以不像直接使用牛顿方法一样可垂直扩展（训练特征的数目）。所以L-BFGS通常比其他一阶优化方法能更快收敛。

[象限有限记忆拟牛顿(OWL-QN)](http://research-srv.microsoft.com/en-us/um/people/jfgao/paper/icml07scalable.pdf)算法是L-BFGS的扩展，它可以有效处理L1和弹性网格正则化。L-BFGS在Spark MLlib中用于线性回归、逻辑回归、AFT生存回归和多层感知器的求解。

## Normal equation solver for weighted least square用于加权最小二乘法的正态方程求解器
MLlib 通过[WeightedLeastSquares](https://github.com/apache/spark/blob/v2.2.1/mllib/src/main/scala/org/apache/spark/ml/optim/WeightedLeastSquares.scala)实现了[加权最小二乘法](https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares)的方程求解器。

Spark MLlib目前支持正态方程的两种求解器：Cholesky分解法和拟牛顿法（L-BFGS / OWL-QN）。乔列斯基因式分解依赖于正定的协方差矩阵（即数据矩阵的列必须是线性无关的），并且如果违反这种条件将会失败。即使协方差矩阵不是正定的，准牛顿方法仍然能够提供合理的解，所以在这种情况下，正规方程求解器也可以退回到拟牛顿法。对于LinearRegression和GeneralizedLinearRegression估计，这种回退目前总是启用的。

WeightedLeastSquares支持L1，L2和弹性网络正则化，并提供启用或禁用正则化和标准化的选项。在没有L1正则化的情况下（即α = 0），存在解析解，可以使用乔列斯基（Cholesky）或拟牛顿（Quasi-Newton）求解器。当α > 0时 不存在解析解，而是使用拟牛顿求解器迭代地求出系数。

为了使正态方程有效，WeightedLeastSquares要求特征数不超过4096个。对于较大的问题，使用L-BFGS代替。

## Iteratively reweighted least squares (IRLS)迭代重新加权最小二乘
MLlib 通过[IterativelyReweightedLeastSquares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)实现[迭代重新加权最小二乘（IRLS）](https://github.com/apache/spark/blob/v2.2.1/mllib/src/main/scala/org/apache/spark/ml/optim/IterativelyReweightedLeastSquares.scala)。它可以用来找到广义线性模型（GLM）的最大似然估计，在鲁棒回归和其他优化问题中找到M估计。有关更多信息，请参阅[迭代重新加权的最小二乘法以获得最大似然估计，以及一些鲁棒性和抗性替代方法](http://www.jstor.org/stable/2345503)。

它通过以下过程迭代地解决某些优化问题：

- 线性化目前的解决方案的目标，并更新相应的权重。
- 通过WeightedLeastSquares解决加权最小二乘（WLS）问题。
- 重复上述步骤直到收敛。

由于它涉及到WeightedLeastSquares每次迭代求解加权最小二乘（WLS）问题，因此它还要求特征数不超过4096个。目前IRLS被用作GeneralizedLinearRegression的默认求解器。

**更多详细信息请查阅[Spark ml-advanced](https://spark.apache.org/docs/latest/ml-advanced.html)**