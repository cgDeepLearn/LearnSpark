# 机器学习裤(MLlib)指南
MLlib是Spark的机器学习库，可让实际的机器学习容易和可扩展，它提供了如下工具：
- **ML算法**：通用学习算法，如分类，回归，聚类和协同过滤

- **特征提取**，特征提取，转换，降维和选择

- **管道**：用于构建，评估和调整ML管道的工具

- **持久性**：保存和加载算法，模型和管道

- **实用程序**：线性代数，统计，数据处理等

## 公告：基于DataFrame的API是主要的API
MLlib基于RDD的API现在处于维护模式。

从Spark 2.0开始，包中的基于RDD的API spark.mllib已进入维护模式。Spark的主要机器学习API现在是包中的基于DataFrame的API spark.ml。

有什么影响？

- MLlib将仍然支持基于RDD的API spark.mllib并修复错误。
- MLlib不会将新功能添加到基于RDD的API。
- 在Spark 2.x版本中，MLlib将为基于DataFrame的API添加功能，以便与基于RDD的API达成功能奇偶校验。
达到功能奇偶校验（大致估计为Spark 2.3）后，基于RDD的API将被弃用。
- 基于RDD的API预计将在Spark 3.0中被删除。

为什么MLlib切换到基于DataFrame的API？

- DataFrames提供比RDD更友好的API。DataFrame的许多优点包括Spark数据源，SQL / DataFrame查询，Tungsten和Catalyst优化以及跨语言的统一API。
- MLlib的基于DataFrame的API提供跨ML算法和跨多种语言的统一API。
- 数据框便于实际的ML管线，特别是功能转换。有关详细信息，请参阅管道指南。

什么是“Spark ML”？

- “Spark ML”不是一个正式的名字，偶尔用于指代基于MLlib DataFrame的API。这主要是由于org.apache.spark.ml基于DataFrame的API所使用的Scala包名以及我们最初用来强调管道概念的“Spark ML Pipelines”术语。

MLlib是否被弃用？

- 编号MLlib包括基于RDD的API和基于DataFrame的API。基于RDD的API现在处于维护模式。但是这两个API都没过时，MLlib也是。

## 依赖
MLlib使用线性代数包Breeze，它依赖于 netlib-java进行优化的数值处理。如果本机库1在运行时不可用，您将看到一条警告消息，而将使用纯JVM实现。

由于运行时专有二进制文件的授权问题，netlib-java默认情况下，我们不包含本地代理。要配置netlib-java/ Breeze以使用系统优化的二进制文件，请包括 com.github.fommil.netlib:all:1.1.2（或者构建Spark -Pnetlib-lgpl）作为项目的依赖项，并阅读netlib-java文档以获取平台的其他安装说明。

要在Python中使用MLlib，您将需要NumPy 1.4或更高版本。

## 2.2中的亮点
下面的列表突出了在2.2 Spark发行版中添加到MLlib中的一些新功能和增强功能：

- ALS为所有用户或项目提供top-k建议的方法，与mllib （SPARK-19535）中的功能相匹配。性能也得到了改善两者ml和mllib （SPARK-11968和 SPARK-20587）
- Correlation和 ChiSquareTest统计功能DataFrames （SPARK-19636和 SPARK-19635）
- FPGrowth频繁模式挖掘算法（SPARK-14503）
- GLM现在支持Tweedie全家（SPARK-18929）
- Imputer特征变换器来估算数据集中的缺失值（SPARK-13​​568）
- LinearSVC 对于线性支持向量机分类（SPARK-14709）
- 逻辑回归现在支持训练期间系数的限制（SPARK-20047）
## 迁移指南
MLlib正在积极开发中。未来发行版中标记为Experimental/ 的API DeveloperApi可能会更改，下面的迁移指南将解释发行版之间的所有更改。


# Basic Statistics
## Correllation(相关性)
计算两组数据之间的相关性是统计学中的一个常见操作。在spark.ml 我们提供的灵活性来计算多个系列之间的成对相关性。支持的相关方法目前是**皮尔逊**和**斯皮尔曼**相关性。

Correlation 使用指定的方法计算输入矢量数据集的相关矩阵。输出将是一个DataFrame，它包含向量列的相关矩阵。
```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()
data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))
spark.stop()
```
*Find full example code at "examples/src/main/python/ml/correlation_example.py" in the Spark repo.*
## Hypothesis testing(假设检验)
假设检验是统计学中一个强大的工具，用来确定一个结果是否具有统计显著性，这个结果是否偶然发生。spark.ml目前支持皮尔逊的卡方（χ2χ2）测试独立性。

ChiSquareTest针对标签的每个特征进行皮尔森独立测试。对于每个特征，（特征，标签）对被转换为计算卡方统计量的可能性矩阵。所有标签和特征值必须是分类的。
```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])

r = ChiSquareTest.test(df, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))
```
*Find full example code at "examples/src/main/python/ml/chi_square_test_example.py" in the Spark repo.*


# Pipelines