# Classification and Regression
本节涵盖分类和回归算法。它还包括讨论特定类别算法的部分，例如线性方法，树和集成方法。

## **Classfication**
### Logistic Regression
Logistic Regression(逻辑回归)是一个流行的分类问题预测方法。它是[Generalized Linear models](https://en.wikipedia.org/wiki/Generalized_linear_model)(广义线性模型)的一个特殊应用以预测结果概率。在spark.ml逻辑回归中，可以使用二分类逻辑回归来预测二分类问题，或者可以使用多分类逻辑回归来预测多类别分类问题。使用family 参数来选择这两个算法，或者不设置，Spark会推断出正确的变量。

1. 将family参数设置为“multinomial”时，多分类逻辑回归可以用于二分类问题。它会产生两套coeficients(w)和两个inercepts(b)。
2. 当在具无拦截的连续非零列的数据集上训练LogisticRegressionModel时，Spark MLlib输出连续非零列零系数。这种行为与R glmnet相同，但与LIBSVM不同。
#### Binomial Logistic Regression
有关二项逻辑回归实现的更多背景和更多细节，请参阅中的[logistic regression spark.mllib文档](https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression)。

**Examples**

下面的例子显示了如何用弹性网络正则化的二元分类问题训练二项和多项逻辑回归模型。elasticNetParam参数对应于α(学习率)，regParam参数对应于λ(正则化参数)。有关参数的更多细节可以在[Python API文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression)中找到。
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LogisticRegressionWithElasticNetExample").getOrCreate()
# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))
spark.stop()
```
output:
```
Coefficients: (692,[244,263,272,300,301,328,350,351,378,379,405,406,407,428,433,434,455,456,461,462,483,484,489,490,496,511,512,517,539,540,568],[-7.35398352419e-05,-9.10273850559e-05,-0.000194674305469,-0.000203006424735,-3.14761833149e-05,-6.84297760266e-05,1.58836268982e-05,1.40234970914e-05,0.00035432047525,0.000114432728982,0.000100167123837,0.00060141093038,0.000284024817912,-0.000115410847365,0.000385996886313,0.000635019557424,-0.000115064123846,-0.00015271865865,0.000280493380899,0.000607011747119,-0.000200845966325,-0.000142107557929,0.000273901034116,0.00027730456245,-9.83802702727e-05,-0.000380852244352,-0.000253151980086,0.000277477147708,-0.000244361976392,-0.00153947446876,-0.000230733284113])
Intercept: 0.22456315961250325
Multinomial coefficients: 2 X 692 CSRMatrix
(0,244) 0.0
(0,263) 0.0001
(0,272) 0.0001
(0,300) 0.0001
(0,350) -0.0
(0,351) -0.0
(0,378) -0.0
(0,379) -0.0
(0,405) -0.0
(0,406) -0.0006
(0,407) -0.0001
(0,428) 0.0001
(0,433) -0.0
(0,434) -0.0007
(0,455) 0.0001
(0,456) 0.0001
..
..
Multinomial intercepts: [-0.120658794459,0.120658794459]
```
Find full example code at "examples/src/main/python/ml/logistic_regression_with_elastic_net.py" in the Spark repo.

spark.ml逻辑回归工具还支持在训练集上提取模型的总结。请注意，在 BinaryLogisticRegressionSummary中存储为DataFrame的预测结果和指标被注释@transient(临时的)，因此仅适用于驱动程序。

[LogisticRegressionTrainingSummary](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionSummary) 为 [LogisticRegressionModel](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel)提供了一个summary。目前只支持二分类问题。将来会增加对多分类问题模型summary的支持。

继续上面的例子
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("LogisticRegressionSummary") \
    .getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# $example on$
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
lr.setThreshold(bestThreshold)
# $example off$

spark.stop()
```
output:
```objectiveHistory:
0.6833149135741672
0.6662875751473734
0.6217068546034618
0.6127265245887887
0.6060347986802873
0.6031750687571562
0.5969621534836274
0.5940743031983118
0.5906089243339022
0.5894724576491042
0.5882187775729587
+---+--------------------+
|FPR|                 TPR|
+---+--------------------+
|0.0|                 0.0|
|0.0|0.017543859649122806|
|0.0| 0.03508771929824561|
|0.0| 0.05263157894736842|
|0.0| 0.07017543859649122|
|0.0| 0.08771929824561403|
|0.0| 0.10526315789473684|
|0.0| 0.12280701754385964|
|0.0| 0.14035087719298245|
|0.0| 0.15789473684210525|
|0.0| 0.17543859649122806|
|0.0| 0.19298245614035087|
|0.0| 0.21052631578947367|
|0.0| 0.22807017543859648|
|0.0| 0.24561403508771928|
|0.0|  0.2631578947368421|
|0.0|  0.2807017543859649|
|0.0|  0.2982456140350877|
|0.0|  0.3157894736842105|
|0.0|  0.3333333333333333|
+---+--------------------+
only showing top 20 rows

areaUnderROC: 1.0
```
Find full example code at "examples/src/main/python/ml/logistic_regression_summary_example.py" in the Spark repo.
#### Multinomial Logistic Regression
多分类通过多项逻辑（softmax）回归来支持。在多项逻辑回归中，算法产生K sets的系数集合(类似机器学习中的W)或维度K × J的矩阵其中K是结果分类数量和J是特征的数量。如果算法拟合时使用了偏置(类似机器学习中的b)，则偏置b也是一个K长度的向量。
1. 多项逻辑回归的系数(coefficients)：coefficientMatrix，偏置(intercepts):interceptVector。
2. coefficients和intercept在用多项逻辑回归训练模型中不适用。请使用coefficientMatrix，interceptVector

结果的条件概率使用的是softmax function建模，我们使用多分类响应模型将加权负对数似然最小化，并使用elastic-net penalty来控制过拟合。

![Multinomial Logistic Regression](https://github.com/cgDeepLearn/LearnSpark/blob/master/pics/logisticregressionsoftmax.png?raw=true)
关于推导的细节请查阅[这里](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model).

下面的例子展示了如何训练具有弹性网络正则化的多类逻辑回归模型。
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MultinomialLogisticRegression").getOrCreate()
# Load training data
training = spark \
    .read \
    .format("libsvm") \
    .load("data/mllib/sample_multiclass_classification_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))
spark.stop()
```
output:
```
Coefficients: 
3 X 4 CSRMatrix
(0,3) 0.3176
(1,2) -0.7804
(1,3) -0.377
Intercept: [0.0516523165983,-0.123912249909,0.0722599333102]

```
Find full example code at "examples/src/main/python/ml/multiclass_logistic_regression_with_elastic_net.py" in the Spark repo.

## Decision Tree Classifier
### Random Forest Classifier
### Gradient-Boosted Tree Classifier
### Multilayer Perception Classifier
### Linear Support Vector Machine
### One-vs-Rest Classifier(a.k.a One-vs-All)
### Naive Bayes

## **Regression**

## **Linear Methods**

## **Decision Trees**

## **Tree Ensembles**

##