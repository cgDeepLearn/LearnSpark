# ML Tuning: model selection and hyperparameter tuning
本节介绍如何使用MLlib的工具来调整ML算法和管道。内置的交叉验证和其他工具允许用户优化算法和管道中的超参数。

## **Model selection(又叫 hyperparameter tuning)**
ML中的一个重要任务是*Model Selection*(选择模型)，或者使用数据为给定任务找到最佳模型或参数。这也被称为*Tuning*(调整)。调整可以是以针对三个Estimators算子如LogisticRegression进行调整，也可以对整个Pipeline进行调整。用户可以一次对Pipeline整体进行调整，而不是对Pipeline的每个元素单独进行调整。

MLlib支持使用[CrossValidator](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.CrossValidator)和[TrainValidationSplit](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.TrainValidationSplit)的工具进行模型选择。这些工具需要下列项目：

- Estimator: 需要调整的算法或Pipeline
- Set of ParamMaps: 可供选择的参数，有时称为“parameter grid”来搜索
- Evaluator: 度量标准：衡量一个拟合Model在测试数据上的表现

在较高层面上，这些模型选择工具的工作如下：

- 他们将输入数据分成单独的训练和测试数据集。
- 对每组训练数据与测试数据对，对参数表集合，用相应参数来拟合估计器，得到训练后的模型，再使用评估器来评估模型表现。
- 选择最好的一组参数生成的模型。

其中，对于回归问题评估器可选择RegressionEvaluator，二值数据可选择BinaryClassificationEvaluator，多分类问题可选择MulticlassClassificationEvaluator。评估器里默认的评估准则可通过setMetricName方法重写。

用户可通过[ParamGridBuilder](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.ParamGridBuilder)构建参数网格。

## **Cross-Validation**
CrossValidator将数据集划分为若干子集分别地进行训练和测试。如当k＝3时，CrossValidator产生3个训练数据与测试数据对，每个数据对使用2/3的数据来训练，1/3的数据来测试。对于一组特定的参数表，CrossValidator计算基于三组不同训练数据与测试数据对训练得到的模型的评估准则的平均值。确定最佳参数表后，CrossValidator最后使用最佳参数表基于全部数据来重新拟合Estimator。

示例：

注意对参数网格进行交叉验证的成本是很高的。如下面例子中，参数网格hashingTF.numFeatures有3个值，lr.regParam有2个值，CrossValidator使用2折交叉验证。这样就会产生(3*2)*2=12中不同的模型需要进行训练。在实际的设置中，通常有更多的参数需要设置，且我们可能会使用更多的交叉验证折数（3折或者10折都是经使用的）。所以CrossValidator的成本是很高的，尽管如此，比起启发式的手工验证，交叉验证仍然是目前存在的参数选择方法中非常有用的一种。

**Examples**

有关API的更多详细信息，请参阅[CrossValidatorPython](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)文档。
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CrossValidatorExample").getOrCreate()
# Prepare training documents, which are labeled.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 0.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 0.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 0.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
# This will allow us to jointly choose parameters for all Pipeline stages.
# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
# this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)

# Prepare test documents, which are unlabeled.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "mapreduce spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    print(row)
selected.show()
spark.stop()
```
output:
```
Row(id=4, text='spark i j k', probability=DenseVector([0.627, 0.373]), prediction=0.0)
Row(id=5, text='l m n', probability=DenseVector([0.3451, 0.6549]), prediction=1.0)
Row(id=6, text='mapreduce spark', probability=DenseVector([0.3351, 0.6649]), prediction=1.0)
Row(id=7, text='apache hadoop', probability=DenseVector([0.2767, 0.7233]), prediction=1.0)
+---+---------------+--------------------+----------+
| id|           text|         probability|prediction|
+---+---------------+--------------------+----------+
|  4|    spark i j k|[0.62703425702535...|       0.0|
|  5|          l m n|[0.34509123755317...|       1.0|
|  6|mapreduce spark|[0.33514123783842...|       1.0|
|  7|  apache hadoop|[0.27672019766802...|       1.0|
+---+---------------+--------------------+----------+
```
Find full example code at "examples/src/main/python/ml/cross_validator.py" in the Spark repo.
## **Train-Validation Split**
除了交叉验证以外，Spark还提供 TrainValidationSplit 用以进行超参数调整。和交叉验证评估K次不同， TrainValidationSplit 只对每组参数评估一次。因此它计算代价更低，但当训练数据集不是足够大时，其结果可靠性不高。

与交叉验证不同， TrainValidationSplit仅需要一个训练数据与验证数据对。使用训练比率参数将原始数据划分为两个部分。如当训练比率为0.75时，训练验证分裂使用75%数据以训练，25%数据以验证。

与交叉验证相同，确定最佳参数表后，训练验证分裂最后使用最佳参数表基于全部数据来重新拟合Estimator。

**Examples**

有关API的更多详细信息，请参阅[TrainValidationSplitPython](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.TrainValidationSplit)文档。
```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TrainValidationSplitExample").getOrCreate()
# Prepare training and test data.
data = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")
train, test = data.randomSplit([0.9, 0.1], seed=12345)

lr = LinearRegression(maxIter=10)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
model.transform(test)\
    .select("features", "label", "prediction")\
    .show()
spark.stop()
```
output:
```
+--------------------+--------------------+--------------------+
|            features|               label|          prediction|
+--------------------+--------------------+--------------------+
|(10,[0,1,2,3,4,5,...|  -23.51088409032297| -1.6659388625179559|
|(10,[0,1,2,3,4,5,...| -21.432387764165806|  0.3400877302576284|
|(10,[0,1,2,3,4,5,...| -12.977848725392104|-0.02335359093652395|
|(10,[0,1,2,3,4,5,...| -11.827072996392571|  2.5642684021108417|
|(10,[0,1,2,3,4,5,...| -10.945919657782932| -0.1631314487734783|
|(10,[0,1,2,3,4,5,...|  -10.58331129986813|   2.517790654691453|
|(10,[0,1,2,3,4,5,...| -10.288657252388708| -0.9443474180536754|
|(10,[0,1,2,3,4,5,...|  -8.822357870425154|  0.6872889429113783|
|(10,[0,1,2,3,4,5,...|  -8.772667465932606|  -1.485408580416465|
|(10,[0,1,2,3,4,5,...|  -8.605713514762092|   1.110272909026478|
|(10,[0,1,2,3,4,5,...|  -6.544633229269576|  3.0454559778611285|
|(10,[0,1,2,3,4,5,...|  -5.055293333055445|  0.6441174575094268|
|(10,[0,1,2,3,4,5,...|  -5.039628433467326|  0.9572366607107066|
|(10,[0,1,2,3,4,5,...|  -4.937258492902948|  0.2292114538379546|
|(10,[0,1,2,3,4,5,...|  -3.741044592262687|   3.343205816009816|
|(10,[0,1,2,3,4,5,...|  -3.731112242951253| -2.6826413698701064|
|(10,[0,1,2,3,4,5,...|  -2.109441044710089| -2.1930034039595445|
|(10,[0,1,2,3,4,5,...| -1.8722161156986976| 0.49547270330052423|
|(10,[0,1,2,3,4,5,...| -1.1009750789589774| -0.9441633113006601|
|(10,[0,1,2,3,4,5,...|-0.48115211266405217| -0.6756196573079968|
+--------------------+--------------------+--------------------+
```
Find full example code at "examples/src/main/python/ml/train_validation_split.py" in the Spark repo.

**更多相关信息请查阅[Spark ml-tuning](https://spark.apache.org/docs/latest/ml-tuning.html)**