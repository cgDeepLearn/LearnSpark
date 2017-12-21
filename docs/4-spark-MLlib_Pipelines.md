# ML Pipelines

## **管道中的主要概念**
MLlib对机器学习算法的API进行了标准化，使得将多种算法合并成一个流水线或工作流变得更加容易。本部分涵盖了Pipelines API引入的关键概念，其中流水线概念主要受scikit-learn项目的启发。

- **DataFrame**：这个ML API使用Spark SQL中的DataFrame作为一个ML数据集，它可以容纳各种数据类型。例如，一个DataFrame可以具有存储文本，特征向量，真实标签和预测的不同列。

- **Transformer**：一个Transformer是可以将一个DataFrame变换成成另一个DataFrame的算法。例如，一个ML模型是一个Transformer将一个DataFrame特征转化为一个DataFrame预测的模型。

- **Estimator**：一个 Estimator是一个可以被应用在DataFrame上来产生一个Transformer的算法。例如，一个学习算法是一种Estimator，它可以在DataFrame上训练并生成模型。

- **Pipeline**：Pipeline将多个Transformers和Estimators连接起来以指定ML工作流程。

- **Parameter**：所有Transformers和Estimators现在对于指定参数共享通用API。

### **DataFrame(数据帧)**
机器学习可以应用于各种数据类型，如向量，文本，图像和结构化数据。这个API采用DataFrameSpark SQL来支持各种数据类型。

DataFrame支持许多基本和结构化的类型; 请参阅Spark SQL数据类型参考以获取受支持类型的列表。除了Spark SQL指南中列出的类型以外，DataFrame还可以使用ML Vector类型。

A DataFrame可以隐式地或显式地从常规创建RDD。有关示例，请参阅下面的代码示例和Spark SQL编程指南。

a DataFrame中的列被命名。下面的代码示例使用“text”，“feature”和“label”等名称。

### **Pipeline components(管道组件)**
**Transformers**

A Transformer是包含特征变换器和学习模型的抽象。从技术上来说，a Transformer实现了一种方法(*transform()*),将一个DataFrame转换为另一个的方法，通常通过附加一列或多列。例如：
- 特征转换器选取一个DataFrame，读取列（例如文本），将其映射到新的列（例如特征向量），并且输出具有附加映射列的新DataFrame
- 学习模型可以选取一个DataFrame，读取包含特征向量的列，预测每个特征向量的标签，并输出带有预测标签的新列的DataFrame。

**Estimators**

一个Estimator是在数据集上训练的学习算法的抽象概念。从技术上讲，一个Estimator实现了一个方法 *fit()*，它接受DataFrame并生成一个 Model，这是一个Transformer。例如，一个学习算法，如LogisticRegression(它是一个Estimator)，调用 *fit()* 函数来训练一个LogisticRegressionModel模型，它是一个Model也是一个Transformer。

**Properties o pipeline components**

Transformer.transform()s和Estimator.fit()s都是无状态的。将来，有状态算法可以通过替代概念来支持。
每个Transformer或者Estimator的实例具有唯一的ID，这在指定参数（在下面讨论）中是有用的。

**Pipeline**
在机器学习中，通常运行一系列算法来处理和学习数据。例如，简单的文本文档处理工作流程可能包括几个阶段：
- 将每个文档的文本分词。
- 将每个文档的单词转换为数字特征向量。
- 使用特征向量和标签来学习预测模型。

MLlib表示这样一个工作流程Pipeline，它由一系列 PipelineStages（Transformers和Estimators）组成，并以特定顺序运行。我们将使用这个简单的工作流程作为本节中的一个运行示例。

**How it works**

A Pipeline是一个阶段序列，每个阶段是一个Transformer或一个Estimator。这些阶段是按顺序运行的，输入的DataFrame在每个阶段都经过转换。对于Transformer阶段，*transform()* 方法被调用作用于DataFrame上。对于Estimator阶段，*fit()*方法被调用，以产生Transformer（它成为PipelineModel或合适的Pipeline的一部分），以及Transformer的transform()方法也被调用作用于DataFrame。

我们用简单的文本文档工作流来说明这一点。下图是a 。训练时间的使用情况Pipeline。
![Pipeline](https://github.com/cgDeepLearn/LearnSpark/blob/master/pics/ml-Pipeline.png?raw=true)

在上面，最上面一行代表一个Pipeline有三个阶段。前两个（Tokenizer和HashingTF）是Transformers（蓝色），第三个（LogisticRegression）是Estimator（红色）。最下面一行代表流经管道的数据，其中圆柱表示DataFrames。这个Pipeline.fit()方法在原始DataFrame文档和标签上被调用。Tokenizer.transform()方法将原始文本文档分词，分词后的words作为一个新列添加到DataFrame中。HashingTF.transform()方法将单词列转换为特征向量，并向这些向量作为一个新列添加到DataFrame中。现在，既然LogisticRegression是一个Estimator，Pipeline首先调用LogisticRegression.fit()方法就生成一个LogisticRegressionModel。如果Pipeline有更多Estimators，它就会在DataFrame传送到下个阶段之前调用LogisticRegressionModel的transform() 方法。

一个Pipeline是一个Estimator。因此，在Pipeline的fit()方法运行后，它产生一个PipelineModel，这是一个 Transformer。这PipelineModel是在测试时使用 ; 下图说明了这种用法。

![PiplineModel](https://github.com/cgDeepLearn/LearnSpark/blob/master/pics/ml-PipelineModel.png?raw=true)

在上面的图中，PipelineModel具有和原始的Pipeline相同数量的阶段，但所有EstimatorS在原始Pipeline中已变成TransformerS。当PipelineModel的transform()方法在测试数据集被调用，数据在管道上按序传递。每个阶段的transform()方法都会更新数据集并将其传递到下一个阶段。

Pipelines和PipelineModels有助于确保训练和测试数据经过相同的特征处理步骤。

### **Details**

*DAG Pipelines*：A Pipeline的阶段被指定为一个有序数组。这里给出的例子都是线性Pipeline的，即Pipeline每个阶段使用前一阶段产生的数据。Pipeline只要数据流图形成有向无环图（DAG），就可以创建非线性的PipelineS。该图当前是基于每个阶段的输入和输出列名（通常指定为参数）隐含指定的。如果Pipeline形式为DAG，那么阶段必须按拓扑顺序指定。

*Runtime checking*：由于Pipelines可以在不同类型的DataFrames上运行，所以不能使用compile-time类型检查。 Pipelines和PipelineModels，而是在实际运行Pipeline之前进行runtime checking检查。这种类型的检查是通过使用DataFrame *schema*来完成的，schema是对DataFrame的列的数据类型的描述。

*Unique Pipeline stages*：A Pipeline的阶段应该是独一无二的实例。例如，同一个实例 myHashingTF不应该插入Pipeline两次，因为Pipeline阶段必须有唯一的ID。然而，不同的实例myHashingTF1和myHashingTF2（两个类型HashingTF）可以放在一起，Pipeline因为创建不同的实例使用不同的ID。

### **Parameters**

MLlib Estimators和Transformers使用统一的API来指定参数。

A Param是一个带有自包含文档的命名参数。A ParamMap是一组（参数，值）对。

将参数传递给算法有两种主要方法：

1. 为实例设置参数。例如，如果lr是的一个实例LogisticRegression，它可以调用lr.setMaxIter(10)让lr.fit()至多10次迭代使用。这个API类似于spark.mllib包中使用的API 。
2. 传递ParamMap给fit()或transform()。任何在ParamMap中额参数将覆盖以前通过setter方法指定的参数。

参数属于Estimators和Transformers的特定实例。例如，如果我们有两个LogisticRegression实例lr1和lr2，然后我们可以建立一个ParamMap与两个maxIter指定的参数：ParamMap(lr1.maxIter -> 10, lr2.maxIter -> 20)。如果一个Pipeline里有两个包含maxIter参数的算法，那么这很有用。

### **Saving and LoadingPipelines**

通常情况下，会将模型或管道保存到磁盘供以后使用。在Spark 1.6中，模型导入/导出功能被添加到管道API中。大多数基本的Transformers都和一些更加基本的ML模型一样被支持。请参阅算法的API文档以查看是否支持保存和加载。

## Code examples

本节给出了说明上述功能的代码示例。有关更多信息，请参阅API文档（Scala， Java和Python）。
### **Example: Estimator, Transformer, and Param**

这个例子涉及的概念Estimator，Transformer和Param。
请参阅[EstimatorPython文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Estimator)，[TransformerPython文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Transformer)和[ParamsPython文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.param.Params)以获取有关API的更多详细信息。
```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName("ParamsExample").getOrCreate()
# Prepare training data from a list of (label, features) tuples.
training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)
# Print out the parameters, documentation, and any default values.
print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(training)

# Since model1 is a Model (i.e., a transformer produced by an Estimator),
# we can view the parameters it used during fit().
# This prints the parameter (name: value) pairs, where names are unique IDs for this
# LogisticRegression instance.
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

# We may alternatively specify parameters using a Python dictionary as a paramMap
paramMap = {lr.maxIter: 20}
paramMap[lr.maxIter] = 30  # Specify 1 Param, overwriting the original maxIter.
paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # Specify multiple Params.

# You can combine paramMaps, which are python dictionaries.
paramMap2 = {lr.probabilityCol: "myProbability"}  # Change output column name
paramMapCombined = paramMap.copy()
paramMapCombined.update(paramMap2)

# Now learn a new model using the paramMapCombined parameters.
# paramMapCombined overrides all parameters set earlier via lr.set* methods.
model2 = lr.fit(training, paramMapCombined)
print("Model 2 was fit using parameters: ")
print(model2.extractParamMap())

# Prepare test data
test = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

# Make predictions on test data using the Transformer.transform() method.
# LogisticRegression.transform will only use the 'features' column.
# Note that model2.transform() outputs a "myProbability" column instead of the usual
# 'probability' column since we renamed the lr.probabilityCol parameter previously.
prediction = model2.transform(test)
result = prediction.select("features", "label", "myProbability", "prediction") \
    .collect()

for row in result:
    print("features=%s, label=%s -> prob=%s, prediction=%s"
          % (row.features, row.label, row.myProbability, row.prediction))

spark.stop()
```
Find full example code at "examples/src/main/python/ml/estimator_transformer_param_example.py" in the Spark repo.

### **Example: Pipeline**

本示例遵循Pipeline上图中所示的简单文本文档。
有关APi的更多详细信息，请参阅[PipelinePython文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName(""PipeLineExample"").getOrCreate()
# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

spark.stop()
```
Find full example code at "examples/src/main/python/ml/pipeline_example.py" in the Spark repo.

### **Model selection (hyperparameter tuning)- 模型选择(超参数调整)**
使用ML管道的一大好处是超参数优化。有关自动模型选择的更多信息，请参阅[ML调整指南](https://spark.apache.org/docs/latest/ml-tuning.html)。