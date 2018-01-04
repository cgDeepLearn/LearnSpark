# Frequent Pattern Mining
频繁项目，项目集，子序列或其他子结构的挖掘通常是分析大规模数据集的第一步，这已经成为数据挖掘领域的一个活跃的研究课题。我们将用户引用到Wikipedia的[关联规则学习](http://en.wikipedia.org/wiki/Association_rule_learning)中以获取更多信息。
## FP-Growth
FP-growth算法在Han等人的文章[“ Mining frequent patterns without candidate generation”](http://dx.doi.org/10.1145/335191.335372)中描述，其中“FP”代表频繁模式。给定交易数据集，FP增长的第一步是计算项目频率并识别频繁项目。与为同样目的而设计的[Apriori-like](http://en.wikipedia.org/wiki/Apriori_algorithm)算法不同，FP-growth的第二步使用后缀树（FP-tree）结构来编码事务，而不显式生成候选集合，这通常是昂贵的。第二步之后，可以从FP-tree中提取频繁项目集。在这里spark.mllib，我们实现了FP-growth的并行版本，称为PFP(详细请查看Li等人：[PFP:Parallel FP-growth for query recommendation](http://dx.doi.org/10.1145/1454008.1454027))。PFP根据事务的后缀分配增长的FP-树的工作，因此比单机实现更具可扩展性。

spark.ml FP的增长实现需要以下（超）参数：

minSupport：一个项目组的最小支持被确定为频繁的。例如，如果一个项目在5个交易中出现3个，则它具有3/5 = 0.6的支持。

minConfidence：生成关联规则的最低置信度。信心是一个关联规则被发现是真实的指标。例如，如果交易项目集X出现4次，X 并且Y只出现2次，则规则的置信度为X => Y2/4 = 0.5。该参数不会影响对频繁项目集的挖掘，但指定从频繁项集生成关联规则的最小置信度。

numPartitions：用于分配工作的分区数量。默认情况下，param未设置，并使用输入数据集的分区数量。

FPGrowthModel规定：

freqItemsets：DataFrame格式的频繁项目集（“items”[Array]，“freq”[Long]）

associationRules：minConfidence以DataFrame（“antecedent”[Array]，“consequent”[Array]，“confidence”[Double]）格式在上面生成的关联规则。

transform：对于每个交易itemsCol，transform方法将比较其项目与每个关联规则的前提。如果记录包含特定关联规则的所有前提条件，则该规则将被视为适用，并将其结果添加到预测结果中。变换法将所有适用规则的后果总结为预测。预测列具有相同的数据类型，itemsCol并且不包含中的现有项目itemsCol。

**Examples**
```python
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FrequentPatternMiningExample").getOrCreate()
df = spark.createDataFrame([
    (0, [1, 2, 5]),
    (1, [1, 2, 3, 5]),
    (2, [1, 2])
], ["id", "items"])

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()
spark.stop()
```
output:
```
+---------+----+
|    items|freq|
+---------+----+
|      [5]|   2|
|   [5, 1]|   2|
|[5, 1, 2]|   2|
|   [5, 2]|   2|
|      [2]|   3|
|      [1]|   3|
|   [1, 2]|   3|
+---------+----+

+----------+----------+------------------+
|antecedent|consequent|        confidence|
+----------+----------+------------------+
|       [5]|       [1]|               1.0|
|       [5]|       [2]|               1.0|
|    [1, 2]|       [5]|0.6666666666666666|
|    [5, 2]|       [1]|               1.0|
|    [5, 1]|       [2]|               1.0|
|       [2]|       [5]|0.6666666666666666|
|       [2]|       [1]|               1.0|
|       [1]|       [5]|0.6666666666666666|
|       [1]|       [2]|               1.0|
+----------+----------+------------------+

+---+------------+----------+
| id|       items|prediction|
+---+------------+----------+
|  0|   [1, 2, 5]|        []|
|  1|[1, 2, 3, 5]|        []|
|  2|      [1, 2]|       [5]|
+---+------------+----------+
```
Find full example code at "examples/src/main/python/ml/fpgrowth_example.py" in the Spark repo.


**更多相关信息请查阅[Spark FPGrowth](https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html)**