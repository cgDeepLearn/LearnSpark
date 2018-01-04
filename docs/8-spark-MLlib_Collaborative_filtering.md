# Collaborative Filtering
[协同过滤](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering)常被用于推荐系统。这类技术目标在于填充“用户－商品”联系矩阵中的缺失项。Spark.ml目前支持基于模型的协同过滤，其中用户和商品以少量的潜在因子来描述，用以预测缺失项。Spark.ml使用[交替最小二乘（ALS）](http://dl.acm.org/citation.cfm?id=1608614)算法来学习这些潜在因子。

spark.ml有以下参数：
- numBlocks是为了并行化计算而将用户和项目分割成的块的数量（默认为10）。
- rank是模型中潜在因素的数量（默认为10）。
- maxIter是要运行的最大迭代次数（默认为10）。
- regParam指定ALS中的正则化参数（默认为1.0）。
- implicitPrefs指定是使用显式反馈ALS变体还是用于隐式反馈数据的变体 （默认false使用显式反馈）。
- alpha是适用于ALS的隐式反馈变体的参数，其支配偏好观察值的 基线置信度（默认为1.0）。
- 非负指定是否对最小二乘使用非负约束（默认为false）。

注意：用于ALS的基于DataFrame的API目前仅支持整数类型的用户和项目ID。用户和项目ID列支持其他数字类型，但ID必须在整数值范围内。

## **Explicit vs Implict feedfack(显示与隐式反馈)**
基于矩阵分解的协同过滤的标准方法中，“用户－商品”矩阵中的条目是用户给予商品的显式偏好，例如，用户给电影评级。然而在现实世界中使用时，我们常常只能访问隐式反馈（如意见、点击、购买、喜欢以及分享等），在spark.ml中我们使用“隐式反馈数据集的协同过滤“来处理这类数据。本质上来说它不是直接对评分矩阵进行建模，而是将数据当作数值来看待，这些数值代表用户行为的观察值（如点击次数，用户观看一部电影的持续时间）。这些数值被用来衡量用户偏好观察值的置信水平，而不是显式地给商品一个评分。然后，模型用来寻找可以用来预测用户对商品预期偏好的潜在因子。
## **Scaling of the regularization parameter(正则化参数缩放)**
我们调整正则化参数regParam来解决用户在更新用户因子时产生新评分或者商品更新商品因子时收到的新评分带来的最小二乘问题。这个方法叫做[“ALS-WR”](http://dx.doi.org/10.1007/978-3-540-68880-8_32),它降低regParam对数据集规模的依赖，所以我们可以将从部分子集中学习到的最佳参数应用到整个数据集中时获得同样的性能。

## **Cold-start strategy(冷启动策略)**
在使用ALSModel进行预测时，通常会遇到测试数据集中用户和/或物品在训练模型期间不存在的情况。这通常发生在两种情况下：

1. 在生产中，对于没有评分历史记录且尚未训练的新用户或物品（这是“冷启动问题”）。
2. 在交叉验证过程中，数据分为训练集和评估集。当Spark的CrossValidator或者TrainValidationSplit中的使用简单随机拆分，实际上在评估集中普遍遇到用户或物品不存在的问题，而在训练集中并未出现这样的问题

默认情况下，Spark NaN在当用户和/或物品因素不存在于模型中时，Spark在ALSModel.transform时使用NAN作为预测。这在生产系统中可能是有用的，因为它表示一个新的用户或物品，所以系统可以做出一个决定，作为预测。

然而，这在交叉验证期间是不好的，因为任何NaN预测值都将导致NaN评估度量的结果（例如在使用RegressionEvaluator时）。这使得模型无法作出选择。

Spark允许用户将coldStartStrategy参数设置为“drop”，以便删除DataFrame包含NaN值的预测中的任何行。评估指标然后在非NaN数据上计算，并且这是有效的。下面的例子说明了这个参数的用法。

注意：目前支持的冷启动策略是“nan”（上面提到的默认行为）和“drop”。未来可能会支持进一步的策略。

**Examples**

在以下示例中，我们将从[MovieLens数据集](http://grouplens.org/datasets/movielens/)中加载评分数据 ，每行由用户，电影，评分和时间戳组成。然后，我们训练一个ALS模型，默认情况下，这个模型的评级是明确的（implicitPrefs是false）。我们通过测量评级预测的均方根误差来评估推荐模型。
```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession

spark = SparkSession.builder.appName("CollaborativeFilteringExample").getOrCreate()
lines = spark.read.text("data/mllib/als/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs.show()
# userRecs.filter(userRecs['userId'] == 1).select('recommendations').show(truncate=False)  # 看看给userId==1的用户推荐了哪10部电影
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.show()
spark.stop()
```
output:
```
Root-mean-square error = 1.742790392299329
+------+--------------------+
|userId|     recommendations|
+------+--------------------+
|    28|[[92,5.0226665], ...|
|    26|[[81,5.6422243], ...|
|    27|[[18,4.069487], [...|
|    12|[[19,6.6280622], ...|
|    22|[[74,5.141776], [...|
|     1|[[46,4.550467], [...|
|    13|[[93,3.4347346], ...|
|     6|[[25,5.163864], [...|
|    16|[[54,4.865331], [...|
|     3|[[75,5.5034533], ...|
|    20|[[22,4.563996], [...|
|     5|[[46,6.402665], [...|
|    19|[[94,4.0123057], ...|
|    15|[[46,4.932741], [...|
|    17|[[46,5.196739], [...|
|     9|[[65,4.703967], [...|
|     4|[[85,4.958973], [...|
|     8|[[43,5.747457], [...|
|    23|[[32,5.279368], [...|
|     7|[[62,5.059422], [...|
+------+--------------------+
only showing top 20 rows

+-------+--------------------+
|movieId|     recommendations|
+-------+--------------------+
|     31|[[12,3.5030043], ...|
|     85|[[14,5.6425133], ...|
|     65|[[23,4.9570875], ...|
|     53|[[14,5.271897], [...|
|     78|[[12,1.4262005], ...|
|     34|[[2,3.9721959], [...|
|     81|[[26,5.6422243], ...|
|     28|[[18,5.0155253], ...|
|     76|[[14,4.9423637], ...|
|     26|[[5,4.06113], [15...|
|     27|[[11,5.220525], [...|
|     44|[[18,3.830072], [...|
|     12|[[28,4.8217144], ...|
|     91|[[12,3.090134], [...|
|     22|[[18,8.003841], [...|
|     93|[[2,4.621838], [2...|
|     47|[[6,4.48774], [25...|
|      1|[[27,3.527709], [...|
|     52|[[8,5.0824013], [...|
|     13|[[23,4.004786], [...|
+-------+--------------------+
only showing top 20 rows
```
Find full example code at "examples/src/main/python/ml/als_example.py" in the Spark repo.

如果评分矩阵是从另一个信息源（即它是从其他信号推断）得出，可以设置implicitPrefs以true获得更好的效果：
```python
als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating")
```

**更多相关信息请查阅[spark 协同过滤](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)**