# Extracting, transforming and selecting features

本节介绍用于处理特征的算法，大致分为以下几组：

- **Extraction提取**：从“原始”数据中提取特征
- **Transformation转换**：缩放，转换或修改特征
- **Selection选择**：从一大组特征集中选择一个子集
- **Locality Sensitive Hashing局部敏感散列（LSH）**：这类算法将特征变换的各个方面与其他算法相结合。

## **Feature Extractors**

### **TF-IDF**
**词频-逆向文档频率（TF-IDF）** 是一种在文本挖掘中广泛使用的特征矢量化方法，用于反映在预料中词语在文档中的重要性。*t*表示一个词，*d*表示文档和*D*表示语料。词频*TF(t,d)*是词语*t*在文档*d*中出现的次数，而文档频率*DF(t, D)*是语料中文档包含词语*t*的的数量。如果我们只用词频来衡量重要性，那么很容易过分强调出现频率很高但是却只承载文档极少的信息的词语，例如“a”，“the”和“of”。如果一个词语经常在整个语料库中出现，这意味着对于特定的文档它并没有承载特殊的信息。**逆文档频率是一个词语提供多少信息的数字度量**： 
```math
IDF(t,D) = log(|D|+1)/(DF(t, D)+1)
```
其中|D|是语料的文档总数。由于使用对数，所以如果一个词语在所有文档中出现，则其IDF值为0.请注意，smoothing term用来避免除以零对于那些在语料外的词语。TF-IDF度量是TF和IDF的乘积：
```math
TFIDF(t,d,D)=TF(t,d)⋅IDF(t,D)
```
词频和文档频率的定义有几个变体。在MLlib中，我们将TF和IDF分开，使它们更灵活。

**TF**：HashingTF与CountVectorizer都可用于生成词频向量。

HashingTF是一个Transformer，其选取词语的集合并将这些集合转换成固定长度的特征向量。在文本处理中，“set of term”可能是一堆文字。 HashingTF利用哈希技巧([hashing trick](http://en.wikipedia.org/wiki/Feature_hashing))。通过应用散列函数(hash function)将原始特征映射成一个索引（term）。这里使用的哈希函数是[MurmurHash 3](https://en.wikipedia.org/wiki/MurmurHash) 。然后根据映射后的索引计算词频。这种方法避免了计算全局term-to-index的映射，而这些计算对于大型语料库来说可能是耗费的，但是它具有潜在的散列冲突，其中不同的原始特征可能在散列之后变成相同的term。为了减少碰撞的几率，我们可以增加目标特征维数，即散列表的桶数(buckets)。**由于使用简单的模来将散列函数转换为列索引，所以建议使用2的幂作为特征维度，否则特征将不会均匀地映射到列**。默认的特征维度是2^18=262144。一个可选的二进制切换参数控制词频计数。当设置为真时，所有非零频率计数都被设置为1.这对于模拟二进制计数而不是整数计数的离散概率模型特别有用。

CountVectorizer将文本文档转换为词条计数的向量。有关更多详细信息，请参阅[CountVectorizer](https://spark.apache.org/docs/latest/ml-features.html#countvectorizer) 。

**IDF**：IDF是一个Estimator,被应用于一个数据集并产生一个IDFModel。IDFModel选取特征向量（通常从HashingTF或CountVectorizer创建）并缩放每一列。直观地，它减少了频繁出现在语料库中的列的权重。

**Note**： spark.ml不提供用于文本分割的工具。我们推荐用户参考[Stanford NLP Grou](http://nlp.stanford.edu/) 和 [scalanlp/chalk](https://github.com/scalanlp/chalk)。

**examples**

在下面的代码段中，我们从一组句子开始。我们使用Tokenizer每个句子分成单词。对于每个句子（词包），我们使用HashingTF把语句散列成一个特征向量。我们用IDF来重新调整特征向量; 使用文本作为特征时，这通常会提高性能。我们的特征向量可以传递给学习算法。

有关API的更多详细信息，请参阅[HashingTF Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.HashingTF)和[IDF Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.IDF)。
```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TF_IDfExample").getOrCreate()
sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("label", "features").show()
spark.stop()
```

|label|            features|
|-----|--------------------|
|  0.0|(20,[0,5,9,17],[0...|
|  0.0|(20,[2,7,9,13,15]...|
|  1.0|(20,[4,6,13,15,18...|



Find full example code at "examples/src/main/python/ml/tf_idf_example.py" in the Spark repo.

### **Word2Vec**

Word2Vec是一个Estimator,它选取表征文件的单词序列(句子)来训练一个Word2VecModel。模型将每个单词映射到一个唯一的固定大小的向量vector。Word2VecModel 用所有单词在文档中的平均值将每个文档转换为一个向量vector; 然后这个vector可以用作预测，文档相似度计算等功能。请参阅[Word2Vec MLlib用户指南](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec)了解更多详细信息。

**Examples**

在下面的代码段中，我们从一组文档开始，每个文档都被表示为一个单词序列。对于每个文档，我们把它转换成一个特征向量。这个特征向量可以传递给一个学习算法。

有关API的更多详细信息，请参阅[Word2Vec Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Word2Vec)。
```python
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Word2VecExample").getOrCreate()
# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
spark.stop()
```
Text: [Hi, I, heard, about, Spark] => 
Vector: [0.0334007266909,0.00213784053922,-0.00239131785929]

Text: [I, wish, Java, could, use, case, classes] => 
Vector: [0.0464252099129,0.0357359477452,-0.000244158453175]

Text: [Logistic, regression, models, are, neat] => 
Vector: [-0.00983053222299,0.0668786892667,-0.0307074898912]

---
Find full example code at "examples/src/main/python/ml/word2vec_example.py" in the Spark repo.

### **CountVectorizer**

CountVectorizer和CountVectorizerModel旨在帮助将文本文档集合转换为标记计数向量。当先验词典不可用时，CountVectorizer可以用作Estimator提取词汇表，并生成一个CountVectorizerModel。该模型生成词汇上的文档的稀疏表示，然后将其传递给其他算法（如LDA）。

在拟合过程中，CountVectorizer将选择整个语料库中词频排在前面vocabSize个的词汇。一个可选参数minDF也会影响拟合过程，方法是指定词汇必须出现的文档的最小数量（或小于1.0）。另一个可选的二进制切换参数控制输出向量。如果设置为true，则所有非零计数都设置为1.这对于模拟二进制计数而不是整数计数的离散概率模型特别有用。

**Examples**

假设我们有列如下数据帧，有id和texts列：
```python
 id | texts
----|----------
 0  | Array("a", "b", "c")
 1  | Array("a", "b", "b", "c", "a")
```
texts列中的每一行都是一个Array [String]类型的文档。调用CountVectorizer产生CountVectorizerModel与词汇（a，b，c）。然后转换后的输出列“向量”包含：
```python
 id | texts                           | vector
----|---------------------------------|---------------
 0  | Array("a", "b", "c")            | (3,[0,1,2],[1.0,1.0,1.0])
 1  | Array("a", "b", "b", "c", "a")  | (3,[0,1,2],[2.0,2.0,1.0])
```
每个向量表示文档在词汇表上的标记计数。

有关API的更多详细信息，请参阅[CountVectorizer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizer) 和[CountVectorizerModel Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizerModel)。
```python
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CountVectorizerExample").getOrCreate()
# Input data: Each row is a bag of words with a ID.
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)
spark.stop()
```
output:
```python
+---+---------------+-------------------------+
|id |words          |features                 |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+
```
Find full example code at "examples/src/main/python/ml/count_vectorizer_example.py" in the Spark repo.


## **Feature Transformers**

## **Feature Selectors**

## **Locality Sensitive Hashing**

