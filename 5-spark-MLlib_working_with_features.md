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
```
Find full example code at "examples/src/main/python/ml/tf_idf_example.py" in the Spark repo.

### **Word2Vec**
### **CountVectorizer**

## **Feature Transformers**

## **Feature Selectors**

## **Locality Sensitive Hashing**

