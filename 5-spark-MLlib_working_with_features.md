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

### **Tokenizer**
[Tokenization](http://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)(分词)是将文本（如句子）分解成单个词（通常是单词）的过程。一个简单的[Tokenizer](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.Tokenizer)类提供了这个功能。下面的例子展示了如何将句子拆分成单词序列。

[RegexTokenizer](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.RegexTokenizer)允许更高级的基于正则表达式（正则表达式）匹配的分词。默认情况下，使用参数“pattern”（正则表达式，默认："\\s+"）作为分隔符来分割输入文本。或者，用户可以将参数“gaps”设置为false，指示正则表达式“pattern”而不是分割间隙来表示“tokens”，并查找所有匹配事件作为分词结果。

**Examples**

有关API的更多详细信息，请参阅[Tokenizer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Tokenizer)和[RegexTokenizer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RegexTokenizer)。
```python
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TokenizerExample").getOrCreate()
sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
# alternatively, pattern="\\w+", gaps(False)

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("sentence", "words")\
    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("sentence", "words") \
    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)
spark.stop()
```
output:
```python
+-----------------------------------+------------------------------------------+------+
|sentence                           |words                                     |tokens|
+-----------------------------------+------------------------------------------+------+
|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
|Logistic,regression,models,are,neat|[logistic,regression,models,are,neat]     |1     |
+-----------------------------------+------------------------------------------+------+

+-----------------------------------+------------------------------------------+------+
|sentence                           |words                                     |tokens|
+-----------------------------------+------------------------------------------+------+
|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
|Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |5     |
+-----------------------------------+------------------------------------------+------+
```
Find full example code at "examples/src/main/python/ml/tokenizer_example.py" in the Spark repo.

### **StopWordsRemover**

[Stop words](https://en.wikipedia.org/wiki/Stop_words)(停止词)是应该从输入中排除的词，通常是因为这些词经常出现而又不具有如此多的含义。

StopWordsRemover将一串字符串（例如一个Tokenizer的输出）作为输入，并从输入序列中删除所有的停止词。停用词表由stopWords参数指定。某些语言的默认停用词可通过调用访问StopWordsRemover.loadDefaultStopWords(language)，可用的选项有“danish”, “dutch”, “english”, “finnish”, “french”, “german”, “hungarian”, “italian”, “norwegian”, “portuguese”, “russian”, “spanish”, “swedish” and “turkish”。布尔参数caseSensitive指示匹配是否区分大小写（默认为false）。

**Examples**

假设我们有列如下数据帧,拥有列id和raw：
```python
 id | raw
----|----------
 0  | [I, saw, the, red, baloon]
 1  | [Mary, had, a, little, lamb]
```
应用StopWordsRemover与raw作为输入列，filtered作为输出列，我们应该得到以下：
```python
 id | raw                         | filtered
----|-----------------------------|--------------------
 0  | [I, saw, the, red, baloon]  |  [saw, red, baloon]
 1  | [Mary, had, a, little, lamb]|[Mary, little, lamb]
```
在这里filtered，“I”，“the”，“had”和“a”这些停用词语已被滤除。\
有关API的更多详细信息，请参阅[StopWordsRemover Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StopWordsRemover)。
```python
from pyspark.ml.feature import StopWordsRemover

sentenceData = spark.createDataFrame([
    (0, ["I", "saw", "the", "red", "balloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"])
], ["id", "raw"])

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)
```
Find full example code at "examples/src/main/python/ml/stopwords_remover_example.py" in the Spark repo.

### **n-gram**

一个[n-gram](https://en.wikipedia.org/wiki/N-gram)是一个包含整数n个tokens（通常是单词）的序列。NGram类可用于输入特征转变成n-grams。

NGram将一串字符串（例如一个Tokenizer的输出）作为输入。参数n用于确定每个n-gram中的terms的数量。输出将由n-grams的序列组成，每个n-gram由空格分隔的n个连续的words的字符串表示。如果输入序列少于n，则没有输出。

**Examples**

有关API的更多细节，请参阅[NGram Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.NGram)。
```python
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("n_gramExample").getOrCreate()
wordDataFrame = spark.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["id", "words"])

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.select("ngrams").show(truncate=False)
spark.stop()
```
output:
```python
+------------------------------------------------------------------+
|ngrams                                                            |
+------------------------------------------------------------------+
|[Hi I, I heard, heard about, about Spark]                         |
|[I wish, wish Java, Java could, could use, use case, case classes]|
|[Logistic regression, regression models, models are, are neat]    |
+------------------------------------------------------------------+
```
Find full example code at "examples/src/main/python/ml/n_gram_example.py" in the Spark repo.
### **Binarizer**
Binarization(二值化)是将数字特征阈值化为二进制（0/1）特征的过程。

Binarizer需传入参数inputCol和outputCol，以及所述threshold参数来进行二值化。大于阈值的特征值被二进制化为1.0; 等于或小于阈值的值被二值化为0.0。inputCol支持Vector和Double类型。

**Examples**

有关API的更多细节，请参阅[Binarizer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Binarizer)。
```python
from pyspark.ml.feature import Binarizer
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName("BinarizerExample").getOrCreate()
continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 0.2)
], ["id", "feature"])

binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
binarizedDataFrame.show()
spark.stop()
```
output:
```python
Binarizer output with Threshold = 0.500000
+---+-------+-----------------+
| id|feature|binarized_feature|
+---+-------+-----------------+
|  0|    0.1|              0.0|
|  1|    0.8|              1.0|
|  2|    0.2|              0.0|
+---+-------+-----------------+
```
Find full example code at "examples/src/main/python/ml/binarizer_example.py" in the Spark repo.

### **PCA**
[PCA](http://en.wikipedia.org/wiki/Principal_component_analysis)是一个统计过程，它使用正交变换将一组可能相关的变量的观察值转换成一组称为主成分的线性不相关变量的值。一个PCA类使用PCA将向量映射到低维空间来训练一个模型。下面的例子显示了如何将五维特征向量投影到三维主成分中。

**Examples**

有关API的更多细节，请参阅[PCA Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.PCA)。
```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("PCA_Example").getOrCreate()
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
spark.stop()
```
output:
```python
+-----------------------------------------------------------+
|pcaFeatures                                                |
+-----------------------------------------------------------+
|[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
+-----------------------------------------------------------+
```
Find full example code at "examples/src/main/python/ml/pca_example.py" in the Spark repo.

### **PolynomialExpansion**
[Polynomial expansion](http://en.wikipedia.org/wiki/Polynomial_expansion)(多项式展开)是将特征扩展到一个多项式空间的过程，这个多项式空间是由原始维度的n-degree组合形成的。一个[PolynomialExpansion](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.PolynomialExpansion)类提供此功能。下面的例子展示了如何将特征扩展到一个三次多项式空间。

**Examples**

有关API的更多详细信息，请参阅[PolynomialExpansion Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.PolynomialExpansion)。
```python
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName("PolynormialExpansionExample").getOrCreate()
df = spark.createDataFrame([
    (Vectors.dense([2.0, 1.0]),),
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([3.0, -1.0]),)
], ["features"])

polyExpansion = PolynomialExpansion(degree=3, inputCol="features", outputCol="polyFeatures")
polyDF = polyExpansion.transform(df)

polyDF.show(truncate=False)
spark.stop()
```
output:
```python
+----------+------------------------------------------+
|features  |polyFeatures                              |
+----------+------------------------------------------+
|[2.0,1.0] |[2.0,4.0,8.0,1.0,2.0,4.0,1.0,2.0,1.0]     |
|[0.0,0.0] |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]     |
|[3.0,-1.0]|[3.0,9.0,27.0,-1.0,-3.0,-9.0,1.0,3.0,-1.0]|
+----------+------------------------------------------+
```
Find full example code at "examples/src/main/python/ml/polynomial_expansion_example.py" in the Spark repo.

### **Discrete Cosine Transform(DCT)**
[ Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)离散余弦变换将时域中的长度为N的实数序列转换为另一个频域中长度为N的实数序列。一个[DCT](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.DCT)类提供此功能，实现 [DCT-II](https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II) 和通过缩放结果1/sqrt(2)倍使得变换的表示矩阵是单一的。被应用于变换的序列是无偏移的（例如变换的序列的第0th个元素是 第0th 个DCT系数而不是N/2个）。

**Examples**

有关API的更多详细信息，请参阅[DCT Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.DCT)。
```python
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DCT_Example").getOrCreate()
df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")

dctDf = dct.transform(df)

dctDf.select("featuresDCT").show(truncate=False)
spark.stop()
```
output:
```python
+----------------------------------------------------------------+
|featuresDCT                                                     |
+----------------------------------------------------------------+
|[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
|[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
+----------------------------------------------------------------+
```
Find full example code at "examples/src/main/python/ml/dct_example.py" in the Spark repo.
### **StringIndexer**
StringIndexer将一串字符串标签编码为标签索引。这些索引范围为[0, numLabels)按照标签频率排序，因此最频繁的标签获得索引0。对于unseen的标签如果用户选择保留它们，它们将被放在索引numLabels处。如果输入列是数字，我们将其转换为字符串值并将其索引。当下游管道组件（例如Estimator或 Transformer）使用此字符串索引标签时，必须将组件的输入列设置为此字符串索引列名称。在许多情况下，您可以使用setInputCol设置输入列

```python
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("StringIndexerExample").getOrCreate()
df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()
spark.stop()
```
output:
```python
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+
```

此外，StringIndexer处理看不见的标签还有三个策略：
1. 抛出一个异常(默认的)
2. 完全跳过unseen标签的行
3. 把一个unseen的标签放在一个特殊的额外桶里，在索引numLabels处

让我们回到之前的例子：
```python
 id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | d
 4  | e
```
如果你没有设置如何StringIndexer处理看不见的标签或将其设置为“错误”，则会抛出异常。但是，如果您已经调用setHandleInvalid("skip")，则会生成以下数据集：
```python
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
```
请注意，包含“d”或“e”的行不显示。

如果你调用setHandleInvalid("keep")，将生成以下数据集：
```python
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
 3  | d        | 3.0
 4  | e        | 3.0
 # d,e 所在的被映射到索引“3.0”
```
Find full example code at "examples/src/main/python/ml/string_indexer_example.py" in the Spark repo.

### **IndexToString**
对称地StringIndexer，IndexToString将一列标签索引映射回包含作为字符串的原始标签的列。一个常见的用例是从标签生成索引StringIndexer，用这些索引对模型进行训练，并从预测索引列中检索原始标签IndexToString。但是，您可以自由提供自己的标签。

### **OneHotEncoding**

### **VectorIndexer**

### **Interaction**

### **Normalizer**

### **StandardScaler**

### **MinMaxScaler**

### **MaxAbsScaler**

### **Bucketizer**

### **ElementwiseProduct**

### **SQLTransformer**

### **VectorAssembler**

### **QuantileDiscretizer**

### **Imputer**


## **Feature Selectors**

## **Locality Sensitive Hashing**

