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
对应于StringIndexer，IndexToString将一列标签索引映射回包含作为字符串的原始标签的列。一个常见的用例是从StringIndexer标签生成索引，用这些索引对模型进行训练，并从预测IndexToString索引列中检索原始标签。然而，你也可以提供自己的标签。

**Examples**

构造tringIndexer例子，假设我们有一个如下的数据帧，其有id和categoryIndex列：
```python
 id | categoryIndex
----|---------------
 0  | 0.0
 1  | 2.0
 2  | 1.0
 3  | 0.0
 4  | 0.0
 5  | 1.0
```
将categoryIndex作为输入列，应用IndexToString， originalCategory作为输出列，我们能够检索我们的原始标签（他们将从列的元数据推断）：

有关API的更多详细信息，请参阅[IndexToString Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.IndexToString)。
```python
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("IndexToStringExample").getOrCreate()

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = indexer.fit(df)
indexed = model.transform(df)

print("Transformed string column '%s' to indexed column '%s'"
      % (indexer.getInputCol(), indexer.getOutputCol()))
indexed.show()

print("StringIndexer will store labels in output column metadata\n")

converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
converted = converter.transform(indexed)

print("Transformed indexed column '%s' back to original string column '%s' using "
      "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
converted.select("id", "categoryIndex", "originalCategory").show()
spark.stop()
```
output:
```
Transformed string column 'category' to indexed column 'categoryIndex'
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

StringIndexer will store labels in output column metadata

Transformed indexed column 'categoryIndex' back to original string column 'originalCategory' using labels in metadata
+---+-------------+----------------+
| id|categoryIndex|originalCategory|
+---+-------------+----------------+
|  0|          0.0|               a|
|  1|          2.0|               b|
|  2|          1.0|               c|
|  3|          0.0|               a|
|  4|          0.0|               a|
|  5|          1.0|               c|
+---+-------------+----------------+
```
Find full example code at "examples/src/main/python/ml/index_to_string_example.py" in the Spark repo.

### **OneHotEncoding**
[One-hot encoding](http://en.wikipedia.org/wiki/One-hot)将一列标签索引映射到一列二进制向量，其中最多只有一个one-value。该编码允许那些期望使用连续特征的算法（例如Logistic回归）使用分类特征。

**Examples**

关于 API的更多细节请参考[OneHotEncoder Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)。
```python
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("OneHotEncoderExample").getOrCreate()
df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()
spark.stop()
```
output:
```python
+---+--------+-------------+-------------+
| id|category|categoryIndex|  categoryVec|
+---+--------+-------------+-------------+
|  0|       a|          0.0|(2,[0],[1.0])|
|  1|       b|          2.0|    (2,[],[])|
|  2|       c|          1.0|(2,[1],[1.0])|
|  3|       a|          0.0|(2,[0],[1.0])|
|  4|       a|          0.0|(2,[0],[1.0])|
|  5|       c|          1.0|(2,[1],[1.0])|
+---+--------+-------------+-------------+
```
Find full example code at "examples/src/main/python/ml/onehot_encoder_example.py" in the Spark repo.

### **VectorIndexer**
VectorIndexer有助于索引Vectors的数据集中的分类特征。它可以自动决定哪些特征是分类的，并将原始值转换为分类索引。具体来说，它做了以下几点：

1. 取一个[Vector](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.linalg.Vector)类型的输入列和一个参数maxCategories。
2. 根据不同值的数量确定哪些特征应该分类，这些特征最多被分为maxCategories类。
3. 计算每个分类特征的分类索引(0-based)。
4. 索引分类特征并将原始特征值转换为索引。

索引分类特征允许Decision Trees(决策树)和Tree Ensembles等算法适当地处理分类特征，提高性能。

**Examples**
在下面的例子中，我们读入一个标记点​​的数据集，然后用VectorIndexer来决定哪些特征应该被视为分类特征。我们将分类特征值转换为它们的索引。这个转换的数据然后可以被传递给诸如DecisionTreeRegressor处理分类特征的算法。

请参阅[VectorIndexer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorIndexer) 以获取有关API的更多详细信息。
```python
from pyspark.ml.feature import VectorIndexer
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName("VectorIndexerExample").getOrCreate()
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
indexerModel = indexer.fit(data)

categoricalFeatures = indexerModel.categoryMaps
print("Chose %d categorical features: %s" %
      (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

# Create new column "indexed" with categorical values transformed to indices
indexedData = indexerModel.transform(data)
indexedData.show()
spark.stop()
```
output:
```python
Chose 351 categorical features: 645, 69, 365, 138, 101, 479, 333, 249, 0, 555, 666, 88, 170, 115, 276, 308, 5, 449, 120, 247, 614, 677, 202, 10, 56, 533, 142, 500, 340, 670, 174, 42, 417, 24, 37, 25, 257, 389, 52, 14, 504, 110, 587, 619, 196, 559, 638, 20, 421, 46, 93, 284, 228, 448, 57, 78, 29, 475, 164, 591, 646, 253, 106, 121, 84, 480, 147, 280, 61, 221, 396, 89, 133, 116, 1, 507, 312, 74, 307, 452, 6, 248, 60, 117, 678, 529, 85, 201, 220, 366, 534, 102, 334, 28, 38, 561, 392, 70, 424, 192, 21, 137, 165, 33, 92, 229, 252, 197, 361, 65, 97, 665, 583, 285, 224, 650, 615, 9, 53, 169, 593, 141, 610, 420, 109, 256, 225, 339, 77, 193, 669, 476, 642, 637, 590, 679, 96, 393, 647, 173, 13, 41, 503, 134, 73, 105, 2, 508, 311, 558, 674, 530, 586, 618, 166, 32, 34, 148, 45, 161, 279, 64, 689, 17, 149, 584, 562, 176, 423, 191, 22, 44, 59, 118, 281, 27, 641, 71, 391, 12, 445, 54, 313, 611, 144, 49, 335, 86, 672, 172, 113, 681, 219, 419, 81, 230, 362, 451, 76, 7, 39, 649, 98, 616, 477, 367, 535, 103, 140, 621, 91, 66, 251, 668, 198, 108, 278, 223, 394, 306, 135, 563, 226, 3, 505, 80, 167, 35, 473, 675, 589, 162, 531, 680, 255, 648, 112, 617, 194, 145, 48, 557, 690, 63, 640, 18, 282, 95, 310, 50, 67, 199, 673, 16, 585, 502, 338, 643, 31, 336, 613, 11, 72, 175, 446, 612, 143, 43, 250, 231, 450, 99, 363, 556, 87, 203, 671, 688, 104, 368, 588, 40, 304, 26, 258, 390, 55, 114, 171, 139, 418, 23, 8, 75, 119, 58, 667, 478, 536, 82, 620, 447, 36, 168, 146, 30, 51, 190, 19, 422, 564, 305, 107, 4, 136, 506, 79, 195, 474, 664, 532, 94, 283, 395, 332, 528, 644, 47, 15, 163, 200, 68, 62, 277, 691, 501, 90, 111, 254, 227, 337, 122, 83, 309, 560, 639, 676, 222, 592, 364, 100
+-----+--------------------+--------------------+
|label|            features|             indexed|
+-----+--------------------+--------------------+
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
|  1.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[151,152,153...|(692,[151,152,153...|
|  0.0|(692,[129,130,131...|(692,[129,130,131...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[99,100,101,...|(692,[99,100,101,...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[153,154,155...|(692,[153,154,155...|
|  0.0|(692,[151,152,153...|(692,[151,152,153...|
|  1.0|(692,[129,130,131...|(692,[129,130,131...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  1.0|(692,[150,151,152...|(692,[150,151,152...|
|  0.0|(692,[124,125,126...|(692,[124,125,126...|
|  0.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[97,98,99,12...|(692,[97,98,99,12...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
+-----+--------------------+--------------------+
only showing top 20 rows
```
Find full example code at "examples/src/main/python/ml/vector_indexer_example.py" in the Spark repo.
### **Interaction**
Interaction是一个Transformer,采用向量或双值列的方法生成一个单一的向量列，其中包含每个输入列的一个值的所有组合的乘积。

例如，如果您有两个向量类型列，每个列都有三个维度作为输入列，那么您将获得一个9维向量作为输出列。

**Examples**

假设我们有以下DataFrame,有列“id1”，“vec1”和“vec2”：
```python
  id1|vec1          |vec2          
  ---|--------------|--------------
  1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] 
  2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] 
  3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] 
  4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] 
  5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]
  6  |[1.0,1.0,4.0] |[2.0,8.0,4.0]   
```
应用Interaction作用于这些输入列，然后interactedCol输出列包含：
```python
  id1|vec1          |vec2          |interactedCol                                         
  ---|--------------|--------------|------------------------------------------------------
  1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] |[8.0,4.0,5.0,16.0,8.0,10.0,24.0,12.0,15.0]            
  2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] |[56.0,72.0,64.0,42.0,54.0,48.0,112.0,144.0,128.0]     
  3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] |[36.0,54.0,108.0,6.0,9.0,18.0,54.0,81.0,162.0]        
  4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] |[360.0,160.0,200.0,288.0,128.0,160.0,216.0,96.0,120.0]
  5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]|[450.0,315.0,135.0,100.0,70.0,30.0,350.0,245.0,105.0] 
  6  |[1.0,1.0,4.0] |[2.0,8.0,4.0] |[12.0,48.0,24.0,12.0,48.0,24.0,48.0,192.0,96.0] 
```
注：该方法暂时并没有python的实现，有scala和Java的
### **Normalizer**
Normalizer是一个Transformer，它转换数据集的Vector行，规范化每个Vector为unit norm。它采用参数p来规范化，它指定用于规范化的p范数。（默认p = 2 ）。这种规范化可以帮助标准化您的输入数据，并改善学习算法的行为。

**Examples**

以下示例演示如何以libsvm格式加载数据集，然后将每行标准化为unit L^1 norm1和unitL^∞ norm。

有关API的更多详细信息，请参阅[Normalizer Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Normalizer)。
```python
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName("NormalizerExample").getOrCreate()
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.5, -1.0]),),
    (1, Vectors.dense([2.0, 1.0, 1.0]),),
    (2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
l1NormData = normalizer.transform(dataFrame)
print("Normalized using L^1 norm")
l1NormData.show()

# Normalize each Vector using $L^\infty$ norm.
lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
print("Normalized using L^inf norm")
lInfNormData.show()
spark.stop()
```
output:
```
Normalized using L^1 norm
+---+--------------+------------------+
| id|      features|      normFeatures|
+---+--------------+------------------+
|  0|[1.0,0.5,-1.0]|    [0.4,0.2,-0.4]|
|  1| [2.0,1.0,1.0]|   [0.5,0.25,0.25]|
|  2|[4.0,10.0,2.0]|[0.25,0.625,0.125]|
+---+--------------+------------------+

Normalized using L^inf norm
+---+--------------+--------------+
| id|      features|  normFeatures|
+---+--------------+--------------+
|  0|[1.0,0.5,-1.0]|[1.0,0.5,-1.0]|
|  1| [2.0,1.0,1.0]| [1.0,0.5,0.5]|
|  2|[4.0,10.0,2.0]| [0.4,1.0,0.2]|
+---+--------------+--------------+
```
Find full example code at "examples/src/main/python/ml/normalizer_example.py" in the Spark repo.
### **StandardScaler**
StandardScaler转换Vector行的数据集，将每个特征归一化为具有单位标准偏差和/或零均值。它需要参数：

- withStd：默认为true。将数据缩放到单位标准偏差。
- withMean：默认为False。在缩放之前将数据集中在平均值上。它会建立一个密集的输出，所以在应用于稀疏输入时要小心。

StandardScaler是一个Estimator，可以fit在一个数据集上产生一个StandardScalerModel; 这相当于计算汇总统计。然后该模型可以转换Vector数据集中的列以具有单位标准偏差和/或零均值特征。

请注意，如果某个要素的标准偏差为零，则会在该特征的Vector中返回默认值0.0。

**Examples**

以下示例演示如何加载数据集，然后将每个特征标准化为单位标准偏差。

有关API的更多详细信息，请参阅[StandardScaler Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StandardScaler)。
```python
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("StandardScalerExample").getOrCreate()
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.5, -1.0]),),
    (1, Vectors.dense([2.0, 1.0, 1.0]),),
    (2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
scaledData.show(truncate=False)
spark.stop()
```
output:
```
+---+--------------+------------------------------------------------------------+
|id |features      |scaledFeatures                                              |
+---+--------------+------------------------------------------------------------+
|0  |[1.0,0.5,-1.0]|[0.6546536707079772,0.09352195295828244,-0.6546536707079771]|
|1  |[2.0,1.0,1.0] |[1.3093073414159544,0.1870439059165649,0.6546536707079771]  |
|2  |[4.0,10.0,2.0]|[2.618614682831909,1.870439059165649,1.3093073414159542]    |
+---+--------------+------------------------------------------------------------+
```
Find full example code at "examples/src/main/python/ml/standard_scaler_example.py" in the Spark repo.

### **MinMaxScaler**
MinMaxScaler转换Vector行数据集，将每个特征重新缩放到特定范围（通常为[0，1]）。它需要参数：

min：默认为0.0。转换后的下界，被所有特征共享。
max：默认为1.0。变换后的上界，被所有的特征共享。
MinMaxScaler计算数据集的汇总统计并生成一个MinMaxScalerModel。然后模型可以单独转换每个特征，使其在给定的范围内。

特征E的重新缩放的值被计算为，
Rescaled(ei) = (ei − Emin) / (Emax − Emin) ∗ (max − min) + min 
对于Emax==Emin的情况Rescaled(ei)=0.5∗(max+min)

请注意，由于零值可能会被转换为非零值，所以transofromer的输出将会是DenseVector，即使输入是稀疏输入。

**Examples**

有关API的更多详细信息，请参阅[MinMaxScaler Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.MinMaxScaler) 和[MinMaxScalerModel Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.MinMaxScalerModel)。
```python
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession  
spark = SparkSession.builder.appName("MinMaxScalerExample").getOrCreate()
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
], ["id", "features"])

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(dataFrame)

# rescale each feature to range [min, max].
scaledData = scalerModel.transform(dataFrame)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaledData.select("features", "scaledFeatures").show()
spark.stop()
```
output:
```
Features scaled to range: [0.000000, 1.000000]
+--------------+--------------+
|      features|scaledFeatures|
+--------------+--------------+
|[1.0,0.1,-1.0]| [0.0,0.0,0.0]|
| [2.0,1.1,1.0]| [0.5,0.1,0.5]|
|[3.0,10.1,3.0]| [1.0,1.0,1.0]|
+--------------+--------------+
```
Find full example code at "examples/src/main/python/ml/min_max_scaler_example.py" in the Spark repo.
### **MaxAbsScaler**
MaxAbsScaler转换Vector行的数据集，通过分割每个特征的最大绝对值来重新缩放每个特征到范围[-1,1]。它不会移动/居中数据，因此不会破坏任何稀疏性。

MaxAbsScaler计算数据集的汇总统计并生成一个MaxAbsScalerModel。该模型可以将每个特征分别转换为范围[-1,1]。

**Examples**

有关API的更多详细信息，请参阅[MaxAbsScaler Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.MaxAbsScaler) 和[MaxAbsScalerModel Python文档](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.MaxAbsScalerModel)。
```python
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession  

spark = SparkSession.builder.appName("MaxAbsScalerExample").getOrCreate()
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -8.0]),),
    (1, Vectors.dense([2.0, 1.0, -4.0]),),
    (2, Vectors.dense([4.0, 10.0, 8.0]),)
], ["id", "features"])

scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(dataFrame)

# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(dataFrame)

scaledData.select("features", "scaledFeatures").show()
spark.stop()
```
output:
```
+--------------+----------------+
|      features|  scaledFeatures|
+--------------+----------------+
|[1.0,0.1,-8.0]|[0.25,0.01,-1.0]|
|[2.0,1.0,-4.0]|  [0.5,0.1,-0.5]|
|[4.0,10.0,8.0]|   [1.0,1.0,1.0]|
+--------------+----------------+
```
Find full example code at "examples/src/main/python/ml/max_abs_scaler_example.py" in the Spark repo.
### **Bucketizer**
Bucketizer将一列连续的特征转换成特征桶列，其中桶由用户指定。它需要一个参数：

splits：用于将连续要素映射到存储桶的参数。有n + 1分裂，有n个桶。由分割x，y定义的存储区保存除了最后一个存储区（也包括y）之外的范围[x，y]中的值。分割应严格增加。必须明确提供-inf，inf的值以涵盖所有Double值; 否则，指定拆分之外的值将被视为错误。两个例子splits是Array(Double.NegativeInfinity, 0.0, 1.0, Double.PositiveInfinity)和Array(0.0, 1.0, 2.0)。
请注意，如果您不知道目标列的上限和下限，则应该添加Double.NegativeInfinity并Double.PositiveInfinity作为分割的界限，以防止出现Bucketizer界限异常。

还要注意，你提供的分割必须严格按照递增顺序，即s0 < s1 < s2 < ... < sn。

更多细节可以在Bucketizer的API文档中找到。
### **ElementwiseProduct**

### **SQLTransformer**

### **VectorAssembler**

### **QuantileDiscretizer**

### **Imputer**


## **Feature Selectors**

## **Locality Sensitive Hashing**

