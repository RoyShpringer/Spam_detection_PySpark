#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:48:01 2021

@author: roy
"""

import findspark

findspark.init('/home/roy/spark-3.1.2-bin-hadoop3.2')

import pyspark

from pyspark.sql import SparkSession 
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF
from pyspark.sql import functions as f
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from sklearn import neighbors
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
import random

#%%


spark= SparkSession.builder.appName('spam').getOrCreate()

df= spark.read.options(header= False, inferschema= True, sep='\t').csv(r'/home/roy/spark_python/SMSSpamCollection')

df_p =df.toPandas()

df= df.withColumnRenamed('_c0','label')
df= df.withColumnRenamed('_c1','info')

df.groupBy('label').agg(f.count('label')).show()


df= df.replace({'ham':'0','spam':'1' } ,subset='label')

df= df.withColumn('label', df.label.cast('int'))

df= df.withColumn('info', f.lower(df.info))

df= df.withColumn('is_free', df.info.contains('free'))

df= df.withColumn('is_free', f.when(df.is_free==True, 1).otherwise(0))

from pyspark.sql.types import IntegerType

lengh_sentence= udf( lambda x: len(x), IntegerType() )

df= df.withColumn('lengh', lengh_sentence(df['info']))


test= df.toPandas()


#%%  building the pipeline:

tokenizer= RegexTokenizer(inputCol='info',outputCol='tokens'
                          ,pattern='\\W')

remover= StopWordsRemover(inputCol='tokens',outputCol='filt_tokens')

cv= CountVectorizer(inputCol='filt_tokens',outputCol='cvfeatures', minDF= 20.0)

idf= IDF(inputCol='cvfeatures', outputCol= 'features_idf')

#indexer= StringIndexer(inputCol='is_free',outputCol='is_free_indexed')

assembler= VectorAssembler(inputCols= ['features_idf','is_free', 'lengh'] ,outputCol='features')


rfc = RandomForestClassifier(featuresCol="features", maxDepth=12,
                              labelCol="label", numTrees=150  ,seed=123
                              )

pipe= Pipeline(stages= [tokenizer,remover, cv, idf, assembler, rfc])


#%%

df_train, df_test= df.randomSplit([0.7,0.3])


model_fitted = pipe.fit(df_train)


results= model_fitted.transform(df_test)

eval1= BinaryClassificationEvaluator()

eval2= MulticlassClassificationEvaluator(labelCol='label',metricName='recallByLabel',
                                         metricLabel=1.0)
                                         

eval1.setLabelCol("label")

print('ROC= ' , eval1.evaluate(results) )  #ROC
print('recall= ', eval2.evaluate(results) )  #Recall

#%%ol\
           
# cross validation :
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


paramgrid= ParamGridBuilder(). \
                    addGrid(rfc.maxDepth, [8,12,14]).build()
            
          
cv= CrossValidator(estimator= pipe
                   ,estimatorParamMaps = paramgrid
                   ,evaluator= eval2,
                   numFolds=3)



fit_cv_model = cv.fit(df_train)
list(zip(fit_cv_model.avgMetrics, paramgrid))       

results_cv= fit_cv_model.transform(df_test)    

eval_cv= MulticlassClassificationEvaluator(labelCol='label',metricName='recallByLabel',
                                         metricLabel=1.0)
print('recall= ', eval_cv.evaluate(results_cv) )  #Recall 0.82

        