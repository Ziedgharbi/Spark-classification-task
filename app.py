# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:10:38 2023


@author: Zied
"""
import findspark
findspark.init()
import pandas as pd
import requests
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType, StringType

import pyspark.ml
dir(pyspark.ml)



spark=SparkSession.builder.appName("Spark").getOrCreate()

path="C:/Users/Zied/Nextcloud/Formation/Python/GITHUB/Spark classification task/"

#path="./"
input_path=path+'input_data/'
def dowload_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv'
    r = requests.get(url, allow_redirects=True)
    

    with open(input_path+"data.csv",'wb') as f:
        f.write(r.content)
   
    #print ("file donwloaded")
    
    return()

def read_csv():
    
    #convert ALB, ALP,ALT,CHOL,PROT
    newdf=[StructField('_c0', DoubleType(), True),
           StructField('Category', StringType(), True),
           StructField('Age', DoubleType(), True),
           StructField('Sex', StringType(), True),
           StructField('ALB', DoubleType(), True),
           StructField('ALP', DoubleType(), True),
           StructField('ALT', DoubleType(), True),
           StructField('AST', DoubleType(), True),
           StructField('BIL', DoubleType(), True),
           StructField('CHE', DoubleType(), True),
           StructField('CHOL', DoubleType(), True),
           StructField('CREA', DoubleType(), True),
           StructField('GGT', DoubleType(), True),
           StructField('PROT', DoubleType(), True),
            ]
    
    finalStructure=StructType(fields=newdf)
    data=spark.read.csv(input_path+"data.csv",header=True, schema=finalStructure)
    return (data)

if __name__=='__main__':
    dowload_data()
    df=read_csv()
    
    df=df.drop('_c0')
    df.columns
    df.show()
    
    df.printSchema()
    df.dtypes
    
    
    print(df.describe().show())
    
    df.groupby("Category").count().show()
    
    # see if there are NA 
    # Find Count of Null, None, NaN of All DataFrame Columns
    from pyspark.sql.functions import col,isnan, when, count
    
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    # Find count for empty, None, Null, Nan with string literals.
    from pyspark.sql.functions import col,isnan,when,count
    df2 = df.select([count(when(col(c).contains('None') | \
                                col(c).contains('NULL') | \
                               (col(c) == '' ) | \
                                col(c).isNull() | \
                                isnan(c), c )).alias(c)
                    for c in df.drop("feature").columns])
    df2.show()   ### there are null because we forced the schema in column of type NA, so it is transformed for null
    
    df=df.dropna()
    
    
    from pyspark.ml.feature import VectorAssembler, StringIndexer 
    
    #unique value of Sex feature
    df.select("Sex").distinct().show()
    
    #convert string feature to numerical feature :label encoding
    SexEncoder=StringIndexer(inputCol='Sex', outputCol="Gender").fit(df)
    df=SexEncoder.transform(df)
    SexEncoder.labels
    
    df.show(5)
    
    #Encoding Categoy feature
    df.select("Category").distinct().show()
    categoryEncoding=StringIndexer(inputCol='Category', outputCol='Category_enc').fit(df)
    df=categoryEncoding.transform(df)
    
    categoryEncoding.labels
    df.show(5)
    
    
    #get label after encoding : inverse of StringIndxer
    from pyspark.ml.feature import IndexToString
    
    convert=IndexToString(inputCol='Category_enc', outputCol='Category_original')
    df=convert.transform(df)
    df.show(5)
    
    
    
    # defining vector assembler 
    feature=['Age','ALB', 'ALP', 'ALT','AST', 'BIL', 'CHE', 'CHOL', 'CREA','GGT','PROT','Gender']
    
    feature_vec=VectorAssembler(inputCols=feature, outputCol="feature")
    
    df=feature_vec.transform(df)
    df.show(5)
    df.count()
    
    
    ## Split dataset for train and test set
    
    train_data, test_data=df.randomSplit([0.75,0.25])

    train_data.show(5)
    train_data.count()

    test_data.show(5)
    test_data.count()
    
    
    
    

    
 
    

    
    
    
    






