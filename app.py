# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:10:38 2023


@author: Zied
"""
import pandas as pd
import requests
import pyspark
from pyspark.sql import SparkSession

import pyspark.ml
dir(pyspark.ml)



spark=SparkSession.builder.appName("Spark").getOrCreate()

path="C:/Users/Zied/Nextcloud/Formation/Python/GITHUB/Spark classification task/"

path="./input_data/"
input_path=path+'input_data/'
def dowload_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv'
    r = requests.get(url, allow_redirects=True)
    

    with open(path+"data.csv",'wb') as f:
        f.write(r.content)
   
    #print ("file donwloaded")
    
    return()

def read_csv():
    data=data=spark.read.csv(path+"data.csv",header=True,inferSchema=True)
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
    

    
    
    
    






