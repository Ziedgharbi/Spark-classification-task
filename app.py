# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:10:38 2023


@author: Zied
"""
import pandas as pd
import requests
import pyspark
from pyspark.sql import SparkSession

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
    
    df=df.dropColumns
    
    df.show()
    df.describe()
    
    df.dtypes
    





