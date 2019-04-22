import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import sys
from elasticsearch import Elasticsearch
import json
import os
import logging
## Requirements
## run elastic search at localhost:9200
## sample command
## python3.6 read_compute_post_outliers_on_es.py metrics --@timestamp net_conntrack_listener_conn_accepted_total 2 anomaly
## python version 3.6 and above
## arg 1 : name of the index on elastic search that contains documents. In this example:  metrics
## arg 2 : the field using which the documents are sorted in the response from Elastic Search. In this example : @timestamp
## arg 3 : number of test samples (the last n samples on which anomaly detection should be run) In this example it is 2
## arg 4 : name of the index in which the predictions are stored
def search(es_object, index_name, document,search):
    res = es_object.search(index=index_name,  body=search)
    return res

def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Successfully connected to Elastic Search')
    else:
        print('Could not connect to Elastic Search')
    return _es
def compute_outliers(metrics, metric_key, test_samples, sort_key_list, es):
  train = metrics[metric_key][:-test_samples]
  test = metrics[metric_key][-test_samples:]
  
  test_time_stamps = sort_key_list[-test_samples:]
  
  X_train = pd.DataFrame(train, columns = ['sample'])
  X_test = pd.DataFrame(test, columns=['sample'])
 
  clf = IsolationForest(max_samples=10000)
  clf.fit(X_train)
  outliers = clf.predict(X_test)
  
  for i in range(0,len(test_time_stamps)):
    ## Store if the metric sampled at a particular time stamp was an anomaly or not
    jsonString = '{"' + sort_key + '": "'+ test_time_stamps[i] + '", "metrics":"' + metric_key + '", "value":' + str(test[i]) + ',"outlier":' + str(outliers[i]) +'}'
    es.index(index=anomalies_index, doc_type='predicted', body=json.loads(jsonString))

if __name__ == '__main__':
  es = connect_elasticsearch()
  logging.basicConfig(level=logging.ERROR)

  index_name = sys.argv[1] ## name of the index from which the metric documents are read
  ## we issue a query to index to retrieve documents from the index sorted by this sortkey (ex: @timeStamp)
  sort_key = sys.argv[2] ## sorty key could be time stamp name of the field in which the response should be sorted
  ## metrics that we need to analyze for isolation forest
  test_samples = int(sys.argv[3]) 
  ## number of samples on which we need to test for anomaly. Rest of the samples will be used for training
  anomalies_index = sys.argv[4]
  sort_key = sort_key.replace('--','').strip()
  
  if es is not None:
 
    search_object = '{"sort" : [{"' + str(sort_key) + '" : {"order" : "asc"}}]}'
    
    response= es.search(index=str(index_name),body=search_object)
    metrics = {}
    sort_key_list = []

    ## The response contains a field called hits which in turn has another field called hits
    ## extract the hits
    allhits = response["hits"]["hits"]
    for hit in allhits: #response["hits"]["hits"]:
      ## Extract the captured metrics from _source->prometheus->metrics
      captured_metrics = hit['_source']['prometheus']['metrics']
      sort_key_value = hit['_source'][str(sort_key)] ## Time stamp in the hit
      sort_key_list.append(sort_key_value)
      for key in captured_metrics:
        if key not in metrics:
          metrics[key] = []
        metrics[key].append(captured_metrics[key])

    for metric_key in metrics:
      print ("computing outliers for " + metric_key)
      input()
      if (len(metrics[metric_key]) != len(sort_key_list)):
        raise "time stamps collected != metric sample list"


      compute_outliers(metrics, metric_key, test_samples, sort_key_list, es)


