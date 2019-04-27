from sklearn.ensemble import RandomForestRegressor
import math
import certifi
import numpy
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import sys
import elasticsearch
from elasticsearch import Elasticsearch
import json
import os
import logging
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

def connect_to_remote_elasticsearch():

  es = elasticsearch.Elasticsearch(['https://'+ os.environ['HOSTNAME'] +':' + os.environ['PORT']+ '/'], http_auth=(os.environ['USER'], os.environ['PASSWORD']),use_ssl=True, ca_certs=certifi.where())
  if es.ping():
    return es
  else:
    raise "Unable to connect to Elastic Search"

  return es


def predict_next_N(metrics, metric_key, N, target_index, es):
  train = metrics[metric_key]
  training_data = []
  label = []
  row =[]
  window = 10
  for i in range(0,len(train)):
    if ((i+1)% window == 0):
      label.append(train[i])
      training_data.append(row)
      row = []
    else:
      row.append(train[i])


  training_data = training_data[:-1]
  label = label[0:len(training_data)]
  X = numpy.array(training_data)
  Y = numpy.array(label)

  tree = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=5, min_samples_split=5, min_samples_leaf=2)
  tree.fit(X,Y)
  
## take the last N values
  queue = train[-window+1:]
  answer = []
  N = int(N)
  for i in range(0,N):
    Xtest = np.array([queue])
    prediction = tree.predict(Xtest)
    scalar = np.asscalar(prediction)
    answer.append(scalar)
    queue = queue[1:]
    queue.append(scalar)

  return answer
if __name__ == '__main__':
  #es = connect_elasticsearch()
  es = connect_to_remote_elasticsearch()
  # populateIndexes(es, 'metrics', '') -- used for populating index with simulated values
  logging.basicConfig(level=logging.ERROR)

  index_name = os.environ['index'] # ## name of the index from which the metric documents are read
  ## we issue a query to index to retrieve documents from the index sorted by this sortkey (ex: @timeStamp)
  sort_key = '@timestamp' # documents returned from elastic search will be sorted by this key 
  
## Starting and end time stamps for training data for training anomaly detector 
  training_start_timestamp = os.environ['train_start_time']
  training_end_timestamp = os.environ['train_end_time']

  next_N_predictions = os.environ['NEXT_N_PREDICT']
 
  target_index = os.environ['NEXT_N_PREDICT_INDEX']
  if es is not None:
    ## This search query retrieves all documents between start_time and end_timestamp for training data
    ## the retrieved results are sorted by timestamp
    search_query = '{"sort" : [{"@timestamp" : {"order" : "asc"}}],"from" : 0, "size" : 10000, "query" : {"range" : { "@timestamp" : { "gte": "' + training_start_timestamp + '", "lte": "'+ training_end_timestamp + '" } }}}'
    response= es.search(index=str(index_name),body=search_query)
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
      print ("predictions for the key: " + metric_key)
      print (len(metrics[metric_key]))
      if (len(metrics[metric_key]) != len(sort_key_list)):
        raise "time stamps collected != metric sample list"
      result = predict_next_N(metrics, metric_key, next_N_predictions, target_index, es)
      counter = 1
      for r in result:
        targetJson = '{ "metric": "' + metric_key + '", "prediction_index": ' + str(counter) + ', "value":' + str(r) + '}'
        es.index(index=target_index, doc_type="prediction", body=json.loads(targetJson))
        counter+=1



