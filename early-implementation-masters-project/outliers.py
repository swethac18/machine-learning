import datetime
from datetime import timedelta
import math
import certifi
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import sys
import elasticsearch
from elasticsearch import Elasticsearch
import json
import os
import logging
## Requirements
## run elastic search at localhost:9200
## update the environment variables as mentioned in env_variables.sh
def search(es_object, index_name, document,search):
    res = es_object.search(index=index_name,  body=search)
    return res

def compute_timestamps(duration):
  duration = int(duration)
  now = datetime.datetime.now()
  test_end_iso = datetime.datetime.now().isoformat()
  now_minus_duration = now -  timedelta(hours=duration)
  test_start_iso = now_minus_duration.isoformat()

  train_end = now - timedelta(days=1)
  train_start = train_end - timedelta(hours=duration)

  train_start_iso = train_start.isoformat()
  train_end_iso = train_end.isoformat()

  return (train_start_iso, train_end_iso, test_start_iso, test_end_iso)

def populateIndexes(es, index_name, docType):
  jsonstr= '{ "@timestamp":"2019-03-01T08:05:34.853Z", "event":{ "dataset":"prometheus.collector", "duration":115000, "module":"prometheus" }, "metricset":{ "name":"collector" }, "prometheus":{ "labels":{ "listener_name":"http" }, "metrics":{ "net_conntrack_listener_conn_accepted_total":3, "net_conntrack_listener_conn_closed_total":0 } }, "service":{ "address":"127.0.0.1:55555", "type":"prometheus" } }'
  i = 0
  counter = 0
  for i in range(0,60):
    for j in range(0,24):
      for k in range(1,15):
        to_replace = "2019-03-01T08:05:34.853Z"
        kstr = str(k)
        if (k < 10):
          kstr = "0" + kstr#+ str(k)
        istr = str(i)
        if (i < 10):
          istr = "0" + istr
        jstr = str(j)
        if (j < 10):
          jstr = "0" + jstr

        replace_with = "2019-03-"+kstr+"T" + jstr+ ":"+ istr+":34.853Z"
        targetTimeStamp = jsonstr.replace(to_replace, replace_with)
        to_replace_metric = '"net_conntrack_listener_conn_accepted_total":3'
        conn = math.sin(counter)*10 + 20
        counter+=1
        if (conn < 0):
          conn *= -1
        conn = int(conn)
        replace_metric_with = '"net_conntrack_listener_conn_accepted_total":' + str(conn)
        targetJson = targetTimeStamp.replace(to_replace_metric, replace_metric_with)
        es.index(index=index_name, doc_type="metric", body=json.loads(targetJson))

def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Successfully connected to Elastic Search')
    else:
        print('Could not connect to Elastic Search')
    return _es

def compute_outliers(metrics, test_metrics,  metric_key, sort_key_list, es):
  train = metrics[metric_key]# [:-test_samples]
  test = test_metrics[metric_key]#[-test_samples:]
  test_time_stamps = sort_key_list
  
  X_train = pd.DataFrame(train, columns = ['sample'])
  X_test = pd.DataFrame(test, columns=['sample'])
 
  clf = IsolationForest(max_samples=1000)
  clf.fit(X_train)
  outliers = clf.predict(X_test)
  
  for i in range(0,len(test_time_stamps)):
    ## Store if the metric sampled at a particular time stamp was an anomaly or not
    jsonString = '{"' + sort_key + '": "'+ test_time_stamps[i] + '", "metrics":"' + metric_key + '", "value":' + str(test[i]) + ',"outlier":' + str(outliers[i]) +'}'
    es.index(index=anomalies_index, doc_type='predicted', body=json.loads(jsonString))

def connect_to_remote_elasticsearch():

  es = elasticsearch.Elasticsearch(['https://'+ os.environ['HOSTNAME'] +':' + os.environ['PORT']+ '/'], http_auth=(os.environ['USER'], os.environ['PASSWORD']),use_ssl=True, ca_certs=certifi.where())
  if es.ping():
    return es
  else:
    raise "Unable to connect to Elastic Search"

  return es

def compute_outliers_allmetrics(training_hits, testing_hits, es):

  training_metrics = {}
  testing_metrics = {}
  for hit in training_hits:
    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in training_metrics):
        training_metrics[key] = []
      training_metrics[key].append(captured_metrics[key])

  for hit in testing_hits:
    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in testing_metrics):
        testing_metrics[key] = []
      testing_metrics[key].append(captured_metrics[key])

  for key in training_metrics:
    X_train = pd.DataFrame(training_metrics[key], columns=['sample'])
    clf = IsolationForest(max_samples=1000)
    
    X_test = pd.DataFrame(testing_metrics[key], columns=['sample'])
    clf.fit(X_train)
    outliers = clf.predict(X_test)

    i = 0
    input("model prediction computed")
    for hit in testing_hits:
      input(outliers[i])
      input(hit)
      es.update(index=hit['_index'],doc_type=hit['_type'],id=hit['_id'],body='{"doc":{"outlier":{"' + key +'":' + str(outliers[i]) + '}}}') 
      i+=1

def main_fn():
  
  #es = connect_elasticsearch()
  es = connect_to_remote_elasticsearch()
  # populateIndexes(es, 'metrics', '') -- used for populating index with simulated values
  logging.basicConfig(level=logging.ERROR)

  index_name = os.environ['index'] # ## name of the index from which the metric documents are read
  ## we issue a query to index to retrieve documents from the index sorted by this sortkey (ex: @timeStamp)
  sort_key = '@timestamp' # documents returned from elastic search will be sorted by this key 
  
## Starting and end time stamps for training data for training anomaly detector 

  (training_start_timestamp, training_end_timestamp, test_start_time_stamp, test_end_time_stamp )= compute_timestamps(os.environ['duration']) 
  training_start_timestamp = os.environ['train_start_time']
  training_end_timestamp = os.environ['train_end_time']

## Starting and end time stamps for the test data on which anaomaly detection is run
  test_start_time_stamp = os.environ['test_start_time']
  test_end_time_stamp = os.environ['test_end_time']

## Index in which the predicted results of outlier (-1) or not-outlier (1) is stored
  anomalies_index = os.environ['results_index'] #sys.argv[4]
  sort_key = sort_key.replace('--','').strip()
  
  if es is not None:
    ## This search query retrieves all documents between start_time and end_timestamp for training data
    ## the retrieved results are sorted by timestamp
    search_query = '{"sort" : [{"@timestamp" : {"order" : "asc"}}],"from" : 0, "size" : 10000, "query" : {"range" : { "@timestamp" : { "gte": "' + training_start_timestamp + '", "lte": "'+ training_end_timestamp + '" } }}}'
    response= es.search(index=str(index_name),body=search_query)
    metrics = {}
    sort_key_list = []

    ## The response contains a field called hits which in turn has another field called hits
    ## extract the hits
    training_hits = response["hits"]["hits"]
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
    
    search_query = '{"sort" : [{"@timestamp" : {"order" : "asc"}}],"from" : 0, "size" : 10000, "query" : {"range" : { "@timestamp" : { "gte": "' + test_start_time_stamp + '", "lte": "'+ test_end_time_stamp + '" } }}}'
    test_response= es.search(index=str(index_name),body=search_query)
    test_metrics = {}
    test_sort_key_list = []

    ## The response contains a field called hits which in turn has another field called hits
    ## extract the hits
    testing_hits = response["hits"]["hits"]
    compute_outliers_allmetrics(training_hits, testing_hits, es)
    input("end")
    allhits = test_response["hits"]["hits"]
    for hit in allhits: #response["hits"]["hits"]:
      ## Extract the captured metrics from _source->prometheus->metrics
      captured_metrics = hit['_source']['prometheus']['metrics']
      sort_key_value = hit['_source'][str(sort_key)] ## Time stamp in the hit
      test_sort_key_list.append(sort_key_value)
      for key in captured_metrics:
        if key not in test_metrics:
          test_metrics[key] = []
        test_metrics[key].append(captured_metrics[key])


    for metric_key in metrics:
      print ("computing outliers for " + metric_key)
      print (len(metrics[metric_key]))
      if (len(metrics[metric_key]) != len(sort_key_list)):
        raise "time stamps collected != metric sample list"


      compute_outliers(metrics, test_metrics,  metric_key, test_sort_key_list, es)

if __name__ == "__main__":
    main_fn()
