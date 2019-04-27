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

def search(es_object, index_name, document,search):
    res = es_object.search(index=index_name,  body=search)
    return res

def compute_timestamps(duration):
  duration = int(duration)
  now = datetime.datetime.utcnow()
  test_end_iso = datetime.datetime.now().isoformat()
  now_minus_duration = now -  timedelta(minutes=duration)
  test_start_iso = now_minus_duration.isoformat()

  train_end = now - timedelta(days=1)
  train_start = train_end - timedelta(minutes=duration)

  train_start_iso = train_start.isoformat()
  train_end_iso = train_end.isoformat()

  return (train_start_iso, train_end_iso, test_start_iso, test_end_iso)
  return (str(train_start_iso)[:-3]+'Z', str(train_end_iso)[:-3]+'Z', str(test_start_iso)[:-3]+'Z', str(test_end_iso)[:-3]+'Z')

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
 
  clf = IsolationForest(max_samples=10)
  clf.fit(X_train)
  outliers = clf.predict(X_test)

  for i in range(0,len(test_time_stamps)):
    ## Store if the metric sampled at a particular time stamp was an anomaly or not
    if ( i > len(outliers)):
      print ("Array out of bounds for outliers")
      break
    jsonString = '{"' + sort_key + '": "'+ test_time_stamps[i] + '", "metrics":"' + metric_key + '", "value":' + str(test[i]) + ',"outlier":' + str(outliers[i]) +'}'
    es.index(index=anomalies_index, doc_type='predicted', body=json.loads(jsonString))

def connect_to_remote_elasticsearch():

  es = elasticsearch.Elasticsearch(['https://'+ os.environ['HOST'] +':' + os.environ['PORT']+ '/'], http_auth=(os.environ['USER'], os.environ['PASSWORD']),use_ssl=True, ca_certs=certifi.where())
  if es.ping():
    return es
  else:
    raise "Unable to connect to Elastic Search"

  return es

def compute_outliers_for_allmetrics(training_hits, testing_hits, es):
  training_metrics = {}
  for hit in training_hits:
    if ('prometheus' not in hit['_source']):
      continue
    if ('metrics' not in hit['_source']['prometheus']):
      continue

    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in training_metrics):
        training_metrics[key] = []
      training_metrics[key].append(captured_metrics[key])

  models_dict = {}
  for key in training_metrics:
## Get training data for that particular metric
    X_train = pd.DataFrame(training_metrics[key], columns=['sample'])
    clf = IsolationForest(max_samples=1000)
    models_dict[key] = clf

  print ("Models trained for :" +str( len(models_dict)) + " different metrics")
  for model in models_dict:
    print (models_dict[model])
  testing_metrics = {}
  testing_metrics_document_tracker = {}
  for hit in testing_hits:
    if ('prometheus' not in hit['_source']):
      continue
    if ('metrics' not in hit['_source']['prometheus']):
      continue
    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in testing_metrics):
        testing_metrics[key] = []
        testing_metrics_document_tracker = []
      testing_metrics[key].append(captured_metrics[key])
      row = (hit['_index'], hit['_type'],hit['_id'])
      testing_metrics_document_tracker.append(row)


  for key in testing_metrics:
    if key in models_dict:
      X_test = pd.DataFrame(testing_metrics[key], columns=['sample'])
      outliers = [1]
      print (key)
      if (len(testing_metrics[key]) > 10):
        outliers = models_dict[key].predict(X_test)
      if (len(outliers) != len(testing_metrics[key])):
        raise "Assertion: prediction not done for all testing samples"
      for i in range(0, len(outliers)):
        label = outliers[i]
        docId = testing_metrics_document_tracker[i][2]
        docType = testing_metrics_document_tracker[i][1]
        docIndex = testing_metrics_document_tracker[i][0]
        partial_body_str = '{"doc":{"outlier":{"' + key +'":' + label + '}}}' 
        

        es.update(index=docIndex,doc_type=docType,id=docId,body=partial_body_str)

def compute_outliers_Version3(training_hits, testing_hits, es):

  training_metrics = {}
  testing_metrics = {}
  for hit in training_hits:

    if ('prometheus' not in hit['_source']):
      continue

    if ('metrics' not in hit['_source']['prometheus']):
      continue

    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in training_metrics):
        training_metrics[key] = []
      training_metrics[key].append(captured_metrics[key])

  test_hit_tracker = {}
  for hit in testing_hits:

    if ('prometheus' not in hit['_source']):
      continue

    if ('metrics' not in hit['_source']['prometheus']):
      continue


    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in testing_metrics):
        testing_metrics[key] = []
        test_hit_tracker[key] = []

      docId = hit['_id']
      docType = hit['_type']
      docIndex = hit['_index']
      row = (docIndex, docType, docId, captured_metrics[key])

      test_hit_tracker[key].append(row)
      testing_metrics[key].append(captured_metrics[key])

  print ("Number of training samples:" + str(len(training_hits)))
  print ("Number of testing samples:" + str(len(testing_hits)))
  print ("total number of tracked metrics:" + str(len(training_metrics)))
  
  for key in training_metrics:
    X_train = pd.DataFrame(training_metrics[key], columns=['sample'])
    clf = IsolationForest(max_samples=1000)
    if (key not in testing_metrics):
      continue
    X_test = pd.DataFrame(testing_metrics[key], columns=['sample'])
    clf.fit(X_train)
    outliers = clf.predict(X_test)
    ratio = (1.0* list(X_test).count(-1))/X_test.shape[0]
    print(key + " Outlier ratio:", list(X_test).count(-1)/X_test.shape[0])
    if (len(outliers) != len(test_hit_tracker[key])):
        raise "Predictions not matching number test hits"

    for i in range(0,len(outliers)):
      label = outliers[i]
      if (outliers[i] == -1):
        label = 1
      else:
        label= 0
      docId = test_hit_tracker[key][i][2]
      docType = test_hit_tracker[key][i][1]
      docIndex = test_hit_tracker[key][i][0]
      if (testing_metrics[key][i] != test_hit_tracker[key][i][3]):
        raise "Exception values do not match"
      es.update(index=docIndex,doc_type=docType,id=docId,body='{"doc":{"outlier":{"' + key +'":' + str(label) + '}}}')

    
def compute_outliers_allmetrics(training_hits, testing_hits, es):

  training_metrics = {}
  testing_metrics = {}
  for hit in training_hits:
    
    if ('prometheus' not in hit['_source']):
      continue
    if ('metrics' not in hit['_source']['prometheus']):
      continue

    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in training_metrics):
        training_metrics[key] = []
      training_metrics[key].append(captured_metrics[key])

  for hit in testing_hits:
#  if ('prometheus' not in hit['_source']):
#      continue
#    if ('metrics' not in hit['_source']['prometheus']):
#      continue

   
    captured_metrics = hit['_source']['prometheus']['metrics']
    for key in captured_metrics:
      if (key not in testing_metrics):
        testing_metrics[key] = []
      testing_metrics[key].append(captured_metrics[key])

  print (len(testing_hits))
  for key in training_metrics:
    X_train = pd.DataFrame(training_metrics[key], columns=['sample'])
    clf = IsolationForest(max_samples=1000)
    
    X_test = pd.DataFrame(testing_metrics[key], columns=['sample'])
    clf.fit(X_train)

    for hit in testing_hits:
      if (i > len(outliers)):
        es.update(index=hit['_index'],doc_type=hit['_type'],id=hit['_id'],body='{"doc":{"outlier":{"' + key +'":' + "1" + '}}}')
        print ("Array out of bounds for outliers " + i)
        continue
      es.update(index=hit['_index'],doc_type=hit['_type'],id=hit['_id'],body='{"doc":{"outlier":{"' + key +'":' + str(outliers[i]) + '}}}') 
      print (hit['_index'])
      print (hit['_id'])
      print (hit['_type'])
      if (i % 100):
        print ("Documented updated " + str(i))
      i+=1

def main_fn():
  
  es = connect_to_remote_elasticsearch()
  logging.basicConfig(level=logging.ERROR)
  index_name = os.environ['INDEX'] # ## name of the index from which the metric documents are read
  ## we issue a query to index to retrieve documents from the index sorted by this sortkey (ex: @timeStamp)
  sort_key = '@timestamp' # documents returned from elastic search will be sorted by this key 
  
## Starting and end time stamps for training data for training anomaly detector 
## Starting and end time stamps for the test data on which anaomaly detection is run
## These time stamps are computed based on duration
  (training_start_timestamp, training_end_timestamp, test_start_time_stamp, test_end_time_stamp )= compute_timestamps(os.environ['DURATION_IN_MINS']) 
  training_start_timestamp = training_start_timestamp.strip()
  training_end_timestamp = training_end_timestamp.strip()
  test_start_time_stamp = test_start_time_stamp.strip()
  test_end_time_stamp = test_end_time_stamp.strip()
## Index in which the predicted results of outlier (-1) or not-outlier (1) is stored
  sort_key = sort_key.replace('--','').strip()
  
  if es is not None:
    ## This search query retrieves all documents between start_time and end_timestamp for training data
    ## the retrieved results are sorted by timestamp
    duration = int(os.environ['DURATION_IN_MINS'])
    GTE = 'now-' + str(2*duration+1) + 'm'
    LTE = 'now-' + str(duration+1) + 'm'
    search_query = '{"sort":[{"@timestamp":{"order":"asc"}}],"from":0,"size":1000,"query":{"bool":{"must":{"exists":{"field":"prometheus"}},"filter":{"range":{"@timestamp":{"gte":"'+GTE +'","lte":"'+ LTE+ '"}}}}}}'
    print("Search Query for training:" + search_query)
    response= es.search(index=str(index_name),body=search_query)
    print ("Length of response:" + str(len(response)))
    print (response)
    metrics = {}
    sort_key_list = []

    ## The response contains a field called hits which in turn has another field called hits
    ## extract the hits
    training_hits = response["hits"]["hits"]
    ## The response contains a field called hits which in turn has another field called hits
    ## extract the hits
    GTE = 'now-' + str(duration) + 'm'
    LTE = 'now'

    search_query = '{"sort":[{"@timestamp":{"order":"asc"}}],"from":0,"size":1000,"query":{"bool":{"must":{"exists":{"field":"prometheus"}},"filter":{"range":{"@timestamp":{"gte":"'+GTE +'","lte":"'+ LTE+ '"}}}}}}'
    print("Search Query for testing:" + search_query)
    response= es.search(index=str(index_name),body=search_query)
    testing_hits = response["hits"]["hits"]
    print ("Length of response:" + str(len(response)))
    print (response)
    testing_hits = response["hits"]["hits"]
    compute_outliers_Version3(training_hits, testing_hits, es)
    
if __name__ == "__main__":
  for key in os.environ:
    print (key + ":" + os.environ[key])
  
  main_fn()
