import datetime
from datetime import timedelta
import unittest

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

def queryConstructionFormat(query):
  stack = []
  for ch in query:
    if ch == '{':
      stack.append(ch)
    if ch == '}':
      stack = stack[-1:]
  return len(stack)
  
class TestingQueryGeneration(unittest.TestCase):

# Unit test for testing computation of timestamps for trianing and testing data
  def test_timestamp(self):
    (training_start_timestamp, training_end_timestamp, test_start_timestamp, test_end_timestamp) = compute_timestamps(1)
    self.assertTrue(training_start_timestamp < training_end_timestamp, True)
    self.assertTrue(training_start_timestamp < test_start_timestamp, True)

# Unit test for testing the validity of braces in a query

  def test_querybrackets(self):
    stack_length = queryConstructionFormat('{{{323}}}')
    self.assertTrue(stack_length, 0)

if __name__ == "__main__":
  unittest.main()
