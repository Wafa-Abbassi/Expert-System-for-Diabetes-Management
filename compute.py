from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

from model import InputForm
from flask import Flask, render_template, request

def compute ( a,b,c,d,e,z,g,h):


# Preparing the data:
      data_file_name = 'diabetes-wisconsin.data.txt'

      first_line = "id,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome"
      with open(data_file_name, "r+") as f:
           content = f.read()
           f.seek(0, 0)
           f.write(first_line.rstrip('\r\n') + '\n' + content)
      df = pd.read_csv(data_file_name)
      df.replace('?', np.nan, inplace = True)
      df.dropna(inplace=True)
      df.drop(['id'], axis = 1, inplace = True)

      df['Outcome'].replace('2',0, inplace = True)
      df['Outcome'].replace('4',1, inplace = True)

      df.to_csv("combined_data.csv", index = False)

# Data sets
      DIABETES_TRAINING = "diabetes_training.csv"
      DIABETES_TEST = "diabetes_test.csv"

# Load datasets.
      training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TRAINING,
                                                       target_dtype=np.int, features_dtype=np.int)
      test_set =     tf.contrib.learn.datasets.base.load_csv_with_header(filename=DIABETES_TEST,
                                                   target_dtype=np.int, features_dtype=np.int)

      feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]
        
      classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=2,
                                              model_dir="/tmp/iris_model")

# Fit model.
      classifier = classifier.fit(training_set.data, training_set.target, steps=2000)
      k =a
      l = b
      m =c
      n= d
      o = e
      p = z
      q = g
      r = h
               
      def new_samples():
          return np.array([[k, l, m, n, o, p, q, r],
                   ], dtype=np.float32)

      r = list(classifier.predict(input_fn=new_samples))    


      if (r == [[1]]):
            return "type1"
      else:
            return "type2"

if __name__ == '__main__':
    
      print (compute(a,b,c,d,e,f,g,h))

