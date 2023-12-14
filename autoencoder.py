# Autoencoder Research Project
# autoencoder.py


import os
import sys
import io
import time
import datetime as dt
import pprint as ppr
import random as rnd
import numpy as np
import pandas as pd
import sklearn as sk
import torch as tor
import skorch as skr
import seaborn as sea
import matplotlib as mpl


################################################################
# Config
################################################################

run_time = dt.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

data_file_source       = 'kddcup.data.gz'
data_file_train        = 'train.data.gz'
data_file_test_normal  = 'test_normal.data.gz'
data_file_test_anomaly = 'test_anomaly.data.gz'
data_file_names = [
  "duration",
  "protocol_type",
  "service",
  "flag",
  "src_bytes",
  "dst_bytes",
  "land",
  "wrong_fragment",
  "urgent",
  "hot",
  "num_failed_logins",
  "logged_in",
  "num_compromised",
  "root_shell",
  "su_attempted",
  "num_root",
  "num_file_creations",
  "num_shells",
  "num_access_files",
  "num_outbound_cmds",
  "is_host_login",
  "is_guest_login",
  "count",
  "srv_count",
  "serror_rate",
  "srv_serror_rate",
  "rerror_rate",
  "srv_rerror_rate",
  "same_srv_rate",
  "diff_srv_rate",
  "srv_diff_host_rate",
  "dst_host_count",
  "dst_host_srv_count",
  "dst_host_same_srv_rate",
  "dst_host_diff_srv_rate",
  "dst_host_same_src_port_rate",
  "dst_host_srv_diff_host_rate",
  "dst_host_serror_rate",
  "dst_host_srv_serror_rate",
  "dst_host_rerror_rate",
  "dst_host_srv_rerror_rate",
  "label"]

data_display_width = 120 
data_display_cols  = 50
data_display_rows  = 4

pp = ppr.pprint
pf = ppr.pformat


################################################################
# Model
################################################################

class NeuralNet:
  def __init__(self, args=None):
    super(NeuralNet, self).__init__()

class Autoencoder(tor.nn.Module):
  def __init__(self, args=None):
    super(Autoencoder, self).__init__()
    self.args = args


################################################################
# Data
################################################################

def load_data(file_name=data_file_source):
  print(f"Loading data {file_name}...")
  data =  pd.read_csv(data_file_source, 
                      compression='gzip', 
                      names=data_file_names, 
                      index_col=False)
  print("Data loaded.")
  print()
  
  return data


def save_data(data, file_name):
  print(f"Saving data {file_name}...")
  data.to_csv(file_name,
              compression='gzip',
              header=data_file_names)
  print("Data saved.")
  print()


def print_data(data, show=False):
  pd.options.display.width = data_display_width
  pd.options.display.max_columns = data_display_cols
  pd.options.display.max_rows = data_display_rows
  num_rows, num_cols = data.shape
  if show == True: print(data)
  
  print("Data Info")
  print(f"  rows: {num_rows}")
  print(f"  cols: {num_cols}")
  print()


def plot_data(data, mode='categorical', feature='label'):
  if mode == 'categorical':
    dist = data.value_counts(feature)
    dist = data[feature].value_counts()
    sea.barplot(dist)
  elif mode == 'continuous':
    pass
  mpl.pyplot.xticks(rotation=90)
  mpl.pyplot.show()    


def clean_data(data):
  print("Checking data for missing values in columns...")
  if data.isna().any().any():
    print("Missing values found.")
  else:
    print("No missing values.")
  print()
  print("Checking data for duplicate rows...")
  print(f"Found {data.duplicated().sum()} duplicates.")
  print()
  print("Dropping duplicates...")
  data_clean = data.drop_duplicates()
  print()
  print_data(data_clean)
  
  return data_clean


def split_data(data):
  return
  save_data(data_clean, data_file_train)  
    

################################################################
# Training
################################################################


################################################################
# Testing
################################################################



################################################################
# Main
################################################################
   
def main():
  print()
  print("----------------------------------------------------------------")
  print("Autoencoder Research Project")
  print("----------------------------------------------------------------")
  print()
  print(f"Starting run at {run_time}")
  print()

  data = load_data()
  data = clean_data(data)
  split_data(data)
  

  print("Run complete.")
  print()
  sys.exit()

  # print_data(data.loc[data['label'] == 'normal.'])
  # print_data(data)
  # plot_data(data)

   #data = preprocess_data(data)

  # df, labels, num_normal,input_size = preprocess_data(df)
  # anomal_ratio = 0.001
  # reduce_data= reduce_data(df, labels, num_normal, anomal_ratio)
  # print('reduced_data shape = ', reduced_data.shape)

  print()
  print("----------------------------------------------------------------")
  print()

    
if __name__ == "__main__":
  main()

