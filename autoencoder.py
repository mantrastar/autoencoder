# Autoencoder Research Project
# autoencoder.py
#
# Ven


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
import sklearn.svm as svm
import torch as tor
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import skorch as skr
import seaborn as sea
import matplotlib as mpl
import psutil
import json
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper, MultiUseParameterConfig, LossWrapper

################################################################
# Config
################################################################

run_time = dt.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

model_file_basic_ae    = 'autoencoder.basic.model'
model_file_adver_ae    = 'autoencoder.adver.model'

data_file_source       = 'kddcup.data.gz'
data_file_train        = 'train.data.gz'
data_file_test_normal  = 'test_normal.data.gz'
data_file_test_anomaly = 'test_anomaly.data.gz'
data_file_names        = 'kddcup.data.names.csv'
data_file_names_source = [
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

class Encoder(nn.Module):
  def __init__(self, input_size, latent_dim, device):
    super(Encoder, self).__init__()
    
    self.model = nn.Sequential(
      nn.Linear(input_size, 128, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(128, 64, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(64, latent_dim, device=device)
    )
    
  def forward(self, x):
    return self.model(x)


class Decoder(nn.Module):
  def __init__(self, latent_dim, output_size, device):
    super(Decoder, self).__init__()
  
    self.model = nn.Sequential(
      nn.Linear(latent_dim, 64, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(64, 128, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(128, output_size, device=device)
    )
    
  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):
  def __init__(self, latent_dim, device):
    super(Discriminator, self).__init__()
    
    self.model = nn.Sequential(
      nn.Linear(latent_dim, 64, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(64, 32, device=device),
      nn.LeakyReLU(0.2),
      nn.Linear(32, latent_dim, device=device),
    )
    
  def forward(self, x):
    return self.model(x)


class Autoencoder(nn.Module):
  def __init__(self, input_size, latent_size, output_size, device):
    super(Autoencoder, self).__init__()
    
    self.encoder = Encoder(input_size, latent_size, device)
    self.decoder = Decoder(latent_size, output_size, device)

    # Create layer labels.
    setattr(self, f'1: encoder', self.encoder)
    setattr(self, f'2: decoder', self.decoder)

  def forward(self, x):
    x = getattr(self, f'1: encoder')(x)
    x = getattr(self, f'2: decoder')(x)
    return x


class AdvAutoencoder(nn.Module):
  def __init__(self, input_size, latent_size, output_size, device):
    super(AdvAutoencoder, self).__init__()

    self.encoder = Encoder(input_size, latent_size, device)
    self.decoder = Decoder(latent_size, output_size, device)
    self.discriminator = Discriminator(latent_size, device)

    # Create layer labels.
    setattr(self, f'1: encoder', self.encoder)
    setattr(self, f'2: decoder', self.decoder)
    setattr(self, f'3: discriminator', self.discriminator)

  def forward(self, x):
    z = getattr(self, f'1: encoder')(x)
    reconstructed = getattr(self, f'2: decoder')(z)
    validity = getattr(self, f'3: discriminator')(z)
    return reconstructed, validity


################################################################
# Data
################################################################

def load_data(file_name=data_file_source, header=data_file_names_source):
  print(f"Loading data {file_name}...")
  data =  pd.read_csv(file_name, 
                      compression='gzip', 
                      names=header, 
                      index_col=False)
  print("Data loaded.")
  print()
  
  return data


def save_data(data, file_name, header):
  print(f"Saving data {file_name}...")
  data.to_csv(file_name,
              compression='gzip',
              header=header,
              index=False)
  print("Data saved.")
  print()


def save_header(data, file_name):
  print(f"Saving column names {file_name}...")
  data.to_csv(file_name,
              index=False)
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


def make_tensor(data, device):
  return tor.tensor(data.values, dtype=tor.float32, device=device)


def process_data(data):
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
  data = data.drop_duplicates()

  print()
  print("One hot encoding categorical features...")
  data = data.join(pd.get_dummies(data['protocol_type'],
                                  prefix='protocol_type',
                                  dtype=int)).drop('protocol_type', axis=1)
  data = data.join(pd.get_dummies(data['service'],
                                  prefix='service',
                                  dtype=int)).drop('service', axis=1)
  data = data.join(pd.get_dummies(data['flag'],
                                  prefix='flag',
                                  dtype=int)).drop('flag', axis=1)

  print()
  print("Min-max scaling data...")
  data_num = data.select_dtypes(include=['number'])
  data_obj = data.select_dtypes(exclude=['number'])
  data_minmax = data_num.copy()
  for col in data_num.columns:
    min_col = data_num[col].min()
    max_col = data_num[col].max()
    if min_col != max_col:
      data_minmax[col] = (data_num[col] - min_col) / (max_col - min_col)
    else:
      data_minmax[col] = 0
  data = pd.concat([data_minmax, data_obj], axis=1)  

  print()
  print("Normalizing 'label' column...")
  data_normal = data.copy(deep=True)
  data_normal.label = data_normal.label.replace('normal.', 'normal')
  data_anomaly = data_normal.copy(deep=True)
  data_anomaly.label = data_anomaly['label'].apply(lambda x: 'anomaly' if x != 'normal' else x)
  data_normal = data_normal.loc[data_normal['label'] == 'normal']
  data_anomaly = data_anomaly.loc[data_anomaly['label'] != 'normal']

  print()
  print("Splitting data into normal train/test and anomaly test...")
  data_normal = data_normal.drop('label', axis=1)
  data_anomaly = data_anomaly.drop('label', axis=1)
  data_normal_train, data_normal_test = sk.model_selection.train_test_split(data_normal,
                                                                            test_size=0.2,
                                                                            random_state=7)

  print()
  print("Saving normal training and test data and anomaly test data...")
  save_data(data_normal_train, data_file_train, header=None)
  save_data(data_normal_test, data_file_test_normal, header=None)
  save_data(data_anomaly, data_file_test_anomaly, header=None)
  save_header(data_normal.head(0), data_file_names)

  print("Data processed and saved for training and testing.")
  print()

  return data_normal_train, data_normal_test, data_anomaly


################################################################
# Training
################################################################

def train(data,
          model_type='basic', 
          epochs=50, 
          batch_size=64, 
          latent_size=16, 
          learning_rate=0.001,
          device=tor.device('cpu'),
          profiler=None,
          use_data_parallel=False,
          use_pipeline_parallel=False):
  data_tensor = make_tensor(data, device)
  dataset = TensorDataset(data_tensor, data_tensor)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  # Initialize the model
  if model_type == 'basic':
    model = Autoencoder(input_size=data_tensor.shape[1],
                        latent_size=latent_size,
                        output_size=data_tensor.shape[1],
                        device=device)
    print("Training basic autoencoder...")
  elif model_type == 'adver':
    model = AdvAutoencoder(input_size=data_tensor.shape[1],
                           latent_size=latent_size,
                           output_size=data_tensor.shape[1],
                           device=device)
    parallel_type=""
    if use_data_parallel: 
      parallel_type=" data parallel" 
    if use_pipeline_parallel:
      parallel_type=" pipeline parallel"
    print(f"Training adversarial autoencoder{parallel_type}...")
  else:
    raise ValueError("Invalid model type specified")

  model.to(device)

  # Criterion and optimizer
  criterion_reconstruction = nn.MSELoss()
  criterion_discriminator = nn.BCEWithLogitsLoss()
  optimizer = opt.Adam(model.parameters(), lr=learning_rate)

  # Data Parallel
  if use_data_parallel:
    model = tor.nn.DataParallel(model)
    model.to(device)

  # Pipeline Parallel
  if use_pipeline_parallel:
    # Setup for torchrun (RANK is an Index for the
    # Process, and WORLD_SIZE is total Processes).
    tor.manual_seed(0)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    annotate_split_points(
        model,
        {
          "2: decoder": PipeSplitWrapper.SplitPoint.BEGINNING,
          "3: discriminator": PipeSplitWrapper.SplitPoint.BEGINNING
        }
    )

    x = tor.randn(data_tensor.shape[1])
    chunks = 1
    sample = (x, )
    pipe = Pipe.from_tracing(model, 
                             chunks,
                             example_args=sample)

    device = tor.device('mps')
    model.to(device)

  peak_mem = 0
  avg_peak_mem = 0
  run_time = 0
  avg_run_time = 0

  device = tor.device("mps")
  model.to(device)

  # Training loop
  for epoch in range(epochs):
    start_epoch_time = time.time()
    total_mem = 0
    total_rloss = 0
    total_dloss = 0
    i = 0

    if f'epoch_{epoch+1}' not in stats:
      stats[f'epoch_{epoch+1}'] = {}

    key = 'adver'
    if use_data_parallel: 
      key = 'adver_data' 
    if use_pipeline_parallel:
      key = 'adver_pipe' 

    stats[f'epoch_{epoch+1}'][key] = {}

    for inputs, _ in data_loader:
      # start_time = time.time()

      if model_type == 'basic':
        # Forward pass
        outputs = model(inputs)
        loss = criterion_reconstruction(outputs, inputs)  

        peak_mem = tor.mps.current_allocated_memory() / (1024*1024)
        peak_mem = psutil.virtual_memory().used / (1024*1024)
        print(f"  Peak Mem = {peak_mem}")

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        run_time = end_time - start_time        

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        # print(f"  Loss: {loss.item()}")
        # print(f"  Memory: {peak_mem:.2f}")
        # print(f"  Time: {run_time:.5f}")

      elif model_type == 'adver':
        # Forward pass
        reconstructed, validity = model(inputs)
 
        # Reconstruction loss
        reconstruction_loss = criterion_reconstruction(reconstructed, inputs)
        
        # Discriminator loss for real and fake samples
        real_samples = tor.randn(batch_size, latent_size, device=device) # Gaussian distribution
        if use_data_parallel:
          real_preds = model.module.discriminator(real_samples)
        else:
          real_preds = model.discriminator(real_samples)
        real_loss = criterion_discriminator(real_preds, tor.ones_like(real_preds))

        if use_data_parallel:
          fake_preds = model.module.discriminator(validity.detach())
        else:
          fake_preds = model.discriminator(validity.detach())
        fake_loss = criterion_discriminator(fake_preds, tor.zeros_like(fake_preds))

        # Total discriminator loss
        discriminator_loss = (real_loss + fake_loss) / 2

        # peak_mem = tor.mps.current_allocated_memory() / (1024*1024)
        peak_mem = psutil.virtual_memory().used / (1024*1024)

        # Backpropagation for autoencoder
        optimizer.zero_grad()
        reconstruction_loss.backward(retain_graph=True)
        optimizer.step()

        # Backpropagation for discriminator
        optimizer.zero_grad()
        discriminator_loss.backward()
        optimizer.step()

        total_mem += peak_mem
        total_rloss += reconstruction_loss.item()
        total_dloss += discriminator_loss.item()

        # print(f"Epoch [{epoch+1}/{epochs}]")
        # print(f"  Reconstruction Loss: {reconstruction_loss.item()}")
        # print(f"  Discriminator Loss: {discriminator_loss.item()}")
        # print(f"  Memory: {peak_mem:.2f}")
        # print(f"  Time: {run_time:.5f}")

        i += 1

    end_epoch_time = time.time()
    epoch_time = end_epoch_time - start_epoch_time

    avg_mem = total_mem / i
    avg_rloss = total_rloss / i
    avg_dloss = total_dloss / i

    stats[f'epoch_{epoch+1}'][key]['time'] = epoch_time
    stats[f'epoch_{epoch+1}'][key]['mem'] = avg_mem
    stats[f'epoch_{epoch+1}'][key]['rloss'] = avg_rloss
    stats[f'epoch_{epoch+1}'][key]['dloss'] = avg_dloss

    print(stats[f'epoch_{epoch+1}'])

  return model


################################################################
# Testing
################################################################

def test():
  pass


################################################################
# Results
################################################################

def results():
  pass


################################################################
# Main
################################################################
   
stats = {}

# Check if M1 GPU (or MPS) is available
device = tor.device("mps" if tor.backends.mps.is_available() else "cpu")

def main():
  print()
  print("----------------------------------------------------------------")
  print("Autoencoder Research Project")
  print("----------------------------------------------------------------")
  print()
  print(f"Starting run at {run_time}...")
  print()

  print("Running on device...")
  print(device)
  print()

  if (os.path.isfile(data_file_train) and
      os.path.isfile(data_file_test_normal) and
      os.path.isfile(data_file_test_anomaly) and
      os.path.isfile(data_file_names)):
    print("Data has been processed already, using existing files.")
    print()
    data_header = pd.read_csv(data_file_names, nrows=0).columns.tolist()
    data_normal_train = load_data(data_file_train, header=data_header)
    data_normal_test  = load_data(data_file_test_normal, header=data_header)
    data_anomaly      = load_data(data_file_test_anomaly, header=data_header)
  else:
    data = load_data()
    print("Processing data...")
    print()
    data_normal_train, data_normal_test, data_anomaly = process_data(data)

  print("Train data")
  print_data(data_normal_train, True)
  print(data_normal_train.info())
  print()
  print("Test data (normal)")
  print_data(data_normal_test, True)
  print(data_normal_test.info())
  print()
  print("Test data (anomaly)")
  print_data(data_anomaly, True) 
  print(data_anomaly.info())
  print()

  # Store stats for each training run for reporting and graphs.
  model_basic_stats = {}
  model_adver_stats = {}

  # model_basic = train(data_normal_train, model_type='basic')
  # print()
  # print("Saving basic autoencoder model...")
  # tor.save(model_basic, model_file_basic_ae)

  model_adver = train(data_normal_train, 
                      model_type='adver',
                      device=device)
  model_adver = train(data_normal_train, 
                      model_type='adver',
                      device=device,
                      use_data_parallel=True)
  model_adver = train(data_normal_train, 
                      model_type='adver',
                      device=device,
                      use_pipeline_parallel=True)

  print()
  print("Saving adversarial autoencoder model...")
  tor.save(model_adver, model_file_adver_ae)

  print()
  print("Saving stats...")

  with open("stats.json", "w") as file:
    json.dump(stats, file)        

  test()
  results()

  print()
  print("Run complete.")
  print()
  sys.exit()

  print()
  print("----------------------------------------------------------------")
  print()

  
if __name__ == "__main__":
  main()

