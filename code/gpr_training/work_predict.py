import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.options.mode.chained_assignment = None

import glob
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import copy

import tqdm.notebook as tqdm
import pickle

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP

from torch.optim.lr_scheduler import StepLR

import torch.nn as nn

city = sys.argv[1]

dir = '../results/'
data_csv_file_name = 'WORK4FOOD'
# Result file name WORK4FOOD_0_24_25_static


# TODO
pay_guarantee_per_second_postfix = '_25_static_frac'
frac_drivers = ["100", "90", "80", "70", "60"]
# pay_guarantee_per_second_postfix = '_25_static'
# frac_drivers = [""]

# os.chdir(dir)

model_n = 2
mode = 'train'
gp_output = 'pay'
days = [2,5]
if (mode == 'train'):
    if (days.__len__() == (days[-1] - days[0] + 1)):
      model_path = '../gpr_models/'+f'model_{city}_days_{days[0]}_to_{days[-1]}{pay_guarantee_per_second_postfix}_{gp_output}_{model_n}.pth'
    else:
        model_path = '../gpr_models/'+f'model_{city}_days_{days[0]}_and_{days[-1]}{pay_guarantee_per_second_postfix}_{gp_output}_{model_n}.pth'
else:
  model_path = '../gpr_models/'+f'model_{city}_days_{1}_to_{2}{pay_guarantee_per_second_postfix}_{gp_output}_{model_n}.pth'


print("model path:", model_path)

data_value_function = pd.DataFrame(columns=['vehicle_id', 'curr_time', 'carrying_orders', 'event', 'vh_lat', 'vh_lon', 'day', 'shift', 'start_time', 'stop_time', 'distance_till_now', 'travel_time_till_now', 'more_distance_till_shift_end', 'more_travel_time_till_shift_end', 'wait_time_till_now', 'more_wait_time_till_shift_end', 'n_drivers_window', 'n_orders_window'])
data_value_function_list = []

t_start = time.time()
for day in days:
  print('day:', day)

  st_hr = 0
  end_hr = 24

  print('reading de_intervals...')

  if (os.path.isfile(f'de_intervals/{city}_{day}.csv')):
    driver_available = pd.read_csv(f'de_intervals/{city}_{day}.csv')
  else:
    csv_files_avail = glob.glob(f'../data/data_{city}_anonymized/food_data/{day}/de_intervals/*.csv')
    driver_available = pd.DataFrame(columns=['vehicle_id', 'start_time', 'stop_time'])
    i = 0
    for filename in csv_files_avail:
        vehicle_id = os.path.splitext(os.path.basename(filename))[0]
        with open(filename) as f:
            next(f)
            for line in f:
                start_time = line.split(None, 1)[0]
                nextline = next(f)
                end_time = nextline.split(None, 1)[0]
                driver_available.loc[i] = [vehicle_id, start_time, end_time]
                i += 1
    driver_available[['vehicle_id', 'start_time', 'stop_time']].to_csv(f'de_intervals/{city}_{day}.csv')

  assert(driver_available.__len__() > 0)
  driver_available = driver_available.astype(float)
  vehicles = driver_available.vehicle_id.unique()
  max_shifts = driver_available.groupby(by='vehicle_id').count()['start_time'].max()

  print('filling value function ...',)

  if (os.path.isfile(f'pay_data/{city}_{day}{pay_guarantee_per_second_postfix}_{str(frac_drivers.__len__())}.csv')):
    data_value_function = data_value_function.append(pd.read_csv(f'pay_data/{city}_{day}{pay_guarantee_per_second_postfix}_{str(frac_drivers.__len__())}.csv')[['vehicle_id', 'curr_time', 'carrying_orders', 'event', 'vh_lat', 'vh_lon', 'day', 'shift', 'start_time', 'stop_time', 'distance_till_now', 'travel_time_till_now', 'more_distance_till_shift_end', 'more_travel_time_till_shift_end', 'wait_time_till_now', 'more_wait_time_till_shift_end', 'n_drivers_window', 'n_orders_window']])
  else:
    for fd in frac_drivers:
      print('frac', fd)
      csv_file_sim = f'../results/{city}/{day}/{data_csv_file_name}_0_24{pay_guarantee_per_second_postfix}{fd}.results' 
      print("result file", csv_file_sim)
      data = pd.read_csv(csv_file_sim, names=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"])
      data_move = data[data['a'] == "MOVE"].drop(['a', 'c', 'k', 'l'], axis = 1)
      data_move.columns = ['vehicle_id', 'curr_time', 'dist_travelled',
                              'carrying_orders', 'time_travelled', 'event', 'vh_lat', 'vh_lon']

      data_assign = data[data['a'] == "ASSIGN"].drop(['a', 'c', 'e', 'f', 'g', 'h', 'i', 'j'], axis = 1)
      data_assign.columns = ['order_id', 'assigned_time', 'n_drivers_window', 'n_orders_window']
      data_assign = data_assign.groupby('order_id').max()

      data_deliver = data[data['a'] == "DELIVER"].drop(['a', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], axis = 1)
      data_deliver.columns = ['order_id', 'delivered_time', 'vehicle_id']
      data_deliver = data_deliver.set_index('order_id')

      data_picked = data[data['a'] == "PICKEDUP"].drop(['a', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], axis = 1)
      data_picked.columns = ['order_id', 'picked_time']
      data_picked = data_picked.set_index('order_id')

      data_reached = data[data['a'] == "REACHED"].drop(['a', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], axis = 1)
      data_reached.columns = ['order_id', 'reached_time']
      data_reached = data_reached.groupby('order_id').min()

      data_order = pd.concat([data_deliver, data_picked, data_reached, data_assign], join='inner', axis=1)
      data_order['wait_time'] = data_order['picked_time'] - data_order['reached_time']

      data_move['vehicle_id'] = data_move['vehicle_id'].astype('int')
      data_move_grouped = dict(tuple(data_move.groupby('vehicle_id')))
      data_order['vehicle_id'] = data_order['vehicle_id'].astype('int')
      data_order_grouped = dict(tuple(data_order.groupby('vehicle_id')))
      
      for n in range(0,max_shifts):
          driver_available_n = driver_available.groupby('vehicle_id').nth(n)
          print(' shift:', n)
          for v, dan_row in driver_available_n.iterrows():
              
              start_time = int(dan_row.start_time)
              stop_time = int(dan_row.stop_time)
              
              try:
                temp = data_move_grouped[int(v)]
              except:
                continue
            #   if (temp.curr_time >= 0).sum() == 0:
            #     start_time -= (0 + 86400*(day-1))
            #     stop_time -= (0 + 86400*(day-1))
                
              temp = temp[(temp.curr_time<stop_time) & (temp.curr_time>start_time)]
              
              if (temp.__len__() == 0):
                  continue   
              temp['day'] = day
              temp['shift'] = n
              temp['start_time'] = start_time
              temp['stop_time'] = stop_time
            #   if (temp.curr_time >= 0).sum() > 0:
            #     temp['curr_time'] = temp['curr_time'] - (0 + 86400*(day-1))
            #     temp['start_time'] = temp['start_time'] - (0 + 86400*(day-1))
            #     temp['stop_time'] = temp['stop_time'] - (0 + 86400*(day-1))
              temp['distance_till_now'] = temp['dist_travelled'].cumsum()
              temp['travel_time_till_now'] = temp['time_travelled'].cumsum()
              temp['more_distance_till_shift_end'] = temp['distance_till_now'].iloc[-1] - temp['distance_till_now']
              temp['more_travel_time_till_shift_end'] = temp['travel_time_till_now'].iloc[-1] - temp['travel_time_till_now']
              temp = temp.drop(columns = ['dist_travelled', 'time_travelled'])
              temp_list = temp.values.tolist()
              
              try:
                temp2 = data_order_grouped[int(v)]  
              except:
                continue
              temp2 = temp2[(temp2.reached_time<stop_time) & (temp2.reached_time>start_time)]
              if (temp2.__len__() == 0):
                  continue
              temp2['curr_time'] = temp2['picked_time'].copy()
              temp2['day'] = day
              temp2['shift'] = n
              temp2['start_time'] = start_time
              temp2['stop_time'] = stop_time
              temp2['wait_time_till_now'] = temp2['wait_time'].cumsum()
              temp2['more_wait_time_till_shift_end'] = temp2['wait_time_till_now'].iloc[-1] - temp2['wait_time_till_now']
              temp2 = temp2.drop(columns = ['delivered_time', 'picked_time', 'reached_time', 'assigned_time', 'wait_time'])
              temp2.reset_index(drop=True, inplace=True)
              temp2_list = temp2.values.tolist()
             
              
              temp3 = pd.DataFrame(columns=['vehicle_id', 'curr_time', 'carrying_orders', 'event',
                                  'vh_lat', 'vh_lon', 'day', 'shift', 'start_time', 'stop_time', 'distance_till_now', 'travel_time_till_now',
                                  'more_distance_till_shift_end', 'more_travel_time_till_shift_end', 'wait_time_till_now', 'more_wait_time_till_shift_end', 
                                  'n_drivers_window', 'n_orders_window'])
              k = 0
              for j in range(temp2.__len__()):
                k = (temp['curr_time'] > temp2.iloc[j]['curr_time']).values.argmax()
                if(temp.iloc[k]['curr_time'] < temp2.iloc[j]['curr_time']):
                  k = temp.__len__() -1;
                temp3 = temp.iloc[k]
                temp3['wait_time_till_now'] = temp2.iloc[j]['wait_time_till_now']
                temp3['more_wait_time_till_shift_end'] = temp2.iloc[j]['more_wait_time_till_shift_end']
                temp3['n_drivers_window'] = temp2.iloc[j]['n_drivers_window']
                temp3['n_orders_window'] = temp2.iloc[j]['n_orders_window']
                temp3_list = temp3.values.tolist()
                data_value_function_list.append(temp3_list)
              
    data_value_function = pd.DataFrame(data_value_function_list, columns=['vehicle_id', 'curr_time', 'carrying_orders', 'event', 'vh_lat', 'vh_lon', 
    'day', 'shift', 'start_time', 'stop_time', 'distance_till_now', 'travel_time_till_now', 'more_distance_till_shift_end', 'more_travel_time_till_shift_end',
    'wait_time_till_now', 'more_wait_time_till_shift_end', 'n_drivers_window', 'n_orders_window'])
    
    data_value_function.reset_index()
    data_value_function[data_value_function.day == day].to_csv(f'pay_data/{city}_{day}{pay_guarantee_per_second_postfix}_{str(frac_drivers.__len__())}.csv')

t_end = time.time()
print("time taken to preprocess:", (t_end-t_start)/60, "mins")

driver_available['active_time'] = driver_available['stop_time']  - driver_available['start_time'] 

if (mode == 'train') or (mode == 'train again'):
  assert(data_value_function.day.nunique() == (days.__len__()))

# Defining the Main GP model class
class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
      """
      This initializes the model with some standard settings including the variational distribution to be used, the number of inducing points
      being used along with the mean and the covariance functions being used. 

      Below ConstantMean refers to the Mean being a constant. This means that the mean learnt will act as an offset of sorts whereas all the 
      variance in the data will be explained by the covariance function. The Mean can also be set to linear where it becomes a linear function
      of the features. 

      A basic RBF kernel has been used here that learns one lengthscale for all the dimensions. ScaleKernel below just imbues the RBF Kernel
      with a scalar scale value that can scale the value of the RBF kernel. 
      """
      variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
      variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
      super(GPModel, self).__init__(variational_strategy)
      self.mean_module = gpytorch.means.ConstantMean()
      self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
      """
      This is akin to the forward pass in Neural Nets in a way. Here the mean and the covariance functions are calculated on some given x.
      Then, using that mean and covariance function, a MultivariateNormal distribtuion is represented. 
      """
      mean_x = self.mean_module(x)
      covar_x = self.covar_module(x)
      return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#### functions for getting hyperparameter values (the details of these can be ignored)
def hyperparams_simple(model):
    """
    Print the models hyperparameters for hyperparameter exploration. Only works for kernels not using additive or product 
    kernels
    """
    kern_dict = {}
    current = model.covar_module
    count = 0
    while True:
        name = str(count) + "_" + str(type(current))[25:-2]
        children = get_children(current)
        cons_hp = [i for i in current.named_parameters_and_constraints()]
        kern_dict[name] = get_hp_dict(cons_hp)
        if children == []:
            break
        else:
            current = children[0][1]
        count += 1
    return pd.DataFrame(kern_dict).T
    
def get_children(current):
    children = []
    for i in current.named_children():
        if "kernel" not in i[0]:
            pass
        else:
            children.append(i)
    return children

def get_hp_dict(cons_hp):
    
    hp_dict = {}
    for i in cons_hp:
        if ("base_kernel" in i[0]) or ("kernels" in i[0]):
            pass
        else:
            curr_item = i[1]
            if curr_item.shape==():
                curr_item_list = [i[2].transform(curr_item)]
            else:
                curr_item_list = []
                for item in curr_item:
                    curr_item_list.append(i[2].transform(item)) 
            hp_dict[i[0][4:]] = curr_item_list[0].cpu().detach().numpy()
    return hp_dict


data_value_function.stop_time = data_value_function.stop_time.astype('float64')
data_value_function.start_time = data_value_function.start_time.astype('float64')
data_value_function.vehicle_id = data_value_function.vehicle_id.astype('int')


if(gp_output == 'pay'):
  df_gp = data_value_function[['vehicle_id', 'curr_time', 'vh_lat', 'vh_lon', 'start_time', 'stop_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'n_drivers_window', 'n_orders_window']]
  df_gp['more_pay_till_shift_end'] = data_value_function['more_travel_time_till_shift_end'] + data_value_function['more_wait_time_till_shift_end']
else:
  print('gp_output not supported')
df_gp['elapsed_time'] = df_gp['curr_time'] - df_gp['start_time']
df_gp['remaining_time'] = df_gp['stop_time'] - df_gp['curr_time']


if ('_frac' in pay_guarantee_per_second_postfix) or ('_dyngp_driverreject' in pay_guarantee_per_second_postfix):
    mean_temp = df_gp[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'n_drivers_window', 'n_orders_window', 'more_pay_till_shift_end']].mean().values
    std_temp = df_gp[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'n_drivers_window', 'n_orders_window', 'more_pay_till_shift_end']].std().values
else:
    mean_temp = df_gp[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'more_pay_till_shift_end']].mean().values
    std_temp = df_gp[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'more_pay_till_shift_end']].std().values

df_gp = (df_gp - df_gp.mean())/df_gp.std()

outF = open(model_path[:-3]+"parameters", "w")
for line in mean_temp:
  outF.write(str(line))
  outF.write("\n")
for line in std_temp:
  outF.write(str(line))
  outF.write("\n")
outF.close()

all_agents = df_gp.vehicle_id.unique()
train_agents = pd.Series(all_agents).sample(frac=0.75)
print(f"Total # agents = {len(all_agents)}\nTrain # agents = {len(train_agents)}")

df_gp = df_gp.sample(frac=1).reset_index(drop=True)

train_ratio = 0.75

df_train = df_gp[df_gp.vehicle_id.isin(train_agents)].sample(frac=1/(days.__len__())).reset_index(drop=True)
if ('_frac' in pay_guarantee_per_second_postfix) or ('_dyngp_driverreject' in pay_guarantee_per_second_postfix):
    train_x = df_train[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'n_drivers_window', 'n_orders_window']].values #.iloc[:int(len(df_gp)*train_ratio)]
else:
    train_x = df_train[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event']].values #.iloc[:int(len(df_gp)*train_ratio)]
train_y = df_train.more_pay_till_shift_end.values
train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)

df_test = df_gp[~df_gp.vehicle_id.isin(train_agents)].sample(frac=1).reset_index(drop=True)
if ('_frac' in pay_guarantee_per_second_postfix) or ('_dyngp_driverreject' in pay_guarantee_per_second_postfix):
    test_x = df_test[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event', 'n_drivers_window', 'n_orders_window']].values
else:
    test_x = df_test[['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event']].values
test_y = df_test.more_pay_till_shift_end.values
test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)

print(f'Length of train data = {len(train_x)}\nLength of test data = {len(test_x)}')

if torch.cuda.is_available():
    train_x, train_y = train_x.cuda(), train_y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y) 
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_dataset = TensorDataset(test_x, test_y) 
test_loader = DataLoader(test_dataset, shuffle=True)

inducing_points = train_x[:500, :]
if (mode == 'train'):
  model = GPModel(inducing_points=inducing_points)
  # The likelihood is learnt separate from the model. This refers to the epsilon/error term in the GP model formulation. 
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  # Below checks if the cuda/GPU is avaliable or not. If it is, then the model and likelihood both are transferred to the GPU
  if torch.cuda.is_available():
      model = model.cuda()
      likelihood = likelihood.cuda()
else:
  state_dict = torch.load(model_path)
  model = GPModel(inducing_points=inducing_points)  # Create a new GP model
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  if torch.cuda.is_available():
      model = model.cuda()
      likelihood = likelihood.cuda()
  model.load_state_dict(state_dict)


# Now we set both to training mode
model.train()
likelihood.train()

# The use of an optimizer is again very similar to how it is used in neural nets. lr is learning rate.
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.05)

# Our loss object. We're using the VariationalELBO. This defines the loss function we are trying to optimize
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

if (mode == 'test'):
  num_epochs = 0                  
if( mode == 'train') or (mode == 'train again'): 
  num_epochs = 5

epochs_iter = range(num_epochs)
for i in epochs_iter:
    count = 0
    mean_loss = 0
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = train_loader
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        count+=1
    if i%1 == 0:
        print("mean mll train Loss for Epoch", i,":", mean_loss/count)    
        mse = nn.MSELoss()
        test_output = model(test_x[:10000])
        test_loss = mse(test_output.loc, test_y[:10000])
        print('test mse loss', test_loss.item())

if (mode == 'train') or (mode == 'train again'): 
  print("Mean Loss for Epoch", i,":", mean_loss/count)    

if (mode == 'train') or (mode == 'train again'): 
  torch.save(model.state_dict(), model_path)

print('mode:', mode)
print('days:', days)
print('model path:', model_path)
if (mode == 'train') or (mode == 'train again'): 
  mse = nn.MSELoss()
  test_output = model(test_x[:10000])
  test_loss = mse(test_output.loc, test_y[:10000])

  train_output = model(train_x[:1000])
  train_loss = mse(train_output.loc, train_y[:1000])

  print('train_loss:', train_loss.item(), ', test_loss:', test_loss.item())
  print('\n')
  print('fraction of points within 100x times std:')
  print({10 * (i+1) : ((model(test_x[:1000]).loc - test_y[:1000]).abs() < 0.1 * (i+1)).sum().item() / test_y[:1000].__len__() for i in range(10)})

if(mode == 'test'):
  mse = nn.MSELoss()
  test_output = model(test_x[:10000])
  test_loss = mse(test_output.loc, test_y[:10000])
  print(test_loss.item())
  print('fraction of points within 100x times std:')
  print({10 * (i+1) : ((model(test_x[:1000]).loc - test_y[:1000]).abs() < 0.1 * (i+1)).sum().item() / test_y[:1000].__len__() for i in range(10)})

class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance

if (mode=='train'):
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model = GPModel(inducing_points=inducing_points.cpu())  # Create a new GP model
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  model.load_state_dict(state_dict)

  wrapped_model = MeanVarModelWrapper(model)

  with torch.no_grad(), gpytorch.settings.trace_mode():
      if ('_frac' in pay_guarantee_per_second_postfix) or ('_dyngp_driverreject' in pay_guarantee_per_second_postfix):
        fake_input = torch.rand(1024,11)
      else:
        fake_input = torch.rand(1024,9)
      pred = wrapped_model(fake_input)  # Compute caches
      traced_model = torch.jit.trace(wrapped_model, fake_input)

# mean1 = wrapped_model(test_x[1:100])[0]
# mean2 = traced_model(test_x[1:100])[0]

# print("from wrapped model", mean1)
# print("from traced model", mean2)

traced_model.save(model_path[:-3] + 'pt')