import pandas as pd
import glob
import time
from datetime import datetime
import numpy as np
import os
import statistics
import sys

city = sys.argv[1]
day = (int)(sys.argv[2])
st_day, end_day = day, day
algo = sys.argv[3]

if(algo == "WORK4FOOD"):
    min_wage_per_second = sys.argv[4]
else:
    min_wage_per_second = ''

base_dir = f"../analysis_results/{city}/{day}/"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

st_hr = 0
end_hr = 24
st = st_hr*3600
end = end_hr*3600

if (algo =="FOOD_MATCH"):
    csv_files = [f'../results/{city}/{day}/{algo}_0_24.results']
elif (algo == "FAIR_FOODY"):
    csv_files = [f'../results/{city}/{day}/{algo}_0_24.results']
elif (algo == "WORK4FOOD"):
    csv_files = [f'../results/{city}/{day}/{algo}_0_24{min_wage_per_second}.results']
else:
    csv_files = []
    print("`algo` not recognized")
    
list_data = []
list_data_move = []
avg_time = 0
avg_hungarian_time = 0
avg_cost_time = 0
overflow = 0
totalslots = 0
i=0
for filename in csv_files:
    data = pd.read_csv(filename, names=["a", "b", "c", "d", "e", "f", "g", "h"])
    data_cost_time = data[data['a'] == "cost_time"].drop(['a', "c", "d", "e", "f", "g", "h"], axis = 1).astype('float64')
    data_cost_time.columns = ['cost_time']
    data_hung_time = data[data['a'] == "hungarian_time"].drop(['a', "c", "d", "e", "f", "g", "h"], axis = 1).astype('float64')
    data_hung_time.columns = ['hungarian_time']
    print("proceesed file",i, "file name", filename, "size:", int(os.stat(filename).st_size/1000000), "MB")
    avg_time_taken_per_slot = (sum(data_hung_time['hungarian_time'])+sum(data_cost_time['cost_time']))/(data_cost_time.shape[0]*1000000)
    avg_time += avg_time_taken_per_slot
    avg_hungarian_time += sum(data_hung_time['hungarian_time'])
    avg_cost_time += sum(data_cost_time['cost_time'])
    i += 1
    assign_time = (data_cost_time.reset_index()['cost_time']+data_hung_time.reset_index()['hungarian_time'])/1000000
    overflow += len(assign_time[assign_time > 180])
    totalslots += len(assign_time)



avg_time_final = avg_time/i

result = "algo," + str(algo) + "\n" 
result += "min_wage_per_second," + str(min_wage_per_second) + "\n" 
result += "avg_window_running_time," + str(avg_time_final) + "s\n" 
result += "overflow_percentage," + str(100*overflow/totalslots) + "\n" 

analysis_result_file = base_dir+f"overflow_analysis_{city}_{day}_{algo}{min_wage_per_second if algo == 'WORK4FOOD' else ''}.csv"
f= open(analysis_result_file,"w")
f.write(result)
f.close()

print(result)

