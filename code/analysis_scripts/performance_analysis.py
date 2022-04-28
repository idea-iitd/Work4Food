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

try:
    if int(min_wage_per_second[1:3]) == 0:
        min_wage_options = [0, 25]
    else:
        min_wage_options = [0, int(min_wage_per_second[1:3])]
except:
    min_wage_options = [0, 25]

print('city:', city, ", day:", day,  ', algo:', algo, ', min_wage_per_second:', min_wage_per_second)

pay_wait_ratio = 1 
base_dir = f"../analysis_results/{city}/{day}/"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

def gini(data, mwo):
    """Compute Gini coefficient of array of values"""
    x = data.copy()
    for i in range(len(x)):
        if x[i] < mwo/100:
            x[i] = mwo/100
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

# def lorenz(x):
#     arr = x.copy()
#     arr.sort()
#     # this divides the prefix sum by the total sum
#     # this ensures all the values are between 0 and 1.0
#     scaled_prefix_sum = arr.cumsum() / arr.sum()
#     # this prepends the 0 value (because 0% of all people have 0% of all wealth)
#     return np.insert(scaled_prefix_sum, 0, 0)

# def lorenz_curve(X, X_label="", Y_label="", title=""): 
#     lorenz_curve = lorenz(X)

#     # we need the X values to be between 0.0 to 1.0
#     plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)
#     # plot the straight line perfect equality curve
#     plt.plot([0,1], [0,1], color='k') #black
    
#     plt.rc('xtick',labelsize=15)
#     plt.rc('ytick',labelsize=15)
#     plt.xlabel("Fraction of " + X_label, fontsize=15)
#     plt.ylabel("Fraction of " + Y_label, fontsize=15)
#     plt.title(title,fontsize = 20)


st_hr, end_hr = 0,24
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
list_data_sdt = []
list_data_move = []
list_data_guarantee = []
list_data_first_available = []
available_vehicles = []
for i, filename in enumerate(csv_files):
    data = pd.read_csv(filename, names=["a", "b", "c", "d", "e", "f", "g", "h"])

    data_sdt = data[data['a'] == "SDT"].drop(['a', 'g', 'h'], axis = 1)
    data_sdt.columns = ['order_id', 'sdt', 'ordered_time', 'prep_time', 'sla']
    data_sdt = data_sdt.set_index('order_id')

    data_sdt = data_sdt[(data_sdt.ordered_time >= st) & (data_sdt.ordered_time <= end)]

    data_assign = data[data['a'] == "ASSIGN"].drop(['a', 'c', 'e', 'f', 'g', 'h'], axis = 1)
    data_assign.columns = ['order_id', 'assigned_time']
    data_assign = data_assign.groupby('order_id').max()

    data_reject = data[data['a'] == 'REJECT'].drop(['a', 'c', 'd', 'e', 'f', 'g', 'h'], axis = 1)
    data_reject.columns = ['order_id']

    data_deliver = data[data['a'] == "DELIVER"].drop(['a', 'e', 'f', 'g', 'h'], axis = 1)
    data_deliver.columns = ['order_id', 'delivered_time', 'vehicle_id']
    data_deliver = data_deliver.set_index('order_id')

    data_picked = data[data['a'] == "PICKEDUP"].drop(['a', 'd', 'e', 'f', 'g', 'h'], axis = 1)
    data_picked.columns = ['order_id', 'picked_time']
    data_picked = data_picked.set_index('order_id')

    data_reached = data[data['a'] == "REACHED"].drop(['a', 'd', 'e', 'f', 'g', 'h'], axis = 1)
    data_reached.columns = ['order_id', 'reached_time']
    data_reached = data_reached.groupby('order_id').min()

    data_move = data[data['a'] == "MOVE"].drop(['a'], axis = 1)
    data_move.columns = ['vehicle_id', 'to_order', 'curr_time', 'dist_travelled',
                            'carrying_orders', 'time_travelled', 'event']
    data_move['to_order'] = data_move['to_order'].astype('int')
    
    data_guarantee = data[data['a'] == "WAGE_GUARANTEE"].drop(['a', 'g', 'h'], axis = 1)
    data_guarantee.columns = ['DE_ID', 'shift', 'pay_guarantee_dyn', 'shift_start', 'shift_stop']

    data_first_available = data[(data['a'].shift(-1) != "MARKED UNAVAILABLE") & (data['a'] == 'FIRST_CHECK_IS_ACTIVE')].drop(['a', 'e', 'g', 'h'], axis = 1)
    data_first_available.columns = ['DE_ID', 'shift_start', 'shift_stop', 'shift']
    
    df = pd.concat([data_sdt, data_deliver, data_picked, data_reached, data_assign], join='inner', axis=1)
    list_data_sdt.append(data_sdt)
    list_data.append(df)
    list_data_move.append(data_move)
    list_data_guarantee.append(data_guarantee)
    list_data_first_available.append(data_first_available)
    print("proceesed file", i+1, "file name", filename, "size:", int(os.stat(filename).st_size/(10**6)), "MB")

data_sdt_all = pd.concat(list_data_sdt, ignore_index=False)
data_all = pd.concat(list_data, ignore_index=False)
data_move_all = pd.concat(list_data_move, ignore_index=False)
data_guarantee_all = pd.concat(list_data_guarantee, ignore_index=False)
data_first_available_all = pd.concat(list_data_first_available, ignore_index=False)

df_active_hours_sliced = pd.DataFrame(columns=['DE_ID', 'DATE', 'HOUR', 'ACTIVE_MINUTE'])
if data_guarantee_all.__len__() > 0:
    for day in range(st_day, end_day+1):
        for hr in range(0,24):
            temp = data_guarantee_all[['shift_start', 'shift_stop']].clip(hr*3600, (hr+1)*3600)
            temp['DE_ID'] = data_guarantee_all['DE_ID']
            temp['DATE'] = f'2000-00-0{day}'
            temp['HOUR'] = hr
            temp['ACTIVE_MINUTE'] = (temp['shift_stop'] - temp['shift_start'])/60
            temp = temp.drop(['shift_start', 'shift_stop'], axis=1)
            df_active_hours_sliced = df_active_hours_sliced.append(temp)

if (df_active_hours_sliced['ACTIVE_MINUTE'].sum() == 0):
    # df_active_hours_sliced = pd.read_csv("{}_active_hours_sliced_new.csv".format(city))
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
        driver_available['vehicle_id'] = driver_available['vehicle_id'].astype('int')
        driver_available['start_time'] = driver_available['start_time'].astype('float')
        driver_available['stop_time'] = driver_available['stop_time'].astype('float')
        driver_available[['vehicle_id', 'start_time', 'stop_time']].to_csv(f'de_intervals/{city}_{day}.csv')

    for day in range(st_day, end_day+1):
        for hr in range(0,24):
            temp = driver_available[['start_time', 'stop_time']].clip(hr*3600, (hr+1)*3600)
            temp['DE_ID'] = driver_available['vehicle_id']
            temp['DATE'] = f'2000-00-0{day}'
            temp['HOUR'] = hr
            temp['ACTIVE_MINUTE'] = (temp['stop_time'] - temp['start_time'])/60
            temp = temp.drop(['start_time', 'stop_time'], axis=1)
            df_active_hours_sliced = df_active_hours_sliced.append(temp)


df_active_hours_sliced = df_active_hours_sliced[(df_active_hours_sliced['DATE']>=f'2000-00-0{st_day}') & (df_active_hours_sliced['DATE']<=f'2000-00-0{end_day}') & (df_active_hours_sliced['ACTIVE_MINUTE'] > 0)]
if (available_vehicles == []):
    available_vehicles = list(df_active_hours_sliced.DE_ID.unique())
df_active_hours_sliced = df_active_hours_sliced[df_active_hours_sliced.DE_ID.isin(available_vehicles)] 

total_active = df_active_hours_sliced.groupby('DE_ID').sum().reset_index()
total_active['DE_ID'] = total_active['DE_ID'].astype(int)

DE_travelled = data_move_all.groupby(['vehicle_id'], as_index=False).agg({'dist_travelled' : 'sum', 'time_travelled' : 'sum'})

df_order = data_all.reset_index()[['order_id', 'vehicle_id', 'delivered_time','picked_time', 'reached_time', 'ordered_time', 'sdt', 'sla']]

df_order['picked_time'] = df_order['picked_time'].astype('float')
df_order['reached_time'] = df_order['reached_time'].astype('float')
df_order['delivered_time'] = df_order['delivered_time'].astype('float')
df_order['ordered_time'] = df_order['ordered_time'].astype('float')

df_order['de_wait_time'] = df_order['picked_time'] - df_order['reached_time']
df_order['order_delivered_time'] = df_order['delivered_time'] - df_order['ordered_time']

DE_wait = df_order.groupby(['vehicle_id'], as_index=False).agg({'de_wait_time' : 'sum'})

df_vehicle_without_shifts = pd.merge(DE_travelled, DE_wait, left_on=['vehicle_id'], right_on=['vehicle_id'])

grouped_active_hours = df_active_hours_sliced.groupby(['DE_ID'], as_index=False).agg({'ACTIVE_MINUTE' : 'sum'})

df_vehicle = pd.merge(df_vehicle_without_shifts, grouped_active_hours, left_on=['vehicle_id'], right_on = ['DE_ID'])


df_daily_dynamic_wage_guarantee = data_guarantee_all[['DE_ID', 'pay_guarantee_dyn']].groupby('DE_ID').sum().reset_index()
df_daily_dynamic_wage_guarantee = pd.merge(df_daily_dynamic_wage_guarantee, grouped_active_hours, on='DE_ID', how='outer')
df_daily_dynamic_wage_guarantee['pay_guarantee_dyn'] /= 60
df_daily_dynamic_wage_guarantee['pay_guarantee_dyn'] = df_daily_dynamic_wage_guarantee['pay_guarantee_dyn'].fillna(0)
 
del df_vehicle['ACTIVE_MINUTE']

df_vehicle = pd.merge(df_vehicle, df_daily_dynamic_wage_guarantee, on='DE_ID', how='outer')
df_vehicle['vehicle_id'] = df_vehicle['DE_ID']

df_vehicle['time_travelled'] = df_vehicle['time_travelled'].fillna(0)
df_vehicle['de_wait_time'] = df_vehicle['de_wait_time'].fillna(0)
df_vehicle['pay'] = df_vehicle['time_travelled']/60 + pay_wait_ratio * df_vehicle['de_wait_time']/60

df_vehicle[f'pay_above_minwage_dyn'] = df_vehicle['pay'] - df_vehicle[f'pay_guarantee_dyn']
df_vehicle[f'pay_above_minwage_dyn'] = (df_vehicle[f'pay_above_minwage_dyn']>0) * df_vehicle[f'pay_above_minwage_dyn']

df_vehicle[f'pay_below_minwage_dyn'] = df_vehicle[f'pay_guarantee_dyn'] - df_vehicle['pay']
df_vehicle[f'pay_below_minwage_dyn'] = (df_vehicle[f'pay_below_minwage_dyn']>0) * df_vehicle[f'pay_below_minwage_dyn']
    
for mwo in min_wage_options:
    df_vehicle[f'pay_guarantee_{mwo}'] = mwo/100 * df_vehicle['ACTIVE_MINUTE'] 

    # Extra cost that the food delivery platform has to incur when a driver exceeds the minimum pay
    df_vehicle[f'pay_above_minwage_{mwo}'] = df_vehicle['pay'] - df_vehicle[f'pay_guarantee_{mwo}']
    df_vehicle[f'pay_above_minwage_{mwo}'] = (df_vehicle[f'pay_above_minwage_{mwo}']>0) * df_vehicle[f'pay_above_minwage_{mwo}']

    df_vehicle[f'pay_below_minwage_{mwo}'] = df_vehicle[f'pay_guarantee_{mwo}'] - df_vehicle['pay']
    df_vehicle[f'pay_below_minwage_{mwo}'] = (df_vehicle[f'pay_below_minwage_{mwo}']>0) * df_vehicle[f'pay_below_minwage_{mwo}']

df_vehicle['pay_per_active_hour'] = df_vehicle.apply(
    lambda row: ((row.time_travelled/60) + pay_wait_ratio * (row.de_wait_time/60))/(row.ACTIVE_MINUTE), axis=1
)
df_vehicle['pay_per_active_hour'] = df_vehicle['pay_per_active_hour'].fillna(0)

pay_per_active_hour_data = np.array(df_vehicle['pay_per_active_hour'])
pay_data = np.array(df_vehicle['pay'])

gini_normalized_foreach_mwo = [0 for mwo in min_wage_options]
for i, mwo in enumerate(min_wage_options):
    gini_normalized_foreach_mwo[i] = gini(pay_per_active_hour_data, mwo)
gini_normalized_mwo_dyn = gini(np.nan_to_num(np.array(np.maximum(df_vehicle.pay_guarantee_dyn, df_vehicle.pay)/df_vehicle.ACTIVE_MINUTE)), mwo=0)

gini_foreach_mwo = [0 for mwo in min_wage_options]
for i, mwo in enumerate(min_wage_options):
    gini_foreach_mwo[i] = gini(np.nan_to_num(np.array(np.maximum(df_vehicle[f'pay_guarantee_{mwo}'], df_vehicle.pay))), 0)
gini_mwo_dyn = gini(np.nan_to_num(np.array(np.maximum(df_vehicle.pay_guarantee_dyn, df_vehicle.pay))), mwo=0)

df_zero_pay_vehicles = df_active_hours_sliced[df_active_hours_sliced.DE_ID.isin(df_vehicle[df_vehicle['pay']==0].vehicle_id)] 

orders_delivered_30min = df_order[df_order['order_delivered_time']/60 < 30]['order_id'].nunique()
orders_delivered_40min = df_order[df_order['order_delivered_time']/60 < 40]['order_id'].nunique()
orders_delivered_50min = df_order[df_order['order_delivered_time']/60 < 50]['order_id'].nunique()
orders_delivered_60min = df_order[df_order['order_delivered_time']/60 < 60]['order_id'].nunique()
orders_delivered_45min = df_order[df_order['order_delivered_time']/60 < 45]['order_id'].nunique()

total_working_time = df_active_hours_sliced.ACTIVE_MINUTE.sum()

orders_assigned = df_order['order_id'].nunique()
# print("orders_assigned", orders_assigned)
total_delivery_time = sum(df_order['order_delivered_time']/60)
total_payment = sum(df_vehicle['pay'])
total_payment_above_minwage_dyn = df_vehicle[f'pay_above_minwage_dyn'].sum()
total_payment_below_minwage_dyn = df_vehicle[f'pay_below_minwage_dyn'].sum()
total_cost_to_platform_dyn = df_vehicle[f'pay_above_minwage_dyn'].sum() + df_vehicle[f'pay_guarantee_dyn'].sum()
total_payment_above_minwage_foreach_mwo = [df_vehicle[f'pay_above_minwage_{mwo}'].sum() for mwo in min_wage_options]
total_payment_below_minwage_foreach_mwo = [df_vehicle[f'pay_below_minwage_{mwo}'].sum() for mwo in min_wage_options]
total_cost_to_platform_foreach_mwo = [df_vehicle[f'pay_above_minwage_{mwo}'].sum() + df_vehicle[f'pay_guarantee_{mwo}'].sum() for mwo in min_wage_options]
num_drivers_below_minwage_dyn = sum(df_vehicle['pay_below_minwage_dyn']>0)
num_drivers_below_minwage_foreach_mwo = [sum(df_vehicle[f'pay_below_minwage_{mwo}']>0) for mwo in min_wage_options]
var_pay = statistics.variance(df_vehicle['pay'])
gap_pay = max(df_vehicle['pay']) - min(df_vehicle['pay'])
var_pay_pah = statistics.variance(df_vehicle['pay_per_active_hour'])
gap_pay_pah = max(df_vehicle['pay_per_active_hour']) - min(df_vehicle['pay_per_active_hour'])

total_wait_time = df_vehicle['de_wait_time'].sum()/60
total_travelled_time = df_vehicle['time_travelled'].sum()/60
total_travelled_distance = df_vehicle['dist_travelled'].sum()/1000
total_active_time = df_vehicle['ACTIVE_MINUTE'].sum()

total_drivers = len(df_active_hours_sliced.DE_ID.unique())
total_xdt = sum(df_order['order_delivered_time']/60) - sum(df_order['sdt']/60)

order_delivered_within_sla = (df_order['order_delivered_time'] <= df_order['sla']).sum()
sla_violations = 100 - order_delivered_within_sla/data_sdt_all.reset_index().order_id.nunique() * 100

per_higher_3_xdt = 100 * ((df_order['order_delivered_time']/60 - df_order['sdt']/60) > 3).sum() / orders_assigned
per_higher_5_xdt = 100 * ((df_order['order_delivered_time']/60 - df_order['sdt']/60) > 5).sum() / orders_assigned
per_higher_10_xdt = 100 * ((df_order['order_delivered_time']/60 - df_order['sdt']/60) > 10).sum() / orders_assigned

total_pay_guarantee_mwo_dyn = df_vehicle['pay_guarantee_dyn'].sum()
total_pay_guarantee_foreach_mwo = [df_vehicle[f'pay_guarantee_{mwo}'].sum() for mwo in min_wage_options]


mwo = min_wage_options[1]
ppahd = pay_per_active_hour_data[pay_per_active_hour_data>0]
excess = (abs(ppahd - mwo/100) + (ppahd - mwo/100))/2

work_for_guarantee = ppahd - excess
gini_work_for_guaranteed_pay = gini(work_for_guarantee, 0)

pay_per_worked_hour = (mwo + excess)/ppahd
gini_pay_per_worked_hour = gini(pay_per_worked_hour, 0)

result = "algo," + str(algo) + "\n" 
result += "min_wage_per_second," + str(min_wage_per_second) + "\n" 
result += f"start_day,{st_day}\n"
result += f"end_day,{end_day}\n" 
result+= "orders_assigned," + str(orders_assigned) + "\n" 
# result+= "sla_violations," + str(sla_violations) + "\n" 
result+= "total_drivers," + f"{total_drivers:.0f}" + "\n" 
result+= "total_active_time," + f"{total_active_time:.0f}" + "\n" 
result+= "total_payment," + f"{total_payment:.0f}" + "\n" 
result+= f"total_payment_above_minwage_mwo_dyn," + f"{total_payment_above_minwage_dyn:.0f}" + "\n"
result+= f"total_payment_below_minwage_mwo_dyn," + f"{total_payment_below_minwage_dyn:.0f}" + "\n"
result+= f"total_pay_guarantee_mwo_dyn," + f"{total_pay_guarantee_mwo_dyn:.0f}" + "\n" 
result+= f"total_cost_to_platform_mwo_dyn," + f"{total_cost_to_platform_dyn:.0f}" + "\n" 
result+= f"num_drivers_below_minwage_mwo_dyn," + f"{num_drivers_below_minwage_dyn:.0f}" + "\n" 
result+= f"gini_normalized_mwo_dyn," + f"{gini_normalized_mwo_dyn:.3f}" + "\n" 

if(min_wage_options[0] == 0):
    mwo = min_wage_options[0]
    result+= f"gini_normalized_foreach_mwo{mwo}," + f"{gini_normalized_foreach_mwo[0]:.3f}" + "\n" 

for i, mwo in enumerate(min_wage_options):
    if (mwo==0):
        continue
    result+= f"total_payment_above_minwage_mwo{mwo}," + f"{total_payment_above_minwage_foreach_mwo[i]:.0f}" + "\n" 
    result+= f"total_payment_below_minwage_mwo{mwo}," + f"{total_payment_below_minwage_foreach_mwo[i]:.0f}" + "\n"
    result+= f"total_pay_guarantee_foreach_mwo{mwo}," + f"{total_pay_guarantee_foreach_mwo[i]:.0f}" + "\n" 
    result+= f"total_cost_to_platform_mwo{mwo}," + f"{total_cost_to_platform_foreach_mwo[i]:.0f}" + "\n" 
    result+= f"num_drivers_below_minwage_mwo{mwo}," + f"{num_drivers_below_minwage_foreach_mwo[i]:.0f}" + "\n" 
    result+= f"gini_normalized_foreach_mwo{mwo}," + f"{gini_normalized_foreach_mwo[i]:.3f}" + "\n" 

result+= "average_wait_time_per_driver," + f"{(total_wait_time/(total_drivers * 60)):.2f}" + "\n"
result+= "average_travelled_time_per_driver," + f"{(total_travelled_time/(total_drivers * 60)):.2f}" + "\n"
result+= "average_active_time_per_driver," + f"{(total_active_time/(total_drivers * 60)):.2f}" + "\n"
result+= "average_worked_fraction_per_driver," + f"{((total_travelled_time + total_wait_time)/total_active_time):.2f}" + "\n"
result+= "average_pay_per_driver," + f"{(total_payment/(total_drivers)):.2f}" + "\n"
result+= "average_delivery_time," + f"{(total_delivery_time/orders_assigned):.2f}" + "\n" 
result+= "average_xdt," + f"{(total_xdt/orders_assigned):.2f}" + "\n"
 
result+= "frac_drivers_with_non-zero_pay," + f"{1 - df_zero_pay_vehicles.DE_ID.nunique()/(total_drivers):.4f}" + "\n"  
result+= "per_higher_3_xdt," + f"{per_higher_3_xdt:.4f}" + "\n"
result+= "per_higher_5_xdt," + f"{per_higher_5_xdt:.4f}" + "\n"
result+= "per_higher_10_xdt," + f"{per_higher_10_xdt:.4f}" + "\n"
result+= "gini_work_for_guaranteed_pay," + f"{gini_work_for_guaranteed_pay:.4f}" + "\n"
result+= "gini_pay_per_worked_hour," + f"{gini_pay_per_worked_hour:.4f}" + "\n"
result+= "total_travelled_distance," + f"{total_travelled_distance}" + "\n"



analysis_result_file = base_dir+f"performance_analysis_{city}_{day}_{algo}{min_wage_per_second if algo == 'WORK4FOOD' else ''}.csv"
f= open(analysis_result_file,"w")
f.write(result)
f.close()

print(result)


