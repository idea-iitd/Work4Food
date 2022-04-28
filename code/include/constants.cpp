#include <bits/stdc++.h>
#include "constants.hpp"

using namespace std;

const double MAX_NUM = 1e7;
const double FP_EPSILON = 0.0000001;
int VERBOSITY = -1;
double vehicle_rest_radius_cap = 1000.0;//2700.0
double rejection_window = 1800;
bool use_sla_constraint = false;
double max_cost_val = 7200;
double fraction_drivers = 1.0;
string assignment_algorithm = "WORK4FOOD";
int batching_cap = 3;
double max_merge_cost_edt = 60;
int vehicle_explore_frac = 200;
int pct_explore_frac = 150;
// cluster pct
double cluster_pct_frac = 0.8;
double vehicle_explore_rad = 1000;
double heuristic_multiplier = 0.5;
string simulation_day = "1";
string simulation_city = "A";
double delta_time = 180;
double start_time = 0*3600;
double end_time = 24*3600;


// Work4Food Constants

string gp_model_path = "";
int GP_NUM_FEATURES = 11; // Model inputs for wage prediction -> ['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event']
                        // chnged to 11 in main id REJECT_DRIVERS, extra columns: 'n_drivers_window' and 'n_orders_window'
double MIN_WAGE_PER_SECOND = 0.25;
double MIN_WAGE_DISCOUNT_FACTOR = 0;
bool USE_WAIT_TIME = true;
string GUARANTEE_TYPE = "static";
bool REJECT_DRIVERS = false; // driver rejection based on minwage
vector<double> hourly_guarantee(24);
float work_ratio_wrt_ff = 1;
float XDT_OPTIMIZATION_WEIGHT = 1;
// Deserialize the ScriptModule from a file using torch::jit::load().
torch::jit::script::Module gp_model;
pair<vector<float>, vector<float>> gp_model_parameters;