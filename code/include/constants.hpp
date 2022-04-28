#include <bits/stdc++.h>
#include <torch/script.h>

using namespace std;

/* Constants */
// Maximum number in simulation
extern const double MAX_NUM;
// Floating point epsilon in simulation
extern const double FP_EPSILON;
// Set to -1 to generate output for evaluation
extern int VERBOSITY;
// 45 minute bound
extern double vehicle_rest_radius_cap;
// Time before rejection
extern double rejection_window;
extern bool use_sla_constraint;
// \Omega (penalty) in seconds
extern double max_cost_val;
// Fraction of drivers to use
extern double fraction_drivers;
// Algorithm Name
extern string assignment_algorithm;
// MAXO
extern int batching_cap;

/* FoodMatch parameters */
// eta
extern double max_merge_cost_edt;
// K
extern int vehicle_explore_frac;
extern int pct_explore_frac;
// cluster pct
extern double cluster_pct_frac;
// BFS limit
extern double vehicle_explore_rad;
// Gamma
extern double heuristic_multiplier;

/* Simulation */
// Day of simulation
extern string simulation_day;
// City of simulation
extern string simulation_city;
// Accumulation Window
extern double delta_time;
// Start time of simulation in seconds
extern double start_time;
// End time of simulation in seconds
extern double end_time;


// Work4Food Constants

extern string gp_model_path;
// size of tensor passed in GP model
extern int GP_NUM_FEATURES;
// Minimum guaranteed wage 
extern double MIN_WAGE_PER_SECOND;
extern double MIN_WAGE_DISCOUNT_FACTOR;
extern bool USE_WAIT_TIME;
extern string GUARANTEE_TYPE; // valid values = static, dynamic, dynamic_lower_bound, dynamic_gp
extern bool REJECT_DRIVERS; // driver rejection based on minwage
// associated array that stores hourly guarantee is hourly_guarantee 
extern vector<double> hourly_guarantee;
extern float work_ratio_wrt_ff;
extern float XDT_OPTIMIZATION_WEIGHT;
// Deserialize the ScriptModule from a file using torch::jit::load().
extern torch::jit::script::Module gp_model;
extern pair<vector<float>, vector<float>> gp_model_parameters;
