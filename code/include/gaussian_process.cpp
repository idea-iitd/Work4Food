#include <bits/stdc++.h>
#include "global.hpp"
#include <omp.h>
#include "routeplan.hpp"
#include "route_recommendation.hpp"
#include "constants.hpp"
#include "gaussian_process.hpp"
#include <torch/script.h>

using namespace std;

// route_plan, plan_cost, delivery_times
typedef pair< pair<vector<event>, double>, unordered_map<string, double>> best_plan_tuple;


// constructor for before o
GP_input::GP_input(double global_time, vehicle &vh, long long int start_node){
 
    float curr_time =  ((((long long int)(global_time))%86400 + 86400)%86400);
    float vh_lat, vh_lon;
    if(start_node == -1){
        vh_lat = nodes_to_latlon[vh.path[vh.path_present_idx]].first;
        vh_lon = nodes_to_latlon[vh.path[vh.path_present_idx]].second;
    }
    else{
        vh_lat = nodes_to_latlon[start_node].first;
        vh_lon = nodes_to_latlon[start_node].second;
    }
    int curr_shift = vh.current_active_shift_AR(global_time);
    float start_time = vh.de_intervals[curr_shift].start_time;
    float end_time = vh.de_intervals[curr_shift].end_time;
    // cout<<"vh_lat,"<<vh_lat<<"vh_lon,"<<vh_lon<<endl;
    // cout<<"curr_shift,"<<curr_shift<<"start_time,"<<start_time<<"end_time"<<end_time<<endl;
    float elapsed_time = global_time - start_time;
    float remaining_time = end_time - global_time;
    float travel_time_till_now = vh.travel_time;
    float wait_time_till_now = vh.wait_time;
    float carrying_orders = (vh.order_set).size()*2 - (vh.route_plan).size();
    float event = (vh.route_plan.size() == 0)? 0 : (vh.route_plan)[0].type; // if vh is idle, its next event must be a pickup event (0)

    if(GP_NUM_FEATURES==11){
        float ndw = n_drivers_window;
        float now = n_orders_window;
        this->input = {curr_time, vh_lat, vh_lon, elapsed_time, remaining_time, travel_time_till_now, wait_time_till_now, carrying_orders, event, ndw, now};
    }
    else{
        this->input = {curr_time, vh_lat, vh_lon, elapsed_time, remaining_time, travel_time_till_now, wait_time_till_now, carrying_orders, event};
    }

    // Normalizing input features
    for(int i=0; i<this->input.size(); i++){
        this->input[i] -= gp_model_parameters.first[i]; // mean
        this->input[i] /= gp_model_parameters.second[i]; // std
    }

}

// constructor after o
GP_input::GP_input(double global_time, vehicle &vh, best_plan_tuple &best_plan){
    // Model inputs for wage prediction -> ['curr_time', 'vh_lat', 'vh_lon', 'elapsed_time', 'remaining_time', 'travel_time_till_now', 'wait_time_till_now', 'carrying_orders', 'event']

    vector<event> route_plan = best_plan.first.first;
    unordered_map<string, double> delivery_times = best_plan.second;
    order last_order = route_plan[route_plan.size()-1].order_obj;
    float last_order_end_time = delivery_times[last_order.order_id];
    string vh_o_lat_lon = last_order.customer.cust_latlon;

    float curr_time = ((((long long int)(last_order_end_time))%86400 + 86400)%86400);
    float vh_lat = stof(vh_o_lat_lon.substr(0, vh_o_lat_lon.find(",")));
    float vh_lon = stof(vh_o_lat_lon.substr(vh_o_lat_lon.find(",")+1, vh_o_lat_lon.size()));

    int curr_shift = vh.current_active_shift_AR(global_time);
    float start_time = vh.de_intervals[curr_shift].start_time;
    float end_time = vh.de_intervals[curr_shift].end_time;
    float elapsed_time = last_order_end_time - start_time;
    float remaining_time = end_time - last_order_end_time; //allow possibility of this being negative when the last order ends after the shift is over


    pair<double, double> travel_and_wait_time = get_travel_and_wait_times_AR(vh, route_plan, global_time);
    float travel_time_till_now = vh.travel_time + travel_and_wait_time.first;
    float wait_time_till_now = vh.wait_time + travel_and_wait_time.second;

    float carrying_orders = 0;
    for(int i=0; i<route_plan.size(); i++){
        carrying_orders += (route_plan[i].type == 0)?-1:1; // finds difference between # of delivery events and # of pickup events
    }
    float event = (route_plan.size() == 0)? 0 : (route_plan)[0].type; // if vh is idle, its next event must be a pickup event (0)


    if(GP_NUM_FEATURES==11){
        float ndw = n_drivers_window;
        float now = n_orders_window;
        this->input = {curr_time, vh_lat, vh_lon, elapsed_time, remaining_time, travel_time_till_now, wait_time_till_now, carrying_orders, event, ndw, now};
    }
    else
        this->input = {curr_time, vh_lat, vh_lon, elapsed_time, remaining_time, travel_time_till_now, wait_time_till_now, carrying_orders, event};

    // Normalizing input features
    for(int i=0; i<this->input.size(); i++){
        this->input[i] -= gp_model_parameters.first[i]; // mean
        this->input[i] /= gp_model_parameters.second[i]; // std
    }
}
