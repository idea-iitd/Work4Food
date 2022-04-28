#include <bits/stdc++.h>
#include "global.hpp"
#include "config.hpp"
#include "order.hpp"
#include "event.hpp"
#include "vehicle.hpp"
#include "graph_file_util.hpp"
#include "food_data_util.hpp"
#include "simulation_util.hpp"
#include "hhl_query.hpp"
#include "vehicle_assignment.hpp"
#include "vehicle_assignment_baseline.hpp"
#include "constants.hpp"
#include "mdrp_baseline.hpp"
#include <stdexcept>

using namespace std;

int ORDER_REJECTIONS = 0;

void simulation(){
    double global_time = start_time;
    vector<order> rejected_orders;
    int prev_time_slot = ((((long long int)(global_time))%86400 + 86400)%86400)/3600;
    load_labels(prev_time_slot);

    // the last index (GP_NUM_FEATURES) stores output mean and std
    vector<float> gp_model_mean(GP_NUM_FEATURES+1); 
    vector<float> gp_model_std(GP_NUM_FEATURES+1);
    
    // Work4Food main

    if(assignment_algorithm == "WORK4FOOD"){
        try {
            cout<< "LOADING MODEL "+gp_model_path+".pt"<<endl;
            gp_model = torch::jit::load(gp_model_path + ".pt");
            cout<< "DONE LOADING MODEL: "<<gp_model_path<<endl;
            cerr<< "DONE LOADING MODEL: "<<gp_model_path<<endl;

            cout<< "BEGINNING FILE READING "<<(gp_model_path + ".parameters")<<endl;
            ifstream gp_mean_std_input_file(gp_model_path + ".parameters");
            float fValue;
            for(int i=0; i<gp_model_mean.size(); i++){
                gp_mean_std_input_file >> gp_model_mean[i];
                // cout<<gp_model_mean[i]<<endl;
            }
            cout<< "DONE READING MEANS"<<endl;
            for(int i=0; i<gp_model_std.size(); i++){
                gp_mean_std_input_file >> gp_model_std[i];
                //  cout<<gp_model_std[i]<<endl;
            }
            cout<< "DONE READING VARIANCE"<<endl;
        }catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return;
        }
    }

    cout<<"PARAMETERS:"<<endl;
    for(int i=0; i<GP_NUM_FEATURES+1; i++){
        gp_model_parameters.first.push_back(gp_model_mean[i]);
        gp_model_parameters.second.push_back(gp_model_std[i]);
        cout<<gp_model_parameters.first[i]<<","<<gp_model_parameters.second[i]<<endl;
    }
    cout<<"GP_NUM_FEATURES: "<<GP_NUM_FEATURES<<endl;
    

    for(global_time = start_time; global_time < end_time + 4*3600 ; global_time += delta_time){
        int curr_time_slot = ((((long long int)(global_time))%86400 + 86400)%86400)/3600;
        // load new graph index when timeslot changes
        if (curr_time_slot != prev_time_slot){
            load_labels(curr_time_slot);
        }
        auto start1 = std::chrono::high_resolution_clock::now();
        // Change online offline status of vehicles
        int count_online = update_online_status(global_time);
        cout << fixed << "Current Time: " << (global_time)/3600.0 << endl;
        cout << "Number of Online Vehicles: " << count_online << endl;
        // get active orders to assign in current round
        vector<order> active_orders = get_delta_orders(global_time, rejected_orders, ORDER_REJECTIONS);
        // get active vehicles that can deliver current set of orders
        vector<int> active_vehicles = get_active_vehicle_indices(global_time, active_orders);
        // Perform Assignment
        if (assignment_algorithm == "FAIR_FOODY"){
            fair_foody(active_vehicles, active_orders, global_time, rejected_orders);
        }
        else if (assignment_algorithm == "FM_BR"){
            foodmatch_BR(active_vehicles, active_orders, global_time, rejected_orders);
        }
        else if (assignment_algorithm == "FM_BFS"){
            foodmatch_BFS(active_vehicles, active_orders, global_time, rejected_orders);
        }
        else if (assignment_algorithm == "FM_FULL" || assignment_algorithm == "FOOD_MATCH" ){
            foodmatch_FULL(active_vehicles, active_orders, global_time, rejected_orders);
        }
        else if (assignment_algorithm == "WORK4FOOD"){
            workforfood(active_vehicles, active_orders, global_time, rejected_orders);
        }
        else{
            throw invalid_argument("ASSIGNMENT ALGORITHM NOT RECOGNISED\n");
            return;
        }
        auto stop1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
        if (VERBOSITY == -1)
            cout << "full_time," << duration1.count() << endl;
        // Move vehicles
        for(int i = 0; i < int(all_vehicles.size()); i++){
            vehicle* vh = &all_vehicles[i];
            vh->move(global_time, global_time + delta_time);
        }
        prev_time_slot = curr_time_slot;
        unordered_map<long long int, double>::iterator it;
        if (time_opt.size() > 5e7){
            it = time_opt.begin();
            advance(it, time_opt.size()/2);
            time_opt.erase(time_opt.begin(), it);
        }
    }
    cout << ORDER_REJECTIONS << "/" << realtime_orders.size() << endl;
}

void usage() {
    cout << "Usage: ./main.o [-algo name] [-k num] [-cpf num] [-gamma num] [-eta num] [-day num] [-city num] [-delta num] [-start num] [-end num] [-gpr_model string] [-minwork fraction] [-guarantee_type string] [-reject_drivers]" << endl
              << "  -algo algo_name         \tAlgorithm name" << endl
              << "  -k num                  \tK value" << endl
              << "  -gamma num              \tGamma value" << endl
              << "  -eta num                \tEta value" << endl
              << "  -day num                \tDay number" << endl
              << "  -city name              \tCity name" << endl
              << "  -delta num              \tAccumulation window" << endl
              << "  -start num              \tStart time" << endl
              << "  -end num                \tEnd time" << endl
              << "  -de_frac                \tDriver fraction" << endl
              << "  -gpr_model              \tGPR model path" << endl
              << "  -minwork                \tFixed minimum work guarantee for all agents (g)" << endl
              << "  -guarantee_type         \tGuarantee type of W4F" << endl
              << "  -reject_drivers         \tEnable driver rejection if this label is used" << endl;
    exit(1);
}

int main(int argc, char *argv[]){
    int argi;
    for (argi = 1; argi < argc; ++argi) {
        if (argv[argi][0] == '-') {
            if (!strcmp("--", argv[argi])) { ++argi; break; }
            else if (!strcmp("-algo", argv[argi])) { if (++argi >= argc) usage(); assignment_algorithm = argv[argi];}
            else if (!strcmp("-k", argv[argi])) { if (++argi >= argc) usage(); vehicle_explore_frac = stoi(argv[argi]);}
            else if (!strcmp("-cpf", argv[argi])) { if (++argi >= argc) usage(); cluster_pct_frac = stod(argv[argi]);}
            else if (!strcmp("-gamma", argv[argi])) { if (++argi >= argc) usage(); heuristic_multiplier = stod(argv[argi]);}
            else if (!strcmp("-eta", argv[argi])) { if (++argi >= argc) usage(); max_merge_cost_edt = stoi(argv[argi]);}
            else if (!strcmp("-city", argv[argi])) { if (++argi >= argc) usage(); simulation_city = argv[argi];}
            else if (!strcmp("-day", argv[argi])) { if (++argi >= argc) usage(); simulation_day = argv[argi];}
            else if (!strcmp("-start", argv[argi])) { if (++argi >= argc) usage(); start_time = stod(argv[argi])*3600;}
            else if (!strcmp("-end", argv[argi])) { if (++argi >= argc) usage(); end_time = stod(argv[argi])*3600;}
            else if (!strcmp("-delta", argv[argi])) { if (++argi >= argc) usage(); delta_time = stod(argv[argi]);}
            else if (!strcmp("-de_frac", argv[argi])) { if (++argi >= argc) usage(); fraction_drivers = stod(argv[argi]);}
            // Work4Food main
            else if (!strcmp("-gpr_model", argv[argi])) { if (++argi >= argc) usage();  gp_model_path = argv[argi];}
            else if (!strcmp("-minwork", argv[argi])) { if (++argi >= argc) usage();  MIN_WAGE_PER_SECOND = stod(argv[argi]);}
            else if (!strcmp("-minwork_discount", argv[argi])) { if (++argi >= argc) usage();  MIN_WAGE_DISCOUNT_FACTOR = stod(argv[argi]);}
            else if (!strcmp("-guarantee_type", argv[argi])) { if (++argi >= argc) usage();  GUARANTEE_TYPE = argv[argi];}
            else if (!strcmp("-reject_drivers", argv[argi])) { REJECT_DRIVERS = true;}
        else usage();
        }
        else break;
    }
    if (argi != argc) usage();

    // Work4Food main
    if(REJECT_DRIVERS)
        GP_NUM_FEATURES = 11;

    string inp_conf = "../../results/" + simulation_city + "/config";
    cout<<"Reading input configuration"<<endl; //AR
    global_conf.read_from_file(inp_conf);
    cout<<"Reading graph input"<<endl; //AR
    read_graph_input();
    cout << "Finish taking graph input" << endl;
    read_food_data();
    cout << "Finish taking order input \nOrders = " << realtime_orders.size() << endl;
    vehicle_init();
    cout << "Finish initialising vehicle\nVehicles = " << all_vehicles.size() << endl;
    auto start = std::chrono::high_resolution_clock::now();
    simulation();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    cout << "Time taken by Simulation: " << duration.count()/60 << " minutes" << endl;
    return 0;
}
