#pragma once

#include <bits/stdc++.h>
#include "event.hpp"

using namespace std;

// route_plan, plan_cost, delivery_times
typedef pair< pair<vector<event>, double>, unordered_map<string, double>> best_plan_tuple;


class GP_input{
	public:
	vector<float> input;

	// constructor for before o
	GP_input(double global_time, vehicle &vh, long long int start_node=-1);

    // constructor after o
	GP_input(double global_time, vehicle &vh, best_plan_tuple &best_plan);
};