## Installation

1. Obtain the anonymized data from "https://bit.ly/3HGdavB" and keep the data according to the following directory structure:

```
.
├─data
├───data_A_anonymized
├─────food_data
├─────map
├───data_B_anonymized
├─────food_data
├─────map
├───data_C_anonymized
├─────food_data
├─────map
├─results
├─scripts
├─gpr_training
├─gpr_models
├─analysis_scripts
├─analysis_results
├─include
├─main.cpp
├─pytorch-cpp
├───build
├───CMakeLists.txt
```

2. Generate the index structures for road networks:
   We use the Hierarchical Hub Labeling code from the following repository [savrus/hl](https://github.com/savrus/hl).  

```bash
    bash scripts/generate_index.sh
```

3. Compile the project with CMake to link pytorch with C++, a Makefile will get created at ```pytorch-cpp/build/```:

    Refer: https://pytorch.org/cppdocs/installing.html#:~:text=mkdir%20build%0Acd%20build%0Acmake%20%2DDCMAKE_PREFIX_PATH%3D/absolute/path/to/libtorch%20..%0Acmake%20%2D%2Dbuild%20.%20%2D%2Dconfig%20Release
    or 
    https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5

```bash
    cd pytorch-cpp/build/
    pip install torch
    cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
    make 
```

4. Install GPyTorch

```bash
    pip install gpytorch
```

## Running the code

1. Simulation:  

   - Usage:
   ```bash
       cd pytorch-cpp/build/
       ./main [-algo algorithm_name] [-city city_name] [-day day_num] [-eta eta_num] [-k k_num] [-gamma gamma_num] [-delta delta_num] [-start start_num] [-end end_num] [-de_frac fraction_agents] [-gpr_model gpr_model_path] [-minwork minwork_guarantee] [-guarantee_type guarantee_type_name] [-reject_drivers]
   ```

   - The parameters are explained below:
     - **algorithm_name**: Name of assigment algorithm to use \[Default: WORK4FOOD\]  
     ```
     'WORK4FOOD' = Work4Food implementation.
     'FOOD_MATCH' = FoodMatch implimentation proposed by Jhoshi et. al. [1], we have reused code provided by [3] in our implementation.
     'FAIR_FOODY' = FairFoody implementation proposed by Anjali et. al. [2], we have reused code provided by [4] in our implementation.
     ```
     - **city_name**: Name of city to run algorithm on \[Default: A\]  
     ```
     '{A, B, C}'
     ```
     - **day_num**: Day on which the algorithm should be run  \[Default: 1\]
     ```
     '{1,2,3,4,5,6}'
     ```
     - **eta_num**: Eta value used in stopping criteria for clustering \[Default: 60\]
     - **k_num**: K value used in fraction of edges to use in BFS \[Default: 200\]
     - **gamma_num**: Gamma value used in weight of heuristic function \[Default: 0.5\]
     - **delta_num**: Accumulation window used \[Default: 180\]
     - **start_num**: Start hour of simulation in 24-hour format \[Default: 0\]
     - **end_num**: End hour of simulation in 24-hour format \[Default: 24\]
     - **fraction_agents**: The fraction of agents from the dataset to be used as available agents \[Default: 1.0\]
     - **gpr_model_path**: Path to the GPR model to be used w/o the .pth or .pt extension \[Default: ""\]
     - **minwork_guarantee**: Minimum work guarantee in terms of work guarantee ratio between 0 and 1 \[Default: 0.25\]
     - **guarantee_type_name**: Type of work guarantee 'static'(same for all agents), 'dynamic_gp'(dynamic based on GPR) or 'rating'(based on agent ratings) \[Default: static\]
     - **-reject_drivers**: If this flag used then driver rejection is set on \[Default: false\]
     
   - Sample command:
   ```bash
       cd pytorch-cpp/build/
       ./main -algo FAIR_FOODY -city A -day 1 -eta 60 -k 200 -gamma 0.5 -delta 180 -start 0 -end 24 > ../../results/A/1/FAIR_FOODY_0_24.results 
       ./main -algo WORK4FOOD -city A -day 1 -eta 60 -k 200 -gamma 0.5 -delta 180 -start 0 -end 24 -gpr_model ../../gpr_models/model_A_days_2_and_5_25_static_pay_2 > ../../results/A/1/WORK4FOOD_0_24_25_static
   ```
   Note that the logs should be saved in file  ```../../results/{city_name}/{day_num}/{algorithm_name}_{start_num}_{end_num}{work4food_file_suffix}``` by convention so that GPR training and evaluation work. The options for **work4food_file_suffix** are: 
   ```
     '{_25_static, _25_static_frac{int(100*fraction_agents)}, _25_static_driverreject, _dynamic, _dynamic_driverreject, _rating}'
   ```

2. GPR Training: 
  
  - Training Data: Day 2 and day 5 are training days. Run WORK4FOOD on these days with 100,90,80,70 and 60 fraction of agents available.
   Example:
   ```bash
   cd pytorch-cpp/build/
   ./main -algo WORK4FOOD -city A -day 2 -eta 60 -k 200 -gamma 0.5 -delta 180 -start 0 -end 24 -de_frac 0.9 -gpr_model ../../gpr_models/model_A_days_2_and_5_25_static_pay_2 > ../../results/A/2/WORK4FOOD_0_24_25_static_frac90.results
   ``` 

   - Usage : 
   ```bash
   python3 gpr_training/work_predict.py [city_name]  
   ``` 

    - **city_name**: Name of city whose GPR model is to be learnt \[Required\]
      ```
      '{A, B, C}'
      ```
    
    - Sample command:
      ```bash
      python3 gpr_training/work_predict.py A
      ```

3. Evaluation:  

   - Usage: 
   ```bash
   python3 analysis_scripts/performance_analysis.py [city_name] [day_num] [algorithm_name] [work4food_file_suffix] 
   python3 analysis_scripts/overflow_analysis.py [city_name] [day_num] [algorithm_name] [work4food_file_suffix] 
   ```

   The results get saved in `analysis_results` directory.

   - The parameters are explained below:
     - **city_name**: Name of city \[Required\]
     ```
     '{A, B, C}'
     ```
     - **day_num**: Day on which the algorithm was run  \[Required\]
     ```
     '{1,2,3,4,5,6}'
     ```
     - **algorithm_name**: Name of assigment algorithm to evaluate  \[Required\]
     ```
     'WORK4FOOD' = Work4Food implementation.
     'FOOD_MATCH' = FoodMatch implimentation proposed by Jhoshi et. al. [1]
     'FAIR_FOODY' = FairFoody implementation proposed by Anjali et. al. [2]
     ```
     - **work4food_file_suffix**: file suffix for different WORK4FOOD variation results  \[Required only for WORK4FOOD\]
     ```
     '{_25_static, _25_static_frac{int(100*fraction_agents)}, _25_static_driverreject, _dynamic, _dynamic_driverreject, _rating}'
     ```
  
   - Sample command:

    ```bash
        python3 analysis_scripts/performance_analysis.py A 1 FAIR_FOODY
        python3 analysis_scripts/overflow_analysis.py A 2 WORK4FOOD _25_static 
    ```

## Code Description

- `main.cpp` - Contains the code that drives the simulation.<br>
  It produces a log of simulation on stdout. This is redirected to a file for GPR training and evaluation.
- `include/` - Contains code for the simulation framework and algorithms.
- `include/constants.cpp` - Contains the default parameters used.
- `include/vehicle_assignment.cpp` - Contains the code for Work4Food,FoodMatch and FairFoody algorithms.
- `gpr_training/work_predict.py` - Contains the GPR training code used in our paper.
- `analysis_scripts/performance_analysis.py` - Reads the simulation log and reports performance metrics defined in our paper.
- `analysis_scripts/overflow_analysis.py` - Reads the simulation log and reports overflow metrics defined in our paper.

## References

[1] M.  Joshi,  A.  Singh,  S.  Ranu,  A.  Bagchi,  P.  Karia,  and  P.  Kala,“Batching and matching for food delivery in dynamic road networks,”inProc. ICDE, 2021.<br>
[2] Anjali and Rahul Yadav and Ashish Nair and Abhijnan Chkraborty and Sayan Ranu and Amitabha Bagchi, "FairFoody: Bringing in Fairness in Food Delivery," inProc. AAAI, 2022.<br>
[3] https://github.com/idea-iitd/FoodMatch<br>
[4] https://github.com/idea-iitd/fairfoody<br>