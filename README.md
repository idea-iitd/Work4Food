# Gigs with Guarantees: Achieving Fair Wage for Food Delivery Workers

This repository contains official implementation of the algorithms defined in our paper.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The algorithms are implemented in C++11 (GCC 7.4.0), and GPR training and evaluation scripts are implemented in Python 3 (Python 3.6.9)

### Installation

Setup a conda environment which includes packages required to run evaluation scripts:

```bash
conda env create -f environment.yml
conda activate w4f_evn
```

### Datasets and evaluation procedure

The code for simulation and algorithms defined in our paper is provided in [./code]. An anonymized version of the proprietary dataset is available at "https://drive.google.com/drive/folders/17e6ZXzD8e9lAdnBEL8aT09MLJDrv1Yh8?usp=sharing" or "https://bit.ly/3HGdavB".


## References

[1] M.  Joshi,  A.  Singh,  S.  Ranu,  A.  Bagchi,  P.  Karia,  and  P.  Kala,“Batching and matching for food delivery in dynamic road networks,”inProc. ICDE, 2021.<br>
[2] Anjali and Rahul Yadav and Ashish Nair and Abhijnan Chkraborty and Sayan Ranu and Amitabha Bagchi, "FairFoody: Bringing in Fairness in Food Delivery," inProc. AAAI, 2022.<br>
[3] https://github.com/idea-iitd/FoodMatch<br>
[4] https://github.com/idea-iitd/fairfoody<br>