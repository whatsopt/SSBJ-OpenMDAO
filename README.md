# SSBJ-OpenMDAO
SSBJ test case solved by MDF, IDF and CO MDO formulations with OpenMDAO 2.x.

# Installation
* Install [OpenMDAO 2.x](https://github.com/OpenMDAO/OpenMDAO) 
* Install [pyOptSparse](https://github.com/mdolab/pyoptsparse)
* Clone the project

# Usage 
## Multidisciplinary Feasible
``` sh
python ssbj_mdf.py [--plot]
```
## Individual Discipline Feasible
``` sh
python ssbj_idf.py [--plot]
```
## Collaborative Optimization (OpenMDAO >= 2.3)
``` sh
python ssbj_co.py
```
