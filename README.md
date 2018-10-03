# SSBJ-OpenMDAO
SSBJ test case solved by MDF, IDF and CO MDO formulations with OpenMDAO 2.x.

# Installation
* Python 2.7
* Install [OpenMDAO>=2.4](https://github.com/OpenMDAO/OpenMDAO) 
* Install [pyOptSparse](https://github.com/mdolab/pyoptsparse)
* Clone the project

# Usage 
## MultiDisciplinary Feasible
``` sh
python ssbj_mdf.py [--plot]
```
## Individual Discipline Feasible
``` sh
python ssbj_idf.py [--plot]
```
## Collaborative Optimization
``` sh
python ssbj_co.py
```
## Bi-Level Integrated System Synthesis 2000
``` sh
python ssbj_bliss2000.py
```
