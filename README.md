# SSBJ-OpenMDAO
[SSBJ test case](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf) solved by MDF, IDF, CO and BLISS 2000 MDO formulations with OpenMDAO 3.x.

# Installation
* Python>=3.6
* Install [OpenMDAO>=3.3](https://github.com/OpenMDAO/OpenMDAO) 
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
