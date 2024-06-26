# README  
This repository contains Python code for the Cost OptimisatioN Framework for Implementing blue-Green infrastructURE (CONFIGURE) version 1.0. The associate article titled **Multi-objective optimisation framework for Blue-Green Infrastructure placement using detailed flood model** is available at: https://doi.org/10.1016/j.jhydrol.2024.131571


CONFIGURE integrates a multiobjective optimisation algorithm (NSGA-II) with a hydrodynamic flood model to optimise the BGI intervention cost for the given decision variables i.e., the locations of BGI features. 
CONFIGURE can be run either in **Standard Mode** (configure10_standard.py) or **Parallelised Mode** (configure10_parallelised.py). The parallelised mode uses the multiprocessing capabilities of Python to run multiple processes (hydrodynamic simulations) simultaneously, thereby speeding up the optimisation process. Please refer to the main article to determine when to use the standard or parallelised mode.


Users can follow the instructions given inside the code to integrate their models and related inputs. 

  The following Python packages are required to run CONFIGURE:
  - Numpy 1.26.0  
  - Pandas 2.1.1  
  - Matplotlib 3.8.0

  After installing the above packages, users can test the functionality of CONFIGURE by directly running the script. Outputs will be saved in **C:\configure**

  ## Note
  Asid Ur Rehman developed this optimisation framework as part of his PhD study at Newcastle University. 
  
  ## Licence  
Asid Ur Rehman (C) 2023. Licenced under the terms of the Apache License 2.0.
