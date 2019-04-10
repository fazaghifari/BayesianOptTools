# BayesianOptTools
The Bayesian Optimization Tools is a Python program developed by Aerodynamics Research Group from Institut Teknologi Bandung (ITB) that contains collections of Bayesian Optimization tools including various surrogate modeling methods, sampling techniques, and optimization methods.
Currently, the program is under development and not implemented yet as a module for Python 3. Also, the coverage of the program are still limited to:

* Kriging
  * Ordinary Kriging
  * Regression Kriging
  * Polynomial Chaos Kriging
  * Composite Kernel Kriging
  * Kriging with Partial Least Square
* Bayesian Optimization
  * Unconstrained Single-Objective Bayesian Optimization (Expected Improvement)
  * Unconstrained Multi-Objective Bayesian Optimization (ParEGO, EHVI)
* Test Cases
  * Branin (Single-Objective)
  * Sasena (Single-Objective)
  * Styblinski-Tang (Single-Objective)
  * Schaffer (Multi-Objective)
  
# Required packages
BayesianOptTools depends on these modules: numpy, scipy, sk-learn and CMA-ES.

# Quick Examples
The demo codes are available in the main folder. 
KrigDemo.py is a demo script for creating surrogate model for Branin test function.
uncMOBOdemo.py is a demo script for performing unconstrained multi-objective optimization for Schaffer test function.
uncSOBOdemo.py is a demo script for performing unconstrained single objective optimization for Branin test function.

# Contact
The original program was written by Pramudita Satria Palar, Kemas Zakaria, Ghifari Adam Faza, and is maintained by Aerodynamics Research Group ITB. 
