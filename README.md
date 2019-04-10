# BayesianOptTools
The Bayesian Optimization Tools is a Python program developed by Aerodynamics Research Group from Institut Teknologi Bandung (ITB) that contains a collection of Bayesian Optimization tools including various surrogate modeling methods, sampling techniques, and optimization methods.
Currently, the program is under development and not yet implemented as a module for Python 3. Also, the coverage of the program are still limited to:

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

# Usage
The demo codes are available in the main folder.

# Contact
The original program was written by Pramudita Satria Palar, Kemas Zakaria, Ghifari Adam Faza, and is maintained by Aerodynamics Research Group ITB. 
