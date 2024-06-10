# Adaptive Constraint-Guided Surrogate Enhanced Evolutionary Algorithm for Horizontal Well Placement Optimization in Oil Reservoir _ Computers and Geosciences

## Authors
> Qinyang Dai, Liming Zhang*, Peng Wang, Kai Zhang, Guodong Chen, Zhangxing Chen, et al.
> Submitted to Computers and Geosciences, 2024.

## Description
This repository contains the code associated with the manuscript titled “Adaptive Constraint-Guided Surrogate Enhanced Evolutionary Algorithm for Horizontal Well Placement Optimization in Oil Reservoir”. The study developed a program based on Python 3.9.13 to optimize the horizontal well placements in oil reservoir using ACG-EBS algorithm.

## Usage
- Ensure you have Python installed.
- Clone this repository to your local machine.
- Establish a reservoir numerical simulation model based on a reservoir numerical simulator (e.g., Eclipse). A model that can be used for optimization has been uploaded in the 'model_data' folder. Users should copy this model to the 'Iterm' folder and modify the relevant model parameters and program running parameters in the 'main_ACGEBS.py' file.
- Run the program using python 'main_ACGEBS.py'.
- During the run, some intermediate evaluation models generated by the algorithm will be temporarily stored in the 'RunIterm' folder. The final model containing the optimal well placement scheme and results will be stored in the 'Result' folder.

## Contact
If you have any questions or require further information, please do not hesitate to contact us at dai_qinyang@163.com.
