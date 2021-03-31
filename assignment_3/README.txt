Code Location
https://github.com/austinard1/MachineLearning/tree/master/assignment_2

0. Clone the above repository...
git clone https://github.com/austinard1/MachineLearning.git
cd MachineLearning/assignment_2/
git checkout master

1. Create conda environment

conda env create -f environment.yml

2. Activate conda environment

conda activate machine-learning

3. Install yellowbrick

pip install yellowbrick

4. Install packages

pip install -e .

5. Enter the unsupervised directory
cd unsupervised/

6. Perform the desired task using the instructions below.

optimization -t <task_option> -p <problem>

Task options...

tune_problem = Generate tuning plots for tuning optimization problem listed in <problem> arg
tuning_plots = Generate tuning plots for tuning optimization algorithms for a problem listed in <problem> arg
complexity_graph = Generate complexity analysis for problem listed in <problem> arg
performance_graph = Generate performance analysis for problem listed in <problem> arg


Problem options....

knapsack = Knapsack
four_peaks = Four Peaks
k_colors = Max K-Colors
neural_network = Neural Network weight optimization

NOTE: Neural network problem only has task options tuning_plots and performance_graph

Here is an example of generating the tuning plots for all algorithms for the four peaks problem

optimization -t tuning_plots -p four_peaks

Here is an example of generating the complexity analysis plot for the Max K-Colors problem

optimization -t complexity_graph -p k_colors
