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

3. Install packages

pip install -e .

4. Clone mlrose-hiive and checkout tag 2.1.7

git clone https://github.com/hiive/mlrose.git
cd mlrose
git checkout 2.1.7
pip install -e .

4. Enter the optimization directory
cd optimization/

5. Perform the desired task using the instructions below.

optimization -t <task_option> -o <optimizer> -p <problem>

NOTE: You must pass in all optimizers for all tasks, otherwise plots won't generate correctly

Task options...

tune_problem = Generate tuning plots for tuning optimization problem listed in <problem> arg
tuning_plots = Generate tuning plots for tuning optimization algorithms listed in the <optimizer> arg for a problem listed in <problem> arg
complexity_graph = Generate complexity analysis for problem listed in <problem> arg
performance_graph = Generate performance analysis for problem listed in <problem> arg

Optimizer options....
rhc = Random Hill Climb
sa = Simulated Annealing
ga = Genetic Algorithm
mimic = MIMIC

Problem options....
knapsack = knapsack
four_peaks = Four Peaks
k_colors = Max K-Colors


Here is an example of generating the tuning plots for all algorithms for the four peaks problem

optimization -t tuning_plots -o rhc sa ga mimic -p four_peaks

Here is an example of generating the complexity analysis plot for the Max K-Colors problem

optimization -t complexity_graph -p rhc sa ga mimic -p k_colors

NOTE: You must pass in all optimizers for all tasks, otherwise plots won't generate correctly