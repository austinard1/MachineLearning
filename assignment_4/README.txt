Code Location
https://github.com/austinard1/MachineLearning/tree/master/assignment_4

0. Clone the above repository...
git clone https://github.com/austinard1/MachineLearning.git
cd MachineLearning/assignment_4/
git checkout master

1. Create conda environment

conda env create -f environment.yml

2. Activate conda environment

conda activate machine-learning

3. Install OpenAI gym

git clone https://github.com/openai/gym.git
cd gym
pip install -e .
cd ..

4. Install MDPToolbox (hiive)

pip install mdptoolbox-hiive
pip install pymdptoolbox

4. Install packages

pip install -e .

5. Enter the unsupervised directory
cd reinforcement/

6. Perform the desired task using the instructions below.

reinforcement -c <problem> -t <task option>

Problem options...

frozen_lake = Generate plots/results for the Frozen Lake problem
forest = Generate plots/results for the Forest Management problem


Task options...

tuning_plots = Generate tuning plots for problem listed in <problem> arg
complexity_graph = Generate complexity analysis for problem listed in <problem> arg
performance_graph = Generate performance analysis for problem listed in <problem>
policy_plots = Generate optimal policy visualizations for problem listed in <problem>

Here is an example of generating the tuning plots for all algorithms for the Frozen Lake problem

reinforcement -t tuning_plots -p frozen_lake

Here is an example of generating the complexity analysis plot for the Forest Management problem

reinforcement -t complexity_graph -p forest