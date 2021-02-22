1. Create conda environment

conda env create -f environment.yml

2. Activate conda environment

conda activate machine-learning

3. Install packages

pip install -e .

4. Enter the supervised directory
cd supervised/

5. Perform the desired task using the instructions below.

Each dataset has JSON file that contains the hyperpamaters used for each algorithm.
Each dataset has a dictionary for each learner, and inside those dictionaries are two nested dictionaries... "best" and "tuning".
The tasks described below determine which nested dictionary is used.

Learner mappings...

dt = decison trees
knn = k-nearest neighbors
boosting = boosting
neural = neural networks
svm = support machine vectors

Task options...

grid_search =  perform grid search looping over all the values listed in the current algorithm's "tuning" dictionary
mca = perform model complexity analysis. In the "tuning" dictionary, set one hyperparameter to a list of 1 value to keep it constant, and set another hyperparameter to a list of 2 values, which will be the range in which to vary it.
plot_best = plot the learner associated with the parameters in the current algorithm's "best" dictionary

There are separate entry points for each dataset, and the commands to run the code take this form...

<dataset_entrypoint> -l <learner_mapping> -t <task_option>

Here is an example of running the mca task of the neural network learner on the bank loan dataset

bank_loan -l neural -t mca

Here is an example of running the grid_search task of the decision tree learner on the smart grid dataset

smart_grid -l dt -t grid_search