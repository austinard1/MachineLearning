Code Location
https://github.com/austinard1/MachineLearning/tree/master/assignment_3

0. Clone the above repository...
git clone https://github.com/austinard1/MachineLearning.git
cd MachineLearning/assignment_3/
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

unsupervised -c <clustering option> -dr <dimensionality reduction option> -nn (flag to perform NN analysis)

Clustering options...

k_means = Generate plots/results for K-Means clustering
em = Generate plots/results for EM clustering


Dimensionality Reduction options...

pca = Generate plots/results for Principal Component Analysis
ica = Generate plots/results for Independent Component Analysis
rp = Generate plots/results for Random Projection
fa = Generate plots/results for Factor Analysis

NOTE: Combining both a clustering and DR option will show results for the combination (see examples below)

Neural Network flag...

The -nn flag will generate neural network results for all algorithms in the class of the other input argument (either cluster or DR)
This means that for clustering neural network results, any clustering option can be passed in alongside the -nn flag (see examples below)


Here is an example of generating the results for just K-means

unsupervised -c k_means

Here is an example of generating the results for just Random Projection

unsupervised -dr rp

Here is an example of generating the combination results for ICA followed by EM

unsupervised -dr ica -c em

Here is an example of generating the neural network results for all dimensionality reduction algorithms

unsupervised -dr pca -nn

Here is an example of generating the neural network results for all clustering algorithms

unsupervised -c em -nn