# Machine Learning: CS 7641 OMS
# Assignment 3 - Unsupervised Learning and Dimensionality Reduction

Name: Austin Ard  &nbsp;&nbsp;&nbsp;&nbsp;      GT ID: aard6

<!---
To generate a PDF:
pandoc report.md -s -o report.pdf -V fontsize=11pt
-->

## Introduction
This report used the sklearn Python module to analyze the performance of 2 unsupervised clustering algorithms (K-Means Clusters and Expectation Maximization) and 4 dimensionality reduction algorithms (Principal Component Analysis, Independent Component Analysis, Random Projection, and Factor Analysis). These algorithms were be analyzed independently as well as in tandem with all possible combinations of clustering and dimensionality reduction algorithms. This report used the same two datasets analyzed in assignment1: a smart grid stability dataset and a bank loan qualification dataset. These datasets were chosen due to their contrasting nature, where the smart grid dataset has all continuous, well balanced, and evenly distributed features, and the bank loan dataset contains mostly imbalanced, discrete features with a ton of outliers. These algorithms were also run prior to feeding the dataset into the same neural network from assignemnt 1 for comparison.

## Clustering

The first part of this report explored the 2 different clustering algorithms discussed in the introduction, as well as illustrated the process used for selecting the optimal number of clusters for each.

### K-Means Clustering

K-Means clustering assigns samples to spherical clusters based on a similarity metric. For simplicity's sake, we used the euclidean distance metric for this experiment. Three metrics to evaluate these clusters were the silhouette score, Calinksi-Harabasz score, and the Davies-Bouldin score. The silhouette score is a metric calculated for each sample using its mean nearest cluster-distance and mean intra-cluster distance, where a higher score indicates that the sample is closer to its assigned cluster and farther away from the other clusters. The Calinksi-Harabasz score is a metric that highlights the ratio between how spread out points are within each cluster and how spread out clusters are from each other, so a higher score indicates that the clusters are tigher together and further apart from each other. The Davies-Bouldin score is a metric that highlights the ratio of intracluster scattering and intercluster separation, so a lower score indicates that the clusters are tighter together and further apart from each other. This analysis only concerns itself with relative scores between models, so the CH and DB scores were scaled down to fit on the same plot with the silhouette score for visual analysis. The results below illustrate the selection process for the number of clusters (K) aned the validation of those clusters against a base boosting learner. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.

| Selection Method | # of Clusters | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Base Learner | | 2.002 | 0.846 | | 0.397 | 0.499 |
| Best Scores | 2 | 1.509 | 0.996 | 3 | 0.408 | 0.920 |

![](unsupervised/plots/k_means/cluster_analysis.png)

When choosing the best combination of metric scores, we chose a best number of clusters of 2 for the smart grid dataset and 3 for the bank loan dataset, which were small compared to the range of cluster quantities explored. This indicates that the metrics we used to choose the number of clusters naturally favor a smaller number of clusters because the inter-cluster distances should get larger with less clusters. According to the tSNE projections, we see the two clusters formed for the smartgrid dataset are equally sized and closely resemble two semi-spheres. The smart grid dataset has evenly distributed and balanced features, so the k-means algorithm is able to easily and effectively cluster the points into 2 clusters by splitting the dataset down the middle somewhere. However, the clusters are close together and appear to contact each other  because the features have continuous values that make it harder for the clustering algorithm to assign points near the midpoints of those feature ranges. The clusters for the bank loan dataset are not evenly sized due to the imbalanced features with many outliers. There is also visible spaces between some of the clusters as a result of this dataset containing more discrete features. The clustering increased cross-validation accuracy significantly for the boosting learners for both datasets. Since this method is a form of pre-processing before passing into the boosting learner, it essentially boosts the booster. The fit time for the base learner also decreased for the larger smart grid dataset, but the fit time remained relatively the same for the smaller bank loan dataset. This highlights the increased effect of clustering on reducing runtime in correlation with dataset size, meaning there is more value when performing K-means on datasets of increasing size.

### Expectation Maximization (EM)

The EM algorithm assigns samples to clusters assuming that the clusters take the form of Gaussian distributions instead of spherical clusters like K-means. The two metrics used to evaluate these clusters were the Bayesian Information Criterion (BIC) and Akaike Information Criteria (AIC). Both scores are driven down by increasing maximum likelihood estimates, with a lower score indicating a better fit. The main difference between the two is that BIC puts a higher penalty on more complex models. When validating the number of components for EM against a base boosting learner, low cross-validation accuracies were output when choosing the minimum point of both AIC and BIC scores. These values usually indicated that number of components should be high. However, this was resulting in a model overfitting to the training data. To supplement these findings with a new selection method, using an elbow method of choosing the point of maximum curvatures for the metric curves was also explored. Below is a table and plot with results from this experiment. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.

| Selection Method | # of Clusters | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Base Learner | | 2.140 | 0.846 | | 0.397 | 0.499 |
| Lowest | 3 | 1.363 | 0.879 | 28 | 0.676 | 0.260 |
| Elbow | 3 | 1.363 | 0.879 | 6 | 0.484 | 0.799 |

![](unsupervised/plots/EM/components_analysis.png)

When validating against a base classifier, using the elbow method resulted in far better cross-validation accuracy as it chose a less complex model that didn't overfit to the data. As a result, we chose the best number of clusters as 3 for the smartgrid dataset and 6 for the bankloan dataset. Interestingly, both selection methods resulted in the same number of components for the smart grid dataset. In fact, the metric curve plot above shows that BIC increases with number of components rather than decreasing alongside AIC like with the bank loan dataset. The key reason for this is the smart grid dataset is the larger of the two and the fact that BIC penalizes more complex models more so than AIC does. Both scores have a constant value added to the maximum likelihood estimate portion of the score formula, however the BIC score calculates this score with respect to the number of samples while AIC does not. Since the smart grid dataset has twice as many samples as the bank loan dataset, it gets increasingly more penalized via this constant term as number of components increases. From this observation, we see the affect that dataset size has on component number selection and that using the base method of selecting the lowest metric scores loses credibility with increasing dataset size. This experiment also suggests that selecting the point of maximum curvatures is a more consistent method for choosing a good solution with varying dataset size. According to the tSNE projections and similar to the k-means results, the clusters for the smartgrid dataset are close together and relatively even in size as a result of the well balanced and distributed continuous features. Also similar to the k-means results, the clusters for the bankloan dataset are unevenly sized and more spread apart due to its imbalanced features and discrete features. Furthermore, some of the clusters appear not well grouped, indicating that the EM algorithm struggles with non-Gaussian datasets with many outliers. The fit time for the base learner significantly decreased for the larger smart grid dataset after performing EM, whereas the fit time remained relatively the same for the smaller bank loan dataset. This highlights the increased effect of clustering on reducing runtime in correlation with dataset size as well, meaning there is more value when performing clustering method like EM on datasets of increasing size.

## Dimensionality Reduction

The next part of this report explored the 4 different dimensionality reduction algorithms discussed in the introduction, as well as illustrates the process used for selecting the optimal number of reduced features for each.

### Principal Component Analysis (PCA)

For PCA component selection, we looked at the variances of each new component as well as their sums. This is a good measure of PCA's performance because PCA aims to find components by maximizing their variance, which should result in features that provide more value to the learning algorithm. To analyze the new components generated by PCA versus the original features, we looked at the variance of each component/feature in relation to each other. The plots below show the results of these experiments. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.

| DR Algorithm | # of Features | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| None | 12 | 1.923 | 0.846 | 13 | 0.526 | 0.499 |
| PCA | 11 | 1.642 | 0.836 | 9 | 0.567 | 0.445 |

![](unsupervised/plots/PCA/variance_analysis.png)

As the number of components decreases, the sum of variance deacreases for both datasets, which makes sense because variance in the overall data will decrease with less datapoints. This decrease is close to linear for the smart grid dataset, indicating that the features all have similar variances and are well distributed internally. The bank data set has more of a curve to it, inidicating that there are a few features which provide very little variance, which is why PCA is able to eliminate more components while maintaining good overall variance. The feature/PCA component bar graphs show that while projecting to a new space, the variance increases for some of the new PCA components as expected considering it tries to maximize variance among components. It also illustrates why PCA is able to remove certain original features due to low variances wile mainting a good overall variance as shown by the max PCA component variance being higher than the highest original feature variance.

### Independent Component Analysis (ICA)

For ICA component selection, we looke at the kurtosis values for each new component as well as their mean. This is a good measure of ICA's performance because a higher kurtosis value would indicate that the components are statistically independent, and ICA tries to find components by maximizing independency. To analyze the new components generated by ICA versus the original features, we look at the kurtosis of each component/feature in relation to each other. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.


| DR Algorithm | # of Features | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| None | 12 | 1.923 | 0.846 | 13 | 0.526 | 0.499 |
| ICA | 11 | 1.676 | 0.849 | 4 | 0.316 | 0.436 |

![](unsupervised/plots/ICA/kurtosis_analysis.png)

The smart grid dataset shows original features with relatively equal kurtoses, which is expected considering it mostly contains well balanced and evenly distributed features. In this case, ICA notices the one original feature with less kurtosis than the rest and projects the data to a space in which that kurtosis gets distributed to the other components, which also results in that component ultimately getting eliminated. We can infer from this that ICA would not have large impacts on datasets that contain balanced and well behaved features. For the bank loan dataset which contains features with high kurtosis as well as one feature that has significantly higher kurtosis than all the others, the ICA algorithm prefers to reduce a much larger number of components. In other words, ICA chooses to let the high kurtosis feature remain the same and lets it dominate the overall kurtosis of the model while also removing spme of the smaller kurtosis components. This process proves effective, as the cross-validation score of the base boosting learner does not diminish by much in the process while still managing to reduce runtime.

### Random Projection (RP)

RP is similar to PCA, however RP selects the new component axes randomly as opposed to trying to maximize variance. For RP component selection, we looked at the reconstruction error values for each new component. This is a good measure of RP's performance because it highlights how far away the random component axes are from the original features, which RP would hopefully minimize. Since the reconstruction error decreases somewhat linearly with increased number of reduced componenets, we set a threshold on 10% reconstruction error when selecting number of components. Since both datasets have just over 10 features to begin with, this should correlate to roughly 1 feature being reconstructed poorly, and we assume that component will be eliminated anyways. To analyze the new components generated by RP versus the original features, we looked at the variance of each component/feature in relation to each other. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.

| DR Algorithm | # of Features | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| None | 12 | 1.923 | 0.846 | 13 | 0.526 | 0.499 |
| RP | 10 | 1.712 | 0.819 | 9 | 0.382 | 0.476 |

![](unsupervised/plots/RP/recon_error_analysis.png)

Reconstruction error decreases as the number of reduced compnents increases as expected because there would be smaller reconstruction error as you approach the same number of reduced components as original features. Similarly to PCA, we can see that RP projects the data into a new space that results in variance being higher for some individual components while also elminating the components whose variances are being reduced as a result, as shown by the variance plots. Cross-validation scores for the base learner are comparable to the scores returned by the PCA reduction analyzed earlier, which indicates that the random projection of new components works almost as well as actually calculating certain values for these new components such as variance. This suggests that random projection is a valid method for reducting dimensionality espeically when datasets become larger and variance calculations become more costly.

### Factor Analysis (FA)
FA is similar to PCA, however FA attempts to find correlations between features based on an underlying latent factor rather than looking to maximize component variance. For FA component selection, we looked at the variances of each new component as well as their sums. This is a good metric to compare this algorithm to PCA because of their similarity. To analyze the new components generated by FA versus the original features, we look at the variance of each component/feature in relation to each other. The values on the left of the table are for the smart grid dataset, and the values on the right are for the bank loan dataset.

| DR Algorithm | # of Features | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| None | 12 | 1.923 | 0.846 | 13 | 0.526 | 0.499 |
| FA | 6 | 1.179 | 0.771 | 7 | 0.376 | 0.449 |

![](unsupervised/plots/FA/variance_analysis.png)

FA returns a relatively low cross-validation score for the smart grid dataset compared to other DR algorithms. The smart grid dataset contains features that are sensor readings of an electrical grid, and the classification label is whether or not the grid is unstable. Factor analysis would try to find correlations based on some underlying factors influencing the readings. In other words, factor analysis works best when the classification labels influence the features, not the other way around. The smart grid isn't becoming unstable because of the sensor values, rather the sensor values are changing because the smart grid is in unstable. Due to this fundamental difference in how the dataset's features and class labels interact versus how factor analysis works, the factor analysis algorithm reduces much more features than it probably should and results in poor accuracies for this dataset. Regardless, FA still reduces runtime for both datasets, indicating it is a feasible method for cutting back on computational cost regardless of

### Dimensionality Reduction Runtime Analysis

Below are the fit times for each of the DR algorithms for comparison against each other.

| DR Algorithm | Smart Grid Fit Time (s) | Bank Loan Fit Time (s) |
| ---- | ---- | ---- | ---- |
| PCA | 0.0029 | 0.00391 |
| ICA | 0.234 | 0.00781 |
| RP | 0.00098 | 0.00042 |
| FA | 0.798 | 1.192 |

As expected, RP is the fastest because it is easy and fast by just choosing a random projection and going with it. FA was the slowest, which also makes sense because the algorithm performs more calculations trying to indentify the underlying factor's affect on the features as well as projecting the features to a new space. Overall, all the algorithms run at reasonable times relative to some of the fit times for the base boosting learners used for validation in this report, meaning that they would all be reasonable methods for reducing computational cost of learning with large datasets.

##  Dimesionality Reduction + Clustering

The next experiment involved running the datasets through each dimensionality reduction algorithm prior to running them through each clustering algorithm. After running each combination of dimensionality reduction and clustering algorithm, each resulting transformed data set was run through a basic boosting learner for cross-validation score comparison. Boosting was chosen due to its ability to perform well with minimal tuning while also minimizing the possibility of overfitting. Below is a table containing the results, with values on the left side corresponding to the smart grid dataset, and values on the right side corresponding to the bank loan dataset.

| DR Algorithm | Cluster Algorithm | # of Clusters | Boosting Fit Time (s) | Boosting CV Score | # of Clusters | Boosting Fit Time (s) | Boosting CV Score |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| PCA | None | | 1.872 | 0.835 | | 0.619 | 0.445 |
| PCA | K-Means | 2 | 0.971 | 0.998 | 3 | 0.460 | 0.932 |
| PCA | EM | 3 | 1.362 | 0.916 | 4 | 0.638 | 0.868 |
| ICA | None | | 1.968 | 0.849 | | 0.351 | 0.436 |
| ICA | K-Means | 3 | 1.006 | 0.885 | 5 | 0.315 | 0.802 |
| ICA | EM | 3 | 1.441 | 0.897 | 6 | 0.410 | 0.470 |
| RP | None | | 1.611 | 0.819 | | 0.428 | 0.476 |
| RP | K-Means | 2 | 0.988 | 0.972 | 2 | 0.400 | 0.986 |
| RP | EM | 3 | 1.282 | 0.847 | 5 | 0.538 | 0.664 |
| FA | None | | 1.109 | 0.771 | | 0.379 | 0.449 |
| FA | K-Means | 6 | 0.693 | 0.730 | 5 | 0.397 | 0.889 |
| FA | EM | 5 | 0.891 | 0.773 | 6 | 0.516 | 0.825 |

Almost all combinations resulted in faster fit times for the boosting learner as expected. For both datasets, K-means increases accuracy while also reducing runtime for almost every combination. For the bank loan dataset, the k-means algorithm in combination with all of the dimensionality reduction algorithms causes accuracy to increase significantly, usually on the order of 50 percentage points. Even though this dataset was chosen specifically for its ability to trick classifiers with many outliers and imbalanced features that don't conform to much of a pattern, the combination of reducing dimensions followed by K-means reworks the dataset into a more manageable form that isn't as tricky for classifiers. However, the EM algorithm in cominbation with dimensionality doesn't improve accuracy, and in some cases actually lowers it. This is a result of the EM algorithm trying to focus on the distributions of the features, which all of the DR algorithms either directly or indirectly change. For both the factor analysis combinations, the bank loan dataset returned higher cross-validation accuracies than the smart grid dataset. Below is a plot showing these results.

![](unsupervised/plots/k_means/fa_k_means_analysis.png)

This poor performance is shown in the tSNE projections, in which the clusters for the bank loan dataset look tighter and more spread apart than those of the smart grid dataset. This behavior is a result of the FA calculation method discussed earlier that doesn't work well with the smart grid dataset. Even after passing the poorly reduced dataset into the clustering algorithm, the resulting accuracies remain low. This tells us that if the dimensionality reduction algorithm doesn't perform well on a given dataset, then even a well performing clustering algorithm won't be able to make up for the decreaed model performance.

The RP combination with k-means returned the highest accuracy for the bank loan dataset. Below is a plot showing these results.

![](unsupervised/plots/k_means/rp_k_means_analysis.png)

Both clustering algorithms return the smallest number of clusters inspected for both datasets, showing us that random projection can project the data in a way that makes it easy for clustering algorithms to work with. These results suggest that the bank loan dataset in its original form is difficult for learners, but a simple projection in other directions (in this case, random directions) provides a new perspective for the learner, which leads to significantly better accuracies. Although this process works out well in this particular case, considering the stochastic nature of the algorithm, we can't assume that this RP + clustering combination would cause a high increase in accuracies in all cases.

The ICA and EM combination is interesting because of how both algorithms operate contrastingly. ICA attempts to project new components that have a high degree of non-Gaussianity and the EM algorithm is trying to form clusters based on Gaussian distributions. This resulted in relatively poor cross-validation score for the smart grid dataset, but an especially poor cross-validation score for the bank loan dataset. Below is a plot showing these results.

![](unsupervised/plots/em/ica_em_analysis.png)

These results indicate that the smart grid dataset is better suited to work with this combination of algorithms due to its features being more Gaussian by nature in the first place in that they are more evenly distributed with less outliers. Even with this underlying nature, the contrasting algorithms reduce model performance. This resulting poor model performance is especially apparent in the bank loan dataset, which does not have Guassian features, where the accuracy returned is the lowest of all combinations.

## Neural Networks

This analysis compared the performance of the neural network from assignment 1 against a neural network run after dimensionality reduction and clustering was performed with the smart grid dataset. The smart grid dataset was chosen due to its high performance in Assignment 1, so this gave a good benchmark to test against. The first experiment here looked at the dimensionality reduction and neural network combination first. The number of features chosen for each algorithm were the same ones identified in their respective individual analysis sections earlier in this report. Below are results for this experiment.

| Dimensionality Reduction | # of Features | Fit Time (s) | Accuracy | Loss |
| ---- | ---- | ---- | ---- |
| None (A1 Learner) | 12 | 3.68 | 0.958 | 0.064 |
| PCA | 11 | 2.38 | 0.957 | 0.053 |
| ICA | 11 | 1.42 | 0.810 | 0.390 |
| RP | 11 | 2.58 | 0.924 | 0.141 |
| FA | 6 | 2.63 | 0.772 | 0.435 |

![](unsupervised/plots/neural_networks/dr_learning_curves.png)

![](unsupervised/plots/neural_networks/dr_loss_curves.png)

Fit times suggest that neural networks are sensitive to feature distributions. ICA resulted in faster runtime and fewer iterations than both PCA and RP for the same number of reduced features, indicating that the neural network learns with statistically independent features more quickly. However, PCA and RP resulted in better accuracies and lower loss, indicating that features selected based on maximizing variance is preferred when high accuracy is more important than runtime. FA performs comparably to PCA and ICA in terms of runtime, but returns the lowest accuracy because it reduced the number of features the most and because of the downside of how FCA calculates a solution for the smart grid dataset as discussed earlier in the FA section of this report. All algorithms shortened runtime, proving they are feasible methods for reducing runtimes of neural networks.

The final experiment analyzed the effects of using a clustering algorithm to reduce the number of features in a dataset prior to feeding into a neural network. This experiment involved removing a set number of features with the lowest variance (as identified in the earlier dimensionality reduction analyses) and inserting a new feature containing the cluster labels output by each clustering algorithm. The results are in the table below and the plots below show learning and loss curves for the highest number of reduced features explored (4).

| Clustering | Reduced Features | Fit Time (s) | Accuracy | Loss |
| ---- | ---- | ---- | ---- |
| None (A1 Learner) | 0 | 2.449 | 0.958 | 0.064 |
| K-Means | 1 | 1.804 | 0.890 | 0.223 |
| EM | 1 | 1.815 | 0.887 | 0.217 |
| K-Means | 2 | 2.019 | 0.850 | 0.311 |
| EM | 2 | 1.671 | 0.844 | 0.312 |
| K-Means | 3 | 1.912 | 0.808 | 0.278 |
| EM | 3 | 1.802 | 0.802 | 0.379 |
| K-Means | 4 | 1.839 | 0.781 | 0.443 |
| EM | 4 | 1.656 | 0.781 | 0.444 |

![](unsupervised/plots/neural_networks/cluster_learning_curves.png)

![](unsupervised/plots/neural_networks/cluster_loss_curves.png)

These results show a correlation between the number of features reduced and runtime, accuracy, and loss. Replacing any number of original features with a new feature containing the labels returned from either clustering algorithm doesn't result in a large drop in accuracy. This illustrates that using a clustering algorithm to perform some preprocessing on the data to speed up neural network runtime is a feasible approach, especially as dataset size increases. For much larger datasets, we would expect to see a much larger decrease in runtime and larger decrease in accuracy with increasing number of reduced features. The learning and loss curves also have similar shapes to those of the base learner from assignment 1, showing that using clustering algorithms in this manner does not contribute to underfitting the data, which is expected since we reduced the number of features.

## Conclusion

In conclusion, the dimensionality reduction algorithms analyzed in this report are feasible methods for decreasing computational cost of both supervised and unsupervised learners, with each algorithm having strengths and weaknesses when solving various problem types. By projecting the data into a new space, it is possible that such a new space contains stronger correlations between features that results in improved accuracy in learners as well. Clustering algorithms are a great method for unsupervised learning that can cluster simple and complicated datasets containing a variety of features and distributions. They can also be used as a preprocessing method for reducing runtimes by removing less valuable features and replacing them with cluster labels instead, which holds significant value in reducing runtime as dataset complexity increases. When using these two types of algorithms in combination, careful consideration should be taken for the datasets they are used for as well as for the combinations themselves, making sure the algorithms don't contrast with how they calculate solutions.

## References
* sklearn documentation/examples
* Modules/Lectures
* Restructuring Sparse High Dimensional Data forEffective Retrieval, Charles Lee Isbell Jr. & Paul Viola
* The Fundamental Difference Between PCA and Factor Analysis, Karen Grace-Martin
* https://www.methodology.psu.edu/resources/AIC-vs-BIC/