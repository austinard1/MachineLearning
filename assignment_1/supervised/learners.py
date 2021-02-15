# -*- coding: utf-8 -*-
"""
Supervised Learning - Learners
Learners:
* Boosting Learner
* Decision Tree (DT) Learner
* k-Nearest Neighbors (KNN) Learner
* Neural Networks (NN) Learner
* Support Vector Machines (SVM) Learner
"""
import ast
import copy
import json
import logging
import multiprocessing
import os
import sys
import time

from sklearn import ensemble
from sklearn import exceptions
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from sklearn.utils import _testing

from suplearn import data
from suplearn import plot

# Logger
LOGGER = logging.getLogger(__name__)


class BaseLearner:
    """Base Learner"""

    def __init__(self, learner, dataset, **kwargs):
        """Constructor for BaseLearner"""
        self.name = self.__class__.__name__
        self.func = learner
        self.learner = learner(**kwargs)
        self.data = dataset
        self.test_params = dict()
        self.best_params = dict()
        self.timing = dict()
        self.scores = list()
        self.grid_searched = False
        self.num_cores = round(multiprocessing.cpu_count() * 0.75)

    def __repr__(self):
        """Simple __repr__"""
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def __str__(self):
        """Simple __str___"""
        return self.__repr__()

    def get_grid_search_params(self):
        """Returns grid search params"""
        grid_search_params = dict()
        test_params = self.data.metadata["params"][self.name]
        for key, value in test_params.items():
            if not value["nested"]:
                grid_search_params[key] = value["vals"]
            else:
                grid_search_params[key] = list()
                for iter_val in value["vals"]:
                    module = sys.modules[value["module"]]
                    constructor = getattr(module, value["class"])
                    instance = constructor(**{value["arg"]: iter_val})
                    grid_search_params[key].append(instance)
        return grid_search_params

    @_testing.ignore_warnings(category=exceptions.ConvergenceWarning)
    def grid_search(self):
        """Run grid search"""
        self.test_params = self.get_grid_search_params()
        LOGGER.info(f"{self.name}: Running grid search with {self.num_cores} cores")
        time_start = time.time()
        grid_search = model_selection.GridSearchCV(
            self.learner, param_grid=self.test_params, n_jobs=self.num_cores,
        )
        grid_search.fit(self.data.x_train, self.data.y_train)
        self.timing["grid_search"] = round(time.time() - time_start, 3)
        self.best_params = grid_search.best_params_
        self.learner = self.func(**self.best_params)
        self.grid_searched = True

    def fit(self):
        """Fit"""
        LOGGER.info(f"{self.name}: Running fit (grid_search: {self.grid_searched})")
        time_start = time.time()
        self.learner.fit(self.data.x_train, self.data.y_train)
        self.timing["fit"] = round(time.time() - time_start, 3)

    def predict(self):
        """Predict"""
        LOGGER.info(f"{self.name}: Running predict (grid_search: {self.grid_searched})")
        time_start = time.time()
        self.data.y_predict = self.learner.predict(self.data.x_test)
        self.timing["predict"] = round(time.time() - time_start, 3)

    def score(self):
        """Score"""
        accuracy = metrics.accuracy_score(self.data.y_test, self.data.y_predict)
        self.scores.append({"grid_search": self.grid_searched, "accuracy": accuracy})
        LOGGER.info(
            f"{self.name}: Accuracy = {accuracy * 100: .2f} (grid_search: {self.grid_searched})"
        )

    def run(self):
        """Fit, predict, and score (without and optionally with grid search)"""
        self.fit()
        self.predict()
        self.score()

    def to_dict(self, for_json=False):
        """Convert self to dictionary"""
        # Create dict
        output_dict = {
            "name": self.name,
            "learner": f"{self.func.__module__}.{self.func.__name__}",
            "filename": self.data.filename,
            "test_params": copy.deepcopy(self.test_params),
            "best_params": copy.deepcopy(self.best_params),
            "timing": self.timing,
            "scores": self.scores,
        }
        # Massage for JSON serialization if need be
        if for_json:
            # Check if JSON serializable as is
            try:
                json.dumps(output_dict, indent=4)
            # If auto-generation doesn't work, convert any stubborn fields
            # to dicts/strings manually
            except TypeError:
                for key, value in output_dict.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            try:
                                json.dumps(nested_value)
                            except TypeError:
                                if hasattr(nested_value, "__dict__"):
                                    output_dict[key][nested_key] = nested_value.__dict__
                                elif hasattr(nested_value, "__len__") and hasattr(
                                    nested_value[0], "__dict__"
                                ):
                                    output_dict[key][nested_key] = [
                                        iter_value.__dict__ for iter_value in nested_value
                                    ]
                                else:
                                    output_dict[key][nested_key] = str(nested_value)
                    else:
                        try:
                            json.dumps(value)
                        except TypeError:
                            output_dict[key] = str(value)
        return output_dict

    def to_json(self, filename):
        """Write to JSON file
        Args:
            filename (str): Path to JSON file to write
        """
        # Write to file
        with open(filename, "w") as open_file:
            open_file.write(json.dumps(self.to_dict(for_json=True), indent=4))

    def plot(self, dirname):
        """Generate plots
        Args:
            dirname (str): Output directory to save figures to
        """
        # TODO: This needs to be reviewed when I am a bit more sober
        # Learning curve
        LOGGER.info(f"{self.name}: Plotting learning curve")
        fig = plot.plot_learning_curve(
            self.learner,
            self.name,
            self.data.x_train,
            self.data.y_train,
            n_jobs=self.num_cores,
        )
        fig.savefig(os.path.join(dirname, f"{self.name}_learning_curve.png"))
        # Validation curve
        for param_name, param_values in self.test_params.items():
            # Remove any null/None values
            param_range = [iter_val for iter_val in param_values if iter_val is not None]
            # If we have values left (and they're not all strings), let's see if
            # we can run a literal eval on it. If we can, f*** it, let's assume it's
            # plot-able data try to plot a validation curve
            if param_range and not all([isinstance(iter_val, str) for iter_val in param_range]):
                try:
                    ast.literal_eval(str(param_range))
                    LOGGER.info(
                        f"{self.name}: Plotting validation curve for parameter: {param_name}"
                    )
                    fig = plot.plot_validation_curve(
                        self.learner,
                        self.name,
                        self.data.x_train,
                        self.data.y_train,
                        param_name=param_name,
                        param_range=param_range,
                        n_jobs=self.num_cores,
                    )
                    fig.savefig(
                        os.path.join(dirname, f"{self.name}_{param_name}_validation_curve.png")
                    )
                except (ValueError, TypeError):
                    LOGGER.info(
                        f"{self.name}: Could not plot validation curve for parameter: {param_name}"
                    )


class BoostingLearner(BaseLearner):
    """Boosting Learner"""

    def __init__(self, dataset, **kwargs):
        super().__init__(ensemble.AdaBoostClassifier, dataset, **kwargs)


class DTLearner(BaseLearner):
    """Decision Tree Learner"""

    def __init__(self, dataset, **kwargs):
        super().__init__(tree.DecisionTreeClassifier, dataset, **kwargs)


class KNNLearner(BaseLearner):
    """k-Nearest Neighbors Learner"""

    def __init__(self, dataset, **kwargs):
        super().__init__(neighbors.KNeighborsClassifier, dataset, **kwargs)


class NNLearner(BaseLearner):
    """Neural Networks Learner"""

    def __init__(self, dataset, **kwargs):
        super().__init__(neural_network.MLPClassifier, dataset, **kwargs)


class SVMLearner(BaseLearner):
    """Support Vector Machines Learner"""

    def __init__(self, dataset, **kwargs):
        super().__init__(svm.SVC, dataset, **kwargs)


def get_learners(dataset, learners):
    """Return list of learner instances
    Args:
        dataset (data.Data): Data class
        learners (iter(str)): Iterable of learner names
    """
    module = sys.modules[__name__]
    learner_instances = list()
    for learner in learners:
        if hasattr(module, learner):
            learner_instances.append(getattr(module, learner)(dataset))
    return learner_instances