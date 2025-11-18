import pandas as pd
import numpy as np

from lets_plot import *

# packages for Supervised Learning


from sklearn.linear_model import LinearRegression # Least Squares

from sklearn.svm import SVR # Support Vector Machine

from sklearn.neighbors import KNeighborsRegressor # K-Nearest Neighbors

from sklearn.linear_model import ElasticNet # Elastic Net Penalty
from sklearn.linear_model import Lasso # LASSO


from sklearn.tree import DecisionTreeRegressor # Decision Tree


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

import statsmodels.tsa.arima.model as stats
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_squared_error


class SupervisedLearning:

    def __init__(self):
        """Initialize all regression models."""
        self.models = []
        self._init_models()

    def _init_models(self):
        # Linear Models
        self.models.append(("LR", LinearRegression()))
        self.models.append(("LASSO", Lasso()))
        self.models.append(("Elastic Net Penalty", ElasticNet()))

        # Tree-Based
        self.models.append(("Decision Tree", DecisionTreeRegressor()))

        # Bagging
        self.models.append(("Random Forest", RandomForestRegressor()))
        self.models.append(("Extra Trees", ExtraTreesRegressor()))

        # Boosting
        self.models.append(("Gradient Boosting", GradientBoostingRegressor()))
        self.models.append(("Adaptive Boosting", AdaBoostRegressor()))

        # Kernel / Distance-based
        self.models.append(("Support Vector Machine", SVR()))
        self.models.append(("K-Nearest Neighbors", KNeighborsRegressor()))

    def get_model_by_name(self, name):
        for n, model in self.models:
            if name == n:
                return model
            
        print("Warning! no name found")
        return self.models[0];


    def sequential_split(self, X: pd.DataFrame, Y: pd.Series, train_frac=0.8):
        if len(X) != len(Y):
            raise ValueError("X and Y must be equal length")

        if train_frac <= 0 or train_frac >= 1:
            raise ValueError("Train fraction must be between 0 and 1")

        train_size = int(len(X) * train_frac)

        X_train, X_test = X[0:train_size], X[train_size:]
        Y_train, Y_test = Y[0:train_size], Y[train_size:]

        print(f"Sequential Split: {len(X_train)} train / {len(X_test)} test samples")
        return X_train, X_test, Y_train, Y_test
    

    def run_all_models(self, X_train, Y_train, X_test, Y_test, 
                       num_folds=10, seed=42, metric="neg_mean_squared_error"):
        names, kfold_results, train_results, test_results = [], [], [], []

        for name, model in self.models:
            names.append(name)

            kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

            # Cross-validation (negative MSE -> convert to positive)
            cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=metric)
            kfold_results.append(cv_results)

            # Fit model
            res = model.fit(X_train, Y_train)

            # Compute train/test MSE
            train_mse = mean_squared_error(Y_train, res.predict(X_train))
            test_mse = mean_squared_error(Y_test, res.predict(X_test))

            train_results.append(train_mse)
            test_results.append(test_mse)

            # Display progress
            print(f"{name}: CV_Mean={cv_results.mean():.4f}, CV_Std={cv_results.std():.4f}, Train_MSE={train_mse:.4f}, Test_MSE={test_mse:.4f}")

        df_for_comparison = pd.DataFrame({
            "Algorithms": names * 2,
            "Data": ["Training Set"] * len(names) + ["Testing Set"] * len(names),
            "Performance": train_results + test_results
        })

        return {
            "names": names,
            "kfold_results": kfold_results,
            "train_results": train_results,
            "test_results": test_results,
            "comparison_df": df_for_comparison
        }

    def plot_performance(self, df_for_comparison: pd.DataFrame):
        performance_comparison =\
        (
            ggplot(df_for_comparison,
                aes(x = "Algorithms",
                    y = "Performance",
                    fill = "Data"
                    )
                )
            + geom_bar(stat = "identity",
                    position = "dodge",
                    width = 0.5)
            + labs(title = "Comparing the Performance of Machine Learning Algorithms on the Training vs. Testing Set",
                y = "Mean Squared Error (MSE)",
                x = "Name of ML Algorithms",
                caption = "Source: Federal Reserve Bank & Yahoo Finance")
            + theme(legend_position = "top")
            + ggsize(1000, 500)
        )

        performance_comparison.show()
