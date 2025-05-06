import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as dp
from termcolor import cprint
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import RegressorMixin
import shap
from typing import List, Dict


# plotting related


def fig_size(w=3, h=3)-> None:
    """
    Lazy function for setting figure size

    Args:
        w (int, optional): set fig width. Defaults to 3.
        h (int, optional): set fig length. Defaults to 3.
    """
    plt.figure(figsize=(w, h))


def bprint(input)-> None:
    """
    Style printing with color 

    Args:
        input (any): content to print
    """
    cprint(f"\n{input}", "green", attrs=["bold"])


def mark_bar(plot)-> None:
    """
    Append bar values on the bars

    Args:
        plot (matplotlib axis): plot
    """
    for i in plot.containers:
        plot.bar_label(
            i,
        )


def mark_percent(ax, col: pd.Series, hue: pd.Series, target_class: str | int)-> None:
    """
    Mark percentage for stacked histogram

    Args:
        ax: axes of plot
        col (pd.Series): dataframe column for x-axis
        hue (pd.Series): dataframe column for hue
        target_class (str): target class in hue column
    """
    tab = pd.crosstab(col, hue)
    tab_index = tab.index.tolist()
    tab_norm = pd.crosstab(col, hue, normalize="index").round(3) * 100
    total_val = tab.sum().sum()
    col_total = tab.sum(axis=1)
    # column percentage
    col_percent = (col_total * 100 / total_val).round(2).tolist()
    # percentage of target class in column
    col_target_class_percent = tab_norm[target_class].tolist()
    # add percentages to df
    percent_df = pd.DataFrame([col_percent], columns=tab_index, index=["col_percent"]).T
    percent_df["col_target_class_percent"] = col_target_class_percent
    # add value count to df
    percent_df["col_count"] = tab.sum(axis=1).tolist()
    percent_df["col_target_class_count"] = tab[target_class]
    x_ticks = [i.get_text() for i in ax.get_xticklabels()]

    # append percentages to histplot
    for i in x_ticks:
        # total percentage
        ax.text(
            i,
            percent_df.col_count.loc[i],
            "%0.2f" % percent_df.col_percent.loc[i] + "%",
            ha="center",
        )
        # target class percentage
        ax.text(
            i,
            percent_df.col_target_class_count.loc[i],
            "%0.2f" % percent_df.col_target_class_percent.loc[i] + "%",
            ha="center",
        )


def mark_df_color(col: pd.Series, id, color="rosybrown")-> List[str]:
    """
    Mark specified column or row with color for dataframe

    Args:
        col: pandas series passed in from apply method
        id: index of the column or row
        color: color for marking, default to rosybrown

    Returns:
        List[str]: list of background color styles for each cell in the column or row
    """

    def mark():
        return [
            (f"background-color: {color}" if idx == id else "background-color: ")
            for idx, _ in enumerate(col)
        ]

    return mark()


# modeling related

def cv_scores(X: pd.DataFrame | np.ndarray, y: np.ndarray, model_ls: List[RegressorMixin], model_name_ls: List[str])-> Dict[str, pd.DataFrame]:
    """
    Use 5-folds cross validation to evalute metrics of models

    Args:
        X : features
        y : target
        model_ls: list of sklearn models
        model_name_ls: list of model names

    Returns:
        dict: metrics scores mean and std of models in dataframe
    """
    # 5-folds cv
    folds = 5
    stratified_kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

    cv_compare_mean = pd.DataFrame()
    cv_compare_std = pd.DataFrame()

    for i in range(len(model_ls)):
        cv_result = cross_validate(
            model_ls[i],
            X,
            y,
            cv=stratified_kfold,
            scoring={
                "roc_auc": "roc_auc_ovr",
                "f1": "f1",
                "pr_auc": "average_precision",
                "recall": "recall",
                "accuracy": "accuracy",
                "precision": "precision",
            },
        )
        cv_result = pd.DataFrame(cv_result)
        cv_result_mean = cv_result.mean()
        cv_result_std = cv_result.std()
        cv_compare_mean[model_name_ls[i]] = cv_result_mean
        cv_compare_std[model_name_ls[i]] = cv_result_std

    cv_compare_mean.index = [
        (i, i[5:])[i.startswith("test")] for i in cv_compare_mean.index.tolist()
    ]
    cv_compare_std.index = [
        (i, i[5:])[i.startswith("test")] for i in cv_compare_std.index.tolist()
    ]

    return {"mean": cv_compare_mean, "std": cv_compare_std}



def plot_confusion(y_pred: np.ndarray, y_train: np.ndarray, title: str)-> None:
    """
    Plot confusion matrix

    Args:
        y_pred (np.ndarray): model prediction
        y_train (np.ndarray): y_train
        title (str): confusion matrix plot title
    """
    cf = confusion_matrix(y_train, y_pred, labels=[0, 1])
    sns.heatmap(cf, annot=True, fmt=".0f")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)


def show_rank_scores(model: RandomizedSearchCV)-> None:
    """
    Plot ranked scores from GridSearch or RandomizedSearch cv results

    Args:
        model (RegressorMixin): Search wrapped estimator
    """
    dp(
        pd.DataFrame(model.cv_results_)
        .sort_values(by="rank_test_score")[
            ["rank_test_score", "mean_test_score", "std_test_score"]
        ]
        .head()
        .style.apply(mark_df_color, id=0, axis=0)
    )


def plot_tree_shap_feature_importance(
    model: RegressorMixin, model_name: str, X_train: pd.DataFrame
)-> None:
    """
    Plot shap feature importance for model

    Args:
        model: tree models
        model_name (str)
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.gcf().set_size_inches(5, 3)
    plt.title(f"{model_name} Shap Feature Importance")
    plt.show()
