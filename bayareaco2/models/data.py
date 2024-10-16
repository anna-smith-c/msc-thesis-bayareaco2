# Anna C. Smith
# GitHub username: edsml-acs223
# Imperial College London - MSc EDSML - IRP

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path


def load_Xy(balanced=True, return_node_id=False):
    """
    Load and merge CO2 and feature data.

    Parameters:
    balanced (bool): If True, load the balanced CO2 dataset. If False, load the full dataset.
    return_node_id (bool): If True, return the node_id along with the features and target variable.

    Returns:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable (CO2 levels).
    node_id (pd.Series): Node ID if return_node_id is True, otherwise not included.
    """
    if balanced == True:
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "balanced_daily_avg_BEACO2N.csv"
        )
        co2 = pd.read_csv(path, index_col=0)
    if balanced == False:
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "daily_avg_BEACO2N.csv"
        )
        co2 = pd.read_csv(path, index_col=0)
    path = Path(__file__).resolve().parent.parent.parent / "Data" / "features.csv"
    features = pd.read_csv(path)

    df = co2.merge(features, on="node_id")
    df.index = co2.index

    assert np.max(df.isna().sum()) == 0, "Dataframe contains missing data"

    y = df["co2"].copy()

    feature_vars = features.drop(columns="node_id").columns.to_list()
    feature_vars.append("temp")
    feature_vars.append("pressure")
    feature_vars.append("rh")

    node_id = df["node_id"].copy()

    # Defining dependent variables
    X = df[feature_vars].copy()

    assert X.shape[1] == 123, "Incorrect number of features"

    zero_columns = [col for col in X.columns if (X[col] == 0).all()]

    X = X.drop(columns=zero_columns)
    print(f"Dropping {len(zero_columns)} features with all zero data")

    assert X.index.equals(y.index), "Index mismatch"

    print(f"Number of observations: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    if return_node_id:
        return X, y, node_id
    else:
        return X, y


def fit_scaler(X_train):
    """
    Fit the StandardScaler on the training data.

    Parameters:
    X_train (pd.DataFrame): The feature matrix for training.

    Returns:
    StandardScaler: The fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_features(scaler, X):
    """
    Transform the data using the fitted scaler.

    Parameters:
    scaler (StandardScaler): The fitted scaler object.
    X (pd.DataFrame): The feature matrix to be scaled.

    Returns:
    pd.DataFrame: Scaled feature matrix with the same index and columns as the original.
    """
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def calculate_vif(X_train):
    """
    Calculate Variance Inflation Factor (VIF) for each feature.

    Parameters:
    X_train (pd.DataFrame): The feature matrix used in model training.

    Returns:
    vif (pd.DataFrame): A DataFrame containing the features and their corresponding VIF values.
    """
    vif = pd.DataFrame()
    vif["variables"] = X_train.columns
    vif["VIF"] = [
        variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])
    ]
    return vif


def drop_high_vif(X_train, threshold=3):
    """
    Iteratively drop features with high VIF until all VIF values are below the threshold.

    Parameters:
    X_train (pd.DataFrame): The feature matrix used in model training.
    threshold (float): The VIF threshold above which features will be dropped.

    Returns:
    X_train (pd.DataFrame): The feature matrix with high VIF features removed.
    vif (pd.DataFrame): A DataFrame containing the features and their corresponding VIF values after removal.
    """
    while True:
        vif = calculate_vif(X_train)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            max_vif_var = vif.loc[vif["VIF"].idxmax(), "variables"]
            X_train = X_train.drop(columns=[max_vif_var])
        else:
            break
    return X_train, vif


def feature_selection(
    X, y, spearman_threshold=0.03, vif_threshold=3, return_spearman_vif=False
):
    """
    Select features based on Spearman's correlation and VIF thresholds.

    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target variable (CO2 levels).
    spearman_threshold (float): The minimum absolute value of Spearman's correlation coefficient for feature selection.
    vif_threshold (float): The maximum VIF value for feature selection.
    return_spearman_vif (bool): If True, return Spearman's correlation and VIF DataFrames.

    Returns:
    selected_features (pd.Index): The selected features after filtering by Spearman's correlation and VIF.
    (Optional) spearman_df (pd.DataFrame): DataFrame of features with their Spearman's correlation coefficients.
    (Optional) vif (pd.DataFrame): DataFrame of features with their VIF values.
    """
    spearman_results = []
    for column in X.columns:
        if np.std(X[column]) == 0:
            print(f"Column {column} is constant.")
            coef = np.nan
        else:
            coef, _ = spearmanr(X[column], y)
        spearman_results.append((column, coef))

    spearman_results_sorted = sorted(
        spearman_results,
        key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0,
        reverse=True,
    )

    filtered_spearman_results = [
        (feature, coef)
        for feature, coef in spearman_results_sorted
        if abs(coef) >= spearman_threshold
    ]

    print(
        f"{len(filtered_spearman_results)} features selected with Spearman's correlation coefficient ≥ {spearman_threshold}"
    )

    filtered_features = [feature for feature, _ in filtered_spearman_results]

    vif_features, vif = drop_high_vif(X[filtered_features], threshold=vif_threshold)

    selected_features = vif_features.columns

    print(f"{len(selected_features)} features selected with VIF < {vif_threshold}: \n")

    print(f"{len(selected_features)} features selected: \n")
    for i in selected_features:
        print(i)

    if return_spearman_vif:
        pd.set_option("display.precision", 2)
        spearman_df = pd.DataFrame(
            filtered_spearman_results, columns=["Feature", "Spearman"]
        )
        return selected_features, spearman_df, vif

    else:
        return selected_features


def plot_corr(X_train):
    """
    Plot the correlation matrix of the training features.

    Parameters:
    X_train (pd.DataFrame): The feature matrix used in model training.

    Returns:
    None: Displays the correlation matrix plot.
    """
    corrmat = X_train.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(12, 11))
    g = sns.heatmap(X_train[top_corr_features].corr(), annot=True, cmap="RdYlGn")
