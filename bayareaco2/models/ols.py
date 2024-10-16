# Anna C. Smith
# GitHub username: edsml-acs223
# Imperial College London - MSc EDSML - IRP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import folium
import branca.colormap as cm
from pathlib import Path
from bayareaco2.models.data import calculate_vif
from bayareaco2.models.data import scale_features


def partial_r2(full_r2, reduced_r2):
    """Calculate the partial R² statistic.

    Parameters:
    full_r2 (float): R² of the full model.
    reduced_r2 (float): R² of the reduced model (with one feature removed).

    Returns:
    float: Partial R² value.
    """
    return (full_r2 - reduced_r2) / (1 - reduced_r2)


class OLS_Model:
    def __init__(self):
        """Initialize the OLS model class."""
        self.model = None

    def add_constant(self, X_train):
        """Add a constant to the feature matrix X_train.

        Parameters:
        X_train (pd.DataFrame): Feature matrix DataFrame.

        Returns:
        pd.DataFrame: X_train DataFrame with constant column added.
        """
        return sm.add_constant(X_train, has_constant="add")

    def fit(self, X_train, y_train):
        """Fit an OLS model to the training data.

        Parameters:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable for training.

        Returns:
        statsmodels.regression.linear_model.OLS: The fitted OLS model.
        """
        X_train_const = self.add_constant(X_train)
        self.model = sm.OLS(y_train, X_train_const).fit()
        return self.model

    def predict(self, X_test):
        """Make predictions on the test data.

        Parameters:
        X_test (pd.DataFrame): Feature matrix for testing.

        Returns:
        np.ndarray: Predicted values.
        """
        X_test_const = self.add_constant(X_test)
        return self.model.predict(X_test_const)

    def calculate_adj_r2(self, n, p, r2):
        """Calculate the adjusted R-squared value.

        Parameters:
        n (int): Number of observations.
        p (int): Number of predictors (including the constant).
        r2 (float): R-squared value.

        Returns:
        float: Adjusted R-squared value.
        """
        return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    def ols_model_metrics(self, y_test, y_pred):
        """Evaluate the fitted model on the test data.

        Parameters:
        y_test (pd.Series): True values for the test data.
        y_pred (np.ndarray): Predicted values from the model.

        Returns:
        results (dict): Dictionary containing evaluation metrics (R², Adjusted R², MSE, RMSE, MAE).
        """
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = self.model.df_model + 1
        adj_r2 = self.calculate_adj_r2(n, p, r2)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        results = {"R²": r2, "Adj. R²": adj_r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

        return results

    def evaluate(self, X_test, y_test, print_results=True):
        """Evaluate the OLS model using test data.

        Parameters:
        X_test (pd.DataFrame): Feature matrix for testing.
        y_test (pd.Series): True values for the test data.
        print_results (bool): Whether to print the evaluation results and plot the results.

        Returns:
        (Optional): Prints evaluation metrics and plots predicted vs. observed values.
        metrics (dict): Dictionary containing evaluation metrics (R², Adjusted R², MSE, RMSE, MAE).
        """
        y_pred = self.predict(X_test)
        metrics = self.ols_model_metrics(y_test, y_pred)

        if print_results:
            print(f"R²: {metrics['R²']:.2f}")
            print(f"Adjusted R²: {metrics['Adj. R²']:.2f}")
            print(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
            print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
            print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")

            # Call plot_results method to plot predicted vs. observed values if print_results is True
            self.plot_results(y_test, y_pred, title="OLS LUR Model")

            return metrics

        else:
            return metrics

    def cross_validate(self, X, y, n_splits=10, random_state=42):
        """Perform cross-validation evaluation of an OLS model.

        Parameters:
        X (pd.DataFrame): Feature matrix for cross-validation.
        y (pd.Series): Target variable for cross-validation.
        n_splits (int): Number of folds for cross-validation (default is 10).
        random_state (int): Random seed for reproducibility (default is 42).

        Returns:
        cv_results (dit): Dictionary containing the mean and standard deviation of the evaluation metrics across folds.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        metrics = {"R²": [], "Adj. R²": [], "MSE": [], "RMSE": [], "MAE": []}

        for train_index, test_index in kf.split(X):
            # Manually splitting data using indices
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create a new instance of the model for each fold
            fold_model = OLS_Model()
            fold_model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = fold_model.predict(X_test)
            fold_metrics = fold_model.ols_model_metrics(y_test, y_pred)

            # Collect metrics for each fold
            for key in metrics:
                metrics[key].append(fold_metrics[key])

        # Calculate mean and std deviation for each metric
        cv_results = {
            key: (np.mean(values), np.std(values)) for key, values in metrics.items()
        }

        print("Cross-Validation Mean Metrics:")
        print(f"R²: {cv_results['R²'][0]:.2f}")
        print(f"Adjusted R²: {cv_results['Adj. R²'][0]:.2f}")
        print(f"Mean Squared Error (MSE): {cv_results['MSE'][0]:.2f}")
        print(f"Root Mean Squared Error (RMSE): {cv_results['RMSE'][0]:.2f}")
        print(f"Mean Absolute Error (MAE): {cv_results['MAE'][0]:.2f}")

        return cv_results

    def plot_results(self, y_test, y_pred, title):
        """Plot predicted vs observed values for the OLS model.

        Parameters:
        y_test (pd.Series): True values for the test data.
        y_pred (np.ndarray): Predicted values from the model.
        title (str): Title of the plot.

        Returns:
        matplotlib.pyplot: Scatter plot of predicted vs observed values.
        """
        plt.figure(figsize=(7, 7))

        # Scatter plot of predicted vs. observed values
        plt.scatter(
            y_test,
            y_pred,
            alpha=0.7,
            edgecolors="w",
            s=100,
            c="royalblue",
            label="Predicted vs. Observed",
        )

        # Plot y=x line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Predicted = Observed",
        )

        # Adding labels and title
        plt.xlabel("Observed CO₂ Values", fontsize=14)
        plt.ylabel("Predicted CO₂ Values", fontsize=14)
        plt.title(
            f"{title}: Predicted vs. Observed $\\mathrm{{CO_2}}$ Levels", fontsize=16
        )

        # Adding grid and legend
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)

        # Show plot
        plt.tight_layout()
        plt.show()

    def feature_stats(self, X_train, y_train):
        """Calculate feature statistics for the trained OLS model.

        Parameters:
        X_train (pd.DataFrame): Feature matrix used in model training.
        y_train (pd.Series): Target variable used in model training.

        Returns:
        summary_table (pd.DataFrame): DataFrame containing feature statistics including coefficients, standard errors, p-values, VIF, and partial R².
        """
        summary_rows = []

        full_r2 = self.model.rsquared

        # Calculating VIF for all features amongst each other
        vif_data = calculate_vif(X_train)

        # Accessing feature specific stats
        for feature in X_train.columns:
            coef = self.model.params[feature]
            std_err = self.model.bse[feature]
            p_value = self.model.pvalues[feature]
            vif = vif_data[vif_data["variables"] == feature]["VIF"].values[0]

            # Calculating partial R² using reduced model
            X_temp = X_train.drop(columns=[feature])
            X_temp_const = sm.add_constant(X_temp)
            reduced_model = sm.OLS(y_train, X_temp_const).fit()
            reduced_r2 = reduced_model.rsquared
            part_r2 = partial_r2(full_r2, reduced_r2)

            summary_rows.append(
                {
                    "Feature": feature,
                    "Coefficient": coef,
                    "Standard Error": std_err,
                    "p-Value": p_value,
                    "VIF": vif,
                    "Partial R²": part_r2,
                }
            )

        summary_table = pd.DataFrame(summary_rows)

        return summary_table

    def evaluate_test_nodes(
        self,
        selected_features,
        scaler,
        non_zero_cols,
        nodes="central",
        print_results=True,
    ):
        """
        Evaluate model performance on each test node and create DataFrames with performance metrics.

        Parameters:
        selected_features (list): List of selected features for the model.
        scaler (StandardScaler): Fitted scaler used to scale the training feature matrix.
        non_zero_cols (list): Feature matrix columns used in feature selection.
        nodes (string): The test node group, either 'central' or 'fringe'.
        print_results (bool): Option to print results as well as returning pd.DataFrame.

        Returns:
        (Optional): prints overall performance metrics and plots node performance.
        metrics_df (pd.DataFrame): DataFrame with node_id as columns and performance metrics as rows.
        overall_metrics_df (pd.DataFrame): DataFrame with overall performance metrics.
        """
        if nodes == "central":
            path = (
                Path(__file__).resolve().parent.parent.parent
                / "Data"
                / "CO2"
                / "central_test_nodes.csv"
            )
            test_nodes = pd.read_csv(path, index_col=0)
        if nodes == "fringe":
            path = (
                Path(__file__).resolve().parent.parent.parent
                / "Data"
                / "CO2"
                / "fringe_test_nodes.csv"
            )
            test_nodes = pd.read_csv(path, index_col=0)
        path = Path(__file__).resolve().parent.parent.parent / "Data" / "features.csv"
        features = pd.read_csv(path)

        # Merge co2 and features data
        df = test_nodes.merge(features, on="node_id")

        # Prepare train and test feature matrix
        feature_vars = features.drop(columns="node_id").columns.to_list()
        feature_vars.extend(["temp", "pressure", "rh"])
        X_test_nodes = df[feature_vars].copy()
        X_test_nodes = X_test_nodes[non_zero_cols]
        X_test_nodes = scale_features(scaler, X_test_nodes)
        X_test_nodes = X_test_nodes[selected_features]

        # Define target variable
        y_test_nodes = df["co2"].copy()
        node_ids = df["node_id"].copy()

        metrics_dict = {"node_id": [], "R²": [], "MSE": [], "RMSE": [], "MAE": []}

        # Evaluate performance for each node
        all_y_test = []
        all_y_pred = []
        for node in node_ids.unique():
            node_filter = node_ids == node
            X_node = X_test_nodes[node_filter]
            y_node = y_test_nodes[node_filter]

            if not X_node.empty and not y_node.empty:
                # Make predictions
                y_pred = self.predict(X_node)

                # Calculate metrics
                r2 = r2_score(y_node, y_pred)
                mse = mean_squared_error(y_node, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_node, y_pred)

                metrics_dict["node_id"].append(node)
                metrics_dict["R²"].append(r2)
                metrics_dict["MSE"].append(mse)
                metrics_dict["RMSE"].append(rmse)
                metrics_dict["MAE"].append(mae)

                # Collect overall test data
                all_y_test.extend(y_node)
                all_y_pred.extend(y_pred)

        # Convert node metrics dictionary to DataFrame
        metrics_df = pd.DataFrame(metrics_dict).set_index("node_id").T

        # Calculate overall performance metrics
        overall_r2 = r2_score(all_y_test, all_y_pred)
        overall_mse = mean_squared_error(all_y_test, all_y_pred)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(all_y_test, all_y_pred)

        overall_metrics = {
            "R²": overall_r2,
            "MSE": overall_mse,
            "RMSE": overall_rmse,
            "MAE": overall_mae,
        }

        overall_metrics_df = pd.DataFrame(overall_metrics, index=["Overall"])

        if print_results:
            # Print overall metrics
            print("Overall Performance Metrics:")
            print(f"R²: {overall_metrics_df.loc['Overall', 'R²']:.2f}")
            print(
                f"Mean Squared Error (MSE): {overall_metrics_df.loc['Overall', 'MSE']:.2f}"
            )
            print(
                f"Root Mean Squared Error (RMSE): {overall_metrics_df.loc['Overall', 'RMSE']:.2f}"
            )
            print(
                f"Mean Absolute Error (MAE): {overall_metrics_df.loc['Overall', 'MAE']:.2f} \n"
            )

            # Create a dictionary to pass R² values to plotting function
            r2_dict = dict(zip(metrics_dict["node_id"], metrics_dict["R²"]))

            # Plot results for each node
            self.plot_node_performance(
                metrics_df, X_test_nodes, y_test_nodes, node_ids, r2_dict
            )

            return metrics_df, overall_metrics_df

        else:
            return metrics_df, overall_metrics_df

    def plot_node_performance(
        self, metrics_df, X_test_nodes, y_test_nodes, node_ids, r2_dict
    ):
        """
        Plot performance metrics for each node in a grid layout.

        Parameters:
        metrics_df (pd.DataFrame): DataFrame with performance metrics for each node.
        X_test_nodes (pd.DataFrame): Test features.
        y_test_nodes (pd.Series): True target values.
        node_ids (pd.Series): Node IDs.
        r2_dict (dict): Dictionary mapping node IDs to their R² values.

        Returns:
        matplotlib.pyplot: Predicted vs. observed values for each node.
        """
        num_nodes = len(metrics_df.columns)
        num_cols = 3
        num_rows = (
            num_nodes + num_cols - 1
        ) // num_cols  # Calculate number of rows needed

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
        )
        axes = axes.flatten()

        for i, node_id in enumerate(metrics_df.columns):
            ax = axes[i]
            y_test = y_test_nodes[node_ids == node_id]
            X_node = X_test_nodes[node_ids == node_id]
            y_pred = self.predict(X_node)

            ax.scatter(
                y_test,
                y_pred,
                alpha=0.7,
                edgecolors="w",
                s=100,
                c="royalblue",
                label="Predicted vs. Observed",
            )

            # Plot y=x line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Predicted = Observed",
            )

            # Adding labels and title with R² value
            r2_value = r2_dict[node_id]
            ax.set_xlabel("Observed CO₂ Values", fontsize=12)
            ax.set_ylabel("Predicted CO₂ Values", fontsize=12)
            ax.set_title(f"Node {node_id}, R² = {r2_value:.2f}", fontsize=14)

            # Adding grid and legend
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(fontsize=10)

        # Hide unused subplots
        for i in range(num_nodes, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()  # Adjusts subplot params for a cleaner layout
        plt.show()

    def make_node_map(self, full_X_train, selected_features, scaler, non_zero_cols):
        """
        Create a map of nodes based on model evaluation on test nodes.

        Parameters:
        full_X_train (pd.DataFrame): The training data used for model training.
        selected_features (list): List of selected features for the model.
        scaler (StandardScaler): Fitted scaler used to scale the training feature matrix.
        non_zero_cols (list): Feature matrix columns used in feature selection.

        Returns:
        eval_metircs_df (pd.DataFrame): DataFrame with node_id as rows and performance metrics columns.
        nodes_map (folium.Map): folium map featuring ndoe locations and associated statistics.
        """
        # Load all valid CO2 daily average data
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "daily_avg_BEACO2N.csv"
        )
        all_valid_data = pd.read_csv(path)

        # Load features data
        path = Path(__file__).resolve().parent.parent.parent / "Data" / "features.csv"
        features = pd.read_csv(path)

        all_valid_test = all_valid_data[
            all_valid_data.index.isin(full_X_train.index) == False
        ]

        # Merge CO2 and features data
        df = all_valid_test.merge(features, on="node_id")
        df.index = all_valid_test.index

        node_eval_y = df["co2"].copy()
        node_ids = df["node_id"].copy()

        # Prepare feature matrix for evaluation
        feature_vars = features.drop(columns="node_id").columns.to_list()
        feature_vars.extend(["temp", "pressure", "rh"])
        node_eval_X = df[feature_vars].copy()
        node_eval_X = node_eval_X[non_zero_cols]
        node_eval_X = scale_features(scaler, node_eval_X)
        node_eval_X = node_eval_X[selected_features]

        # Initialize a dictionary to store evaluation metrics
        eval_metrics = {
            "node_id": [],
            "n": [],
            "R²": [],
            "MSE": [],
            "RMSE": [],
            "MAE": [],
        }

        # Evaluate model performance for each node
        for node in node_ids.unique():
            node_filter = node_ids == node
            X_node = node_eval_X[node_filter]
            y_node = node_eval_y[node_filter]
            n = len(y_node)
            eval_metrics["n"].append(n)

            if not X_node.empty and not y_node.empty:
                # Make predictions using the trained model (self.predict assumed to be defined elsewhere)
                y_pred = self.predict(X_node)

                # Calculate evaluation metrics
                r2 = r2_score(y_node, y_pred)
                mse = mean_squared_error(y_node, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_node, y_pred)

                # Store the metrics for this node
                eval_metrics["node_id"].append(node)
                eval_metrics["R²"].append(r2)
                eval_metrics["MSE"].append(mse)
                eval_metrics["RMSE"].append(rmse)
                eval_metrics["MAE"].append(mae)

        # Convert metrics dictionary to DataFrame and set node_id as the index
        eval_metrics_df = pd.DataFrame(eval_metrics).set_index("node_id")

        eval_metrics_df = eval_metrics_df.round(2).reset_index()

        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "node_locations.csv"
        )
        node_locations = pd.read_csv(path)
        node_results_locations = eval_metrics_df.merge(node_locations, on="node_id")

        # Creating a color map for the node categories
        color_map = {
            "imbalanced/invalid node": "grey",
            "training node": "blue",
            "central test node": "red",
            "fringe test node": "darkred",
        }

        # Creating a colormap for the R² values
        linear = cm.LinearColormap(
            colors=["red", "yellow", "green"],
            index=[0, 0.35, 0.75],
            vmin=-0.01,
            vmax=1.0,
        )

        map_center = [37.955, -122.2]
        nodes_map = folium.Map(location=map_center, zoom_start=10)

        # Add CircleMarkers to the map
        for _, row in node_results_locations.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=6,
                color=color_map.get(row["label"], "black"),
                fill=True,
                fill_color=linear(row["R²"]),
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"""
                    <div style="width: 82px; padding: 5px; font-size: 12px;">
                        <b>node_id: {row['node_id']}</b><br>
                        n: {row['n']}<br>
                        R²: {row['R²']}<br>
                    </div>
                    """,
                    max_width=300,
                ),
            ).add_to(nodes_map)

        # Create a colorbar for the R² values
        colorbar_html = linear._repr_html_().replace(
            "height: 20px;", "height: 40px; width: 80px; margin: 5px auto;"
        )

        # Create a legend for the map with the colorbar and node categories
        legend_html = f"""
            <div style="position: fixed; 
            bottom: 0px; left: 100px; width: 475px; height: auto; 
            border:2px solid grey; background-color:white;
            z-index:9999; font-size:14px; padding:10px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.5);
            ">
            &nbsp; <b>Node Category</b><br>
            &nbsp; <i class="fa fa-circle" style="font-size:8px; color:transparent; border: 4px solid grey; border-radius: 50%; padding: 2px;"></i>&nbsp; Imbalanced node<br>
            &nbsp; <i class="fa fa-circle" style="font-size:8px; color:transparent; border: 4px solid blue; border-radius: 50%; padding: 2px;"></i>&nbsp; Training node<br>
            &nbsp; <i class="fa fa-circle" style="font-size:8px; color:transparent; border: 4px solid red; border-radius: 50%; padding: 2px;"></i>&nbsp; Central test node<br>
            &nbsp; <i class="fa fa-circle" style="font-size:8px; color:transparent; border: 4px solid darkred; border-radius: 50%; padding: 2px;"></i>&nbsp; Fringe test node<br><br>
            &nbsp; <b>R² Colorbar</b><br>
            {colorbar_html}
            </div>
        """

        # Add legend to the map
        folium.Marker(
            location=[node_locations["lat"].mean(), node_locations["lng"].mean()],
            icon=folium.DivIcon(html=legend_html),
        ).add_to(nodes_map)

        output_html_path = (
            Path(__file__).resolve().parent.parent
            / "notebooks"
            / "explore"
            / "lur_node_map.html"
        )
        nodes_map.save(output_html_path)

        # Display the map
        return eval_metrics_df, nodes_map
