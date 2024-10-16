# Anna C. Smith
# GitHub username: edsml-acs223
# Imperial College London - MSc EDSML - IRP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import shap
import folium
import branca.colormap as cm
from pathlib import Path
from bayareaco2.models.data import scale_features


class XGBoost_Model:
    def __init__(self, **params):
        """
        Initialize the XGBoost model with the given parameters.

        Parameters:
        **params: Keyword arguments for XGBoost model parameters.
        """
        # Use the given parameters to initialize the model
        self.params = params

        # Inheriting the XGBoost regressor model
        self.model = xgb.XGBRegressor(objective="reg:squarederror", **params)

    def fit(self, X_train, y_train):
        """
        Fit the XGBoost model to the training data.

        Parameters:
        X_train (pd.DataFrame or np.array): The training feature data.
        y_train (pd.Series or np.array): The training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the fitted XGBoost model.

        Parameters:
        X_test (pd.DataFrame or np.array): The test feature data.

        Returns:
        np.array: The predicted values.
        """
        return self.model.predict(X_test)

    def calculate_metrics(self, y_test, y_pred):
        """
        Calculate evaluation metrics for the model's predictions.

        Parameters:
        y_test (pd.Series or np.array): The true target values.
        y_pred (np.array): The predicted values.

        Returns:
        dict: A dictionary containing the evaluation metrics:
        - "R²": Coefficient of determination.
        - "MSE": Mean Squared Error.
        - "RMSE": Root Mean Squared Error.
        - "MAE": Mean Absolute Error.
        """
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        return {"R²": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

    def evaluate(self, X_test, y_test, print_results=True):
        """
        Evaluate the XGBoost model's performance on test data.

        Parameters:
        X_test (pd.DataFrame or np.array): The test feature data.
        y_test (pd.Series or np.array): The test target data.
        print_results (bool): If True, prints the evaluation metrics and plots the results.

        Returns:
        dict: A dictionary containing the evaluation metrics.
        """
        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)

        if print_results:
            # Print evaluation metrics
            print(f"R²: {metrics['R²']:.2f}")
            print(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
            print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
            print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")

            # Call plot_results method if print_results is True
            self.plot_results(y_test, y_pred, title="XGBoost")

            return metrics

        else:
            return metrics

    def plot_results(self, y_test, y_pred, title):
        """
        Plot a scatter plot comparing the true target values with the predicted values.

        Parameters:
        y_test (pd.Series or np.array): The true target values.
        y_pred (np.array): The predicted values.
        title (str): The title for the plot.

        Returns:
        matplotlib.pyplot: A scatter plot of the predicted vs. observed values.
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

        metrics = {"R²": [], "MSE": [], "RMSE": [], "MAE": []}

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Create a new instance of the model for each fold
            fold_model = XGBoost_Model(**self.params)

            # Train the model
            fold_model.fit(X_train, y_train)

            # Make predictions
            y_pred = fold_model.predict(X_test)

            # Calculate metrics
            fold_metrics = fold_model.calculate_metrics(y_test, y_pred)

            # Collect metrics for each fold
            for key in metrics:
                metrics[key].append(fold_metrics[key])

        # Calculate mean for each metric
        cv_results = {key: np.mean(values) for key, values in metrics.items()}

        print(f"{n_splits}-fold CV R²: {cv_results['R²']:.2f}")
        print(f"{n_splits}-fold CV Mean Squared Error (MSE): {cv_results['MSE']:.2f}")
        print(
            f"{n_splits}-fold CV Root Mean Squared Error (RMSE): {cv_results['RMSE']:.2f}"
        )
        print(f"{n_splits}-fold CV Mean Absolute Error (MAE): {cv_results['MAE']:.2f}")

        return cv_results

    def plot_importances(self, X_train):
        """
        Plot feature importances of the fitted XGBoost model.

        Parameters:
        X_train (pd.DataFrame): Training feature data for plotting

        Returns:
        matplotlib.pyplot: Feature importances plot
        """
        # Check if model is fitted
        # if not hasattr(self.model, 'feature_importances_'):
        #    raise RuntimeError("Model must be fitted before plotting feature importances.")

        # Get feature importances and sort them
        feature_importances = self.model.feature_importances_
        sorted_indices = np.argsort(feature_importances)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Retrieve feature names from X_train
        # if isinstance(X_train, pd.DataFrame):
        #    feature_names = X_train.columns
        # else:
        #    feature_names = range(len(feature_importances))  # Fallback if X_train is not a DataFrame

        # if len(sorted_indices) != len(feature_names):
        #    sorted_indices = sorted_indices[:len(feature_names)]

        # Plot horizontal bar chart
        plt.barh(
            X_train.columns[sorted_indices],
            feature_importances[sorted_indices],
            color="royalblue",
            alpha=0.7,
            edgecolor="k",
            height=0.6,
        )

        # Add labels and title
        plt.xlabel("Feature Importance", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.title("XGBoost Feature Importance", fontsize=16)

        # Add horizontal gridlines
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Show the plot
        plt.show()

    def plot_shap(self, X_train):
        """
        Plot SHAP values of the fitted XGBoost model.

        Parameters:
        X_train (pd.DataFrame): Training feature data for SHAP values

        Returns:
        None: Displays the SHAP summary plot
        """

        # Create SHAP explainer and compute SHAP values
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_train)

        # Create the summary plot
        shap.summary_plot(shap_values, X_train)

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
        # Read in data
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

        # Merge data
        df = test_nodes.merge(features, on="node_id")
        df.index = test_nodes.index

        # Prepare features
        feature_vars = features.drop(columns="node_id").columns.to_list()
        feature_vars.extend(["temp", "pressure", "rh"])
        X_test_nodes = df[feature_vars].copy()
        X_test_nodes = X_test_nodes[non_zero_cols]
        X_test_nodes = scale_features(scaler, X_test_nodes)
        X_test_nodes = X_test_nodes[selected_features]

        # Define dependent variables
        y_test_nodes = df["co2"].copy()
        node_ids = df["node_id"].copy()

        # Create a DataFrame to store performance metrics for each node
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
        ) // num_cols  # Calculate the number of rows

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

        plt.tight_layout()
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
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "daily_avg_BEACO2N.csv"
        )
        all_valid_data = pd.read_csv(path)
        path = Path(__file__).resolve().parent.parent.parent / "Data" / "features.csv"
        features = pd.read_csv(path)

        all_valid_test = all_valid_data[
            all_valid_data.index.isin(full_X_train.index) == False
        ]

        df = all_valid_test.merge(features, on="node_id")
        df.index = all_valid_test.index

        node_eval_y = df["co2"].copy()
        node_ids = df["node_id"].copy()

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
                # Make predictions using the trained model
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

        # Create a color map for node categories
        color_map = {
            "imbalanced/invalid node": "grey",
            "training node": "blue",
            "central test node": "red",
            "fringe test node": "darkred",
        }

        # Create a colormap for R² values
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

        # Creating colorbar for R² values
        colorbar_html = linear._repr_html_().replace(
            "height: 20px;", "height: 40px; width: 80px; margin: 5px auto;"
        )

        # Creating legend for node categories and colorbar for R² values
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
            / "xgboost_node_map.html"
        )
        nodes_map.save(output_html_path)

        # Display the map
        return eval_metrics_df, nodes_map
