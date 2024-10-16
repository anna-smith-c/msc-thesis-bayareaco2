# Anna C. Smith
# GitHub username: edsml-acs223
# Imperial College London - MSc EDSML - IRP

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Flatten,
    Dense,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.losses import MeanSquaredError
from livelossplot import PlotLossesKeras
import pandas as pd
import folium
import branca.colormap as cm
from pathlib import Path
from bayareaco2.models.data import scale_features


def reshape_1D_X(X):
    """
    Reshapes a 2D array into a 3D array suitable for input into a 1D CNN.

    Parameters:
    X (pd.DataFrame): The 2D array of features to be reshaped.

    Returns:
    np.ndarray: A 3D array with shape (samples, timesteps, features),
    """
    return X.values.reshape(-1, X.shape[1], 1)


class CNN_1D:
    def __init__(self, input_shape, verbose=True):
        """
        Initializes a 1D Convolutional Neural Network (CNN) model.

        Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).
        verbose (bool): Whether to print the model summary.
        """
        self.verbose = verbose
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        """
        Builds the architecture of the 1D CNN model.

        Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).

        Returns:
        keras.models.Sequential: The constructed 1D CNN model.
        """
        # Define the model architecture using best structure and hyperparameters
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(64, 3, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.2),
                Conv1D(128, 3, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1),  # Output layer for regression
            ]
        )

        # print model summary if verbose is True
        if self.verbose:
            model.summary()

        return model

    def compile(self, learning_rate=0.001):
        """
        Compiles the 1D CNN model with the specified optimizer and loss function.

        Parameters:
        learning_rate (float): The learning rate for the Adam optimizer.
        """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=MeanSquaredError())

    def fit(
        self,
        X_train,
        y_train,
        filepath,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        plot_loss=True,
        verbose=2,
    ):
        """
        Trains the 1D CNN model on the provided training data.

        Parameters:
        X_train (np.ndarray): The training features
        y_train (np.ndarray): The training labels
        filepath (str): The path to save the best model during training
        epochs (int): Number of epochs to train the model
        batch_size (int): The size of the batches during training
        validation_split (float): The fraction of data to be used for validation
        plot_loss (bool): Whether to plot the training and validation loss during training
        verbose (int): Verbosity mode

        Returns:
        keras.callbacks.History: A record of training loss values and metrics values
        """
        # Early stopping callbak to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        # Reduce learning rate on plateau to improve convergence
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
        )
        # Save the best model during training
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True)
        # Add callbacks to list
        callbacks = [early_stopping, reduce_lr, checkpoint]

        # Add livelossplot callback if plot_loss is True
        if plot_loss:
            plot_losses = PlotLossesKeras()
            callbacks.append(plot_losses)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    def predict(self, X_test):
        """
        Predicts the target values for the provided test data.

        Parameters:
        X_test (np.ndarray): The test features,

        Returns:
        np.ndarray: The predicted values.
        """
        return self.model.predict(X_test, verbose=0)

    def evaluate(self, X_test, y_test, print_results=True):
        """
        Evaluates the model's performance on the test data and optionally prints the results.

        Parameters:
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The true target values for the test set.
        print_results (bool): Whether to print the evaluation metrics.

        Returns:
        dict: A dictionary containing R², MSE, RMSE, and MAE metrics.
        """
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        results = {"R²": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

        if print_results:
            # Print evaluation metrics
            print(f"R²: {r2:.2f}")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")

            self.plot_results(y_test, y_pred, title="1D CNN")

            return results

        else:
            return results

    def plot_results(self, y_test, y_pred, title):
        """
        Plots a scatter plot comparing the true and predicted values.

        Parameters:
        y_test (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
        title (str): The title for the plot.

        Returns:
        matplotlib.pyplot.figure: The scatter plot of true vs. predicted values.
        """
        plt.figure(figsize=(7, 7))

        # Scatter plot of true vs. predicted values
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
        plt.tight_layout()  # Adjusts subplot params for a cleaner layout
        plt.show()

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

        # Prepare features
        feature_vars = features.drop(columns="node_id").columns.to_list()
        feature_vars.extend(["temp", "pressure", "rh"])
        X_test_nodes = df[feature_vars].copy()
        X_test_nodes = X_test_nodes[non_zero_cols]
        X_test_nodes = scale_features(scaler, X_test_nodes)
        X_test_nodes = X_test_nodes[selected_features]

        # Reshape for CNN
        X_test_nodes = reshape_1D_X(X_test_nodes)

        # Define dependent variables
        y_test_nodes = df["co2"].copy().values
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

            y_pred = self.predict(
                X_node
            ).flatten()  # Flatten to match the shape of y_test_nodes

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
        num_rows = (num_nodes + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
        )
        axes = axes.flatten()

        for i, node_id in enumerate(metrics_df.columns):
            ax = axes[i]
            y_test = y_test_nodes[node_ids == node_id]
            X_node = X_test_nodes[node_ids == node_id]
            y_pred = self.predict(
                X_node
            ).flatten()  # Flatten to match the shape of y_test

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
        # Load necessary data
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "daily_avg_BEACO2N.csv"
        )
        all_valid_data = pd.read_csv(path)
        path = Path(__file__).resolve().parent.parent.parent / "Data" / "features.csv"
        features = pd.read_csv(path)

        # Filter for test data by excluding training indices
        all_valid_test = all_valid_data[~all_valid_data.index.isin(full_X_train.index)]
        df = all_valid_test.merge(features, on="node_id")
        df.index = all_valid_test.index  # Retain the original index

        # Define dependent variables and node IDs
        node_eval_y = df["co2"].copy()
        node_ids = df["node_id"].copy()

        # Prepare features
        feature_vars = features.drop(columns="node_id").columns.to_list()
        feature_vars.extend(["temp", "pressure", "rh"])
        node_eval_X = df[feature_vars].copy()
        node_eval_X = node_eval_X[non_zero_cols]
        node_eval_X = scale_features(scaler, node_eval_X)
        node_eval_X = node_eval_X[selected_features]

        # Reshape features while preserving index information
        X_reshaped = reshape_1D_X(node_eval_X)

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
            X_node = X_reshaped[node_filter]
            y_node = node_eval_y[node_filter]
            n = len(y_node)
            eval_metrics["n"].append(n)

            if not X_node.size == 0 and not y_node.size == 0:
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
        eval_metrics_df = (
            pd.DataFrame(eval_metrics).set_index("node_id").round(2).reset_index()
        )

        # Merge with node locations
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "Data"
            / "CO2"
            / "node_locations.csv"
        )
        node_locations = pd.read_csv(path)
        node_results_locations = eval_metrics_df.merge(node_locations, on="node_id")

        # Define color map for node categories
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

        # Create HTML for R² colorbar
        colorbar_html = linear._repr_html_().replace(
            "height: 20px;", "height: 40px; width: 80px; margin: 5px auto;"
        )

        # Create legend HTML for node categories and R² colorbar
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
            / "cnn_node_map.html"
        )
        nodes_map.save(output_html_path)

        # Display the map
        return eval_metrics_df, nodes_map
