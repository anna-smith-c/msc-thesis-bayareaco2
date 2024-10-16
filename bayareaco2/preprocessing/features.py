# Anna C. Smith
# GitHub username: edsml-acs223
# Imperial College London - MSc EDSML - IRP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import folium
from folium import GeoJson, GeoJsonPopup, FeatureGroup, LayerControl
from shapely.geometry import box
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
import warnings


def create_buffers(gdf, buffer_sizes):
    """
    Create buffers around node locations in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing the geometries to buffer.
    buffer_sizes (list of float): List of buffer sizes in meters.

    Returns:
    buffer_gdf (GeoDataFrame): A new GeoDataFrame with additional columns for each buffer geometry.
    """
    buffer_gdf = gdf.copy()
    for buffer_size in buffer_sizes:
        buffer_column_name = f"buffer_{buffer_size}m"
        buffer_gdf[buffer_column_name] = (
            buffer_gdf.geometry.to_crs(epsg=3310).buffer(buffer_size).to_crs(epsg=4326)
        )
    return buffer_gdf


def preprocess_gdf(gdf):
    """
    Preprocess a GeoDataFrame by ensuring it is in EPSG:4326 and filtering by bounding box.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame to preprocess.

    Returns:
    bay_area_gdf (GeoDataFrame): Preprocessed GeoDataFrame within the Bay Area bounding box.
    """
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    lat_min, lat_max = 37.50, 38.35
    lng_min, lng_max = -122.70, -121.95
    bounding_box = box(lng_min, lat_min, lng_max, lat_max)
    bay_area_gdf = gdf[gdf.intersects(bounding_box)]
    return bay_area_gdf


def plot_gdf(
    gdf, nodes_gdf=None, geometry="geometry", callout=None, feature_name="Base Layer"
):
    """
    Plot a GeoDataFrame on a Folium map with optional buffer layers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing feature data to plot.
    nodes_gdf (GeoDataFrame, optional): GeoDataFrame with buffer geometries to overlay.
    geometry (str, optional): Column name for the geometry in the gdf GeoDataFrame.
    callout (str, optional): Column name for the popup information.
    feature_name (str, optional): Name for the base layer on the map.

    Returns:
    map (folium.Map): A Folium map object with the plotted data.
    """
    map_center = [37.975, -122.3]
    map = folium.Map(location=map_center, zoom_start=10)

    def style_function(feature):
        return {
            "fillColor": "red",
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.5,
        }

    gdf = gdf.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in gdf.columns:
            if gdf[col].dtype == "geometry" and col != geometry:
                gdf[col] = gdf[col].astype(str)

    gdf.set_geometry(geometry, inplace=True)
    geojson_data = gdf.to_json()
    base_layer_name = feature_name
    base_layer = FeatureGroup(name=base_layer_name)

    geojson = GeoJson(
        geojson_data,
        style_function=style_function,
        popup=GeoJsonPopup(fields=[callout]) if callout else None,
    )

    geojson.add_to(base_layer)
    base_layer.add_to(map)

    # Create and add buffer layers
    if nodes_gdf is not None:
        for buffer_col in [
            "buffer_50m",
            "buffer_100m",
            "buffer_200m",
            "buffer_300m",
            "buffer_500m",
            "buffer_1000m",
            "buffer_2000m",
            "buffer_3000m",
            "buffer_4000m",
            "buffer_5000m",
        ]:
            layer_name = buffer_col
            nodes_layer = FeatureGroup(name=layer_name, show=False)

            nodes_gdf_buffer = nodes_gdf[["node_id", buffer_col]].copy()
            nodes_gdf_buffer = nodes_gdf_buffer.set_geometry(buffer_col)

            def node_style_function(feature):
                return {
                    "fillColor": "blue", 
                    "color": "blue",
                    "weight": 2,
                    "fillOpacity": 0.05,
                }

            node_geojson_data = nodes_gdf_buffer.to_json()
            node_geojson = GeoJson(
                node_geojson_data,
                style_function=node_style_function,
                popup=GeoJsonPopup(fields=["node_id"]),
            )

            node_geojson.add_to(nodes_layer)
            nodes_layer.add_to(map)

    # Add layer control to the map
    layer_control = LayerControl(collapsed=False)
    layer_control.add_to(map)

    return map


def plot_nodes_gdf(gdf):
    """
    Plot nodes from a GeoDataFrame on a Folium map with color-coded CircleMarkers.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing the node data to plot.

    Returns:
    nodes_map (folium.Map): A Folium map object with the plotted nodes and legend.
    """
    color_map = {
        "imbalanced/invalid node": "grey",
        "training node": "blue",
        "central test node": "red",
        "fringe test node": "darkred",
    }

    map_center = [37.955, -122.3]
    nodes_map = folium.Map(location=map_center, zoom_start=10)

    # Add CircleMarkers to the map
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=5,
            color=color_map.get(
                row["label"], "black"
            ),
            fill=False,
            fill_color=color_map.get(
                row["label"], "black"
            ),
            fill_opacity=0.5,
            popup=folium.Popup(
                f"<div style='width: 65px;'><b>node_id  </b> {row['node_id']}</div>"
            ),
        ).add_to(nodes_map)

    legend_html = """
        <div style="position: fixed; 
                    bottom: 10px; left: 60px; width: 240px; height: auto; 
                    border:2px solid grey; background-color:white;
                    z-index:9999; font-size:14px; padding:10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.5);
                    ">
        &nbsp; <b>Legend</b><br>
        &nbsp; <i class="fa fa-circle" style="color:grey"></i>&nbsp; Invalid / Imbalanced node<br>
        &nbsp; <i class="fa fa-circle" style="color:blue"></i>&nbsp; Training node<br>
        &nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; Central test node<br>
        &nbsp; <i class="fa fa-circle" style="color:darkred"></i>&nbsp; Fringe test node<br>
        </div>
    """

    # Add legend to the map
    folium.Marker(
        location=[gdf["lat"].mean(), gdf["lng"].mean()],
        icon=folium.DivIcon(html=legend_html),
    ).add_to(nodes_map)

    # Display the map
    return nodes_map


def land_use_buffer_extract(
    buffer_gdf, land_use_gdf, land_use_column="ucd_description"
):
    """
    Extract land use information within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    land_use_gdf (GeoDataFrame): GeoDataFrame with land use data.
    land_use_column (str): Column name for land use descriptions.

    Returns:
    result_df (pd.DataFrame): DataFrame with land use area sums for each buffer around nodes.
    """
    result = {}
    buffer_gdf = buffer_gdf.copy()
    original_node_ids = buffer_gdf["node_id"].unique()

    assert (
        buffer_gdf.crs == land_use_gdf.crs
    ), "CRS mismatch between node and land use data"

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)
            intersections = gpd.overlay(
                buffer_gdf[["node_id", buffer_column]], land_use_gdf, how="intersection"
            )
            intersections["intersection_area"] = intersections.to_crs(epsg=3310).area

            total_area = buffer_gdf.set_geometry(buffer_column).to_crs(epsg=3310).area
            total_area = total_area.rename("total_area")

            intersections = intersections.merge(
                total_area, left_on="node_id", right_index=True, how="left"
            )
            intersections["land_use_area"] = intersections["intersection_area"]
            land_use_areas = (
                intersections.groupby(["node_id", land_use_column])["land_use_area"]
                .sum()
                .unstack(fill_value=0)
            )
            land_use_areas.columns = [
                f'{label.replace(" ", "_")}_area_{buffer_column.split("_")[1]}'
                for label in land_use_areas.columns
            ]

            result[buffer_column] = land_use_areas

    result_df = pd.concat(result.values(), axis=1).reset_index()

    missing_node_ids = set(original_node_ids) - set(result_df["node_id"])
    if missing_node_ids:
        empty_rows = pd.DataFrame({"node_id": list(missing_node_ids)})
        empty_rows = empty_rows.set_index("node_id")
        empty_rows = empty_rows.reindex(
            columns=result_df.columns.difference(["node_id"]), fill_value=0
        ).reset_index()
        result_df = pd.concat([result_df, empty_rows], ignore_index=True)

    result_df.fillna(0, inplace=True)
    result_df.sort_values("node_id", inplace=True)

    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df


def read_and_crop_tiff(tiff_path):
    """
    Read a TIFF file and crop it to a specified Bay Area bounding box.

    Parameters:
    tiff_path (str): Path to the input TIFF file.

    Returns:
    cropped_data: Cropped data array and its corresponding transformation.
    new_transform: Affine transformation for the cropped data.
    """
    with rasterio.open(tiff_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Shape: {src.count} channels, {src.height} height, {src.width} width")

        transformer_to_src = Transformer.from_crs("EPSG:4326", src.crs)
        transformer_to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326")

        lat_min, lat_max = 37.50, 38.35
        lng_min, lng_max = -122.70, -121.95

        x_min, y_min = transformer_to_src.transform(lat_min, lng_min)
        x_max, y_max = transformer_to_src.transform(lat_max, lng_max)
        print(
            f"Transformed Bounding Box Coordinates (EPSG:32610): x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}"
        )

        window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
        cropped_data = src.read(window=window)
        print(f"Cropped Shape: {cropped_data.shape}")

        cropped_bounds = src.window_bounds(window)
        print(f"Cropped Data Bounds (EPSG:32610): {cropped_bounds}")

        new_transform = rasterio.transform.Affine.translation(
            x_min, y_max
        ) * rasterio.transform.Affine.scale(10, -10)
        print(f"Cropped Data Transform: {new_transform}")

        cropped_bounds_epsg4326 = transformer_to_wgs84.transform_bounds(
            cropped_bounds[0], cropped_bounds[1], cropped_bounds[2], cropped_bounds[3]
        )
        print(
            f"Cropped Data Bounds (EPSG:4326): lng_min={cropped_bounds_epsg4326[0]}, lat_min={cropped_bounds_epsg4326[1]}, lng_max={cropped_bounds_epsg4326[2]}, lat_max={cropped_bounds_epsg4326[3]}"
        )

        return cropped_data, new_transform


def plot_land_use_tif(data):
    """
    Plot ESRI Sentinel-2 10m LULC classification from a TIFF image.

    Parameters:
    data (ndarray): Array containing land use classification data.

    Returns:
    None: Displays the land use classification plot.
    """
    class_labels = [
        "Water",
        "Trees",
        "Flooded Vegetation",
        "Crops",
        "Built Area",
        "Bare Ground",
        "Snow/Ice",
        "Clouds",
        "Rangeland",
    ]
    colors = [
        "#419bdf",
        "#397d49",
        "#7a87c6",
        "#e49635",
        "#c4281b",
        "#a59b8f",
        "#a8ebff",
        "#616161",
        "#e3e2c3",
    ]
    class_int = [1, 2, 4, 5, 7, 8, 9, 10, 11]

    data = np.squeeze(data)
    cmap = mcolors.ListedColormap(colors)
    bounds = class_int + [12]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]

    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=ticks)
    cbar.ax.set_yticklabels(class_labels)
    cbar.ax.tick_params(size=0)
    cbar.ax.yaxis.set_ticks_position("right")
    plt.title("Bay Area Land Use (2021-2022)")
    plt.show()


def plot_land_use_gdf(gdf):
    """
    Plot ESRI Sentinel-2 10m LULC data from a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing land use data.

    Returns:
    None: Displays the land use classification plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    class_labels = [
        "Water",
        "Trees",
        "Flooded Vegetation",
        "Crops",
        "Built Area",
        "Bare Ground",
        "Snow/Ice",
        "Clouds",
        "Rangeland",
    ]
    colors = [
        "#419bdf",
        "#397d49",
        "#7a87c6",
        "#e49635",
        "#c4281b",
        "#a59b8f",
        "#a8ebff",
        "#616161",
        "#e3e2c3",
    ]

    color_dict = dict(zip(class_labels, colors))

    for label in class_labels:
        subset = gdf[gdf["Label"] == label]
        if not subset.empty:
            subset.plot(ax=ax, color=color_dict[label], label=label)

    handles = [
        plt.Line2D([0, 1], [0, 1], color=color_dict[label], linewidth=3)
        for label in class_labels
    ]
    ax.legend(handles, class_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Bay Area Land Use (2021-2022)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def sentinel_land_use_buffer_extract(
    buffer_gdf, land_use_gdf_dissolved, land_use_column="Label"
):
    """
    Extract land use information within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    land_use_gdf_dissolved (GeoDataFrame): GeoDataFrame with dissolved land use data.
    land_use_column (str): Column name for land use descriptions.

    Returns:
    result_df (pd.DataFrame): DataFrame with land use area sums for each buffer around nodes.
    """
    result = []
    assert (
        buffer_gdf.crs == land_use_gdf_dissolved.crs
    ), "CRS mismatch between node and land use data"

    unique_land_use_labels = land_use_gdf_dissolved[land_use_column].unique()

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)

            intersections = gpd.overlay(
                buffer_gdf[["node_id", buffer_column]],
                land_use_gdf_dissolved,
                how="intersection",
            )
            intersections = intersections.to_crs(epsg=3310)
            intersections["intersection_area"] = intersections.area

            intersecting_node_ids = intersections["node_id"].unique()
            all_node_ids = buffer_gdf["node_id"].unique()
            non_intersecting_node_ids = set(all_node_ids) - set(intersecting_node_ids)
            if non_intersecting_node_ids:
                print(
                    f"Warning: Sensors with node_ids {list(non_intersecting_node_ids)} do not fall within any land use polygon for buffer {buffer_column}."
                )

            land_use_areas = (
                intersections.groupby(["node_id", land_use_column])["intersection_area"]
                .sum()
                .unstack(fill_value=0)
            )

            for label in unique_land_use_labels:
                if label not in land_use_areas.columns:
                    land_use_areas[label] = 0

            land_use_areas = land_use_areas.rename(
                columns=lambda x: f'{x.replace(" ", "_")}_area_{buffer_column.split("_")[1]}'
            )
            result.append(land_use_areas)

    result_df = pd.concat(result, axis=1).reset_index()
    result_df.index.name = None

    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df


def aadt_buffer_extract(buffer_gdf, aadt_gdf, aadt_column="AADT"):
    """
    Extract AADT (Annual Average Daily Traffic) information within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    aadt_gdf (GeoDataFrame): GeoDataFrame with AADT data.
    aadt_column (str): Column name for AADT values.

    Returns:
    result_df (pd.DataFrame): DataFrame with total AADT for each buffer around nodes.
    """
    result = {}
    assert buffer_gdf.crs == aadt_gdf.crs, "CRS mismatch between node and aadt data"

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)
            intersections = gpd.overlay(
                buffer_gdf[["node_id", buffer_column]],
                aadt_gdf,
                how="intersection",
                keep_geom_type=False,
            )
            aadt_sum = intersections.groupby("node_id")[aadt_column].sum()
            result[f'total_AADT_{buffer_column.split("_")[1]}'] = aadt_sum

    result_df = pd.DataFrame(result).reset_index().fillna(0)
    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df


def road_len_buffer_extract(buffer_gdf, roads_gdf):
    """
    Extract total road length within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    roads_gdf (GeoDataFrame): GeoDataFrame with road geometries.

    Returns:
    result_df (pd.DataFrame): DataFrame with total road length for each buffer around nodes.
    """
    result = {}
    assert buffer_gdf.crs == roads_gdf.crs, "CRS mismatch between node and road data"

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)
            intersections = gpd.overlay(
                buffer_gdf[["node_id", buffer_column]],
                roads_gdf,
                how="intersection",
                keep_geom_type=False,
            )
            intersections["road_length"] = intersections.to_crs(epsg=3310).length
            total_road_length = intersections.groupby("node_id")["road_length"].sum()
            result[f'total_road_length_{buffer_column.split("_")[1]}'] = (
                total_road_length
            )

    result_df = pd.DataFrame(result).reset_index().fillna(0)
    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df


def get_bay_area_ndvi(band4_path, band5_path):
    """
    Calculate the NDVI (Normalized Difference Vegetation Index) for the Bay Area.

    Parameters:
    band4_path (str): Path to the Red band (Band 4) TIFF file.
    band5_path (str): Path to the NIR band (Band 5) TIFF file.

    Returns:
    ndvi: NDVI array and its corresponding.
    nir_transformation: affine transformation.
    """
    lat_min, lat_max = 37.50, 38.35
    lng_min, lng_max = -122.70, -121.95

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610")
    x_min, y_min = transformer.transform(lat_min, lng_min)
    x_max, y_max = transformer.transform(lat_max, lng_max)

    with rasterio.open(band4_path) as band4_src:
        window = from_bounds(x_min, y_min, x_max, y_max, band4_src.transform)
        red_band = band4_src.read(1, window=window).astype(float)
        red_transform = band4_src.window_transform(window)

    with rasterio.open(band5_path) as band5_src:
        window = from_bounds(x_min, y_min, x_max, y_max, band5_src.transform)
        nir_band = band5_src.read(1, window=window).astype(float)
        nir_transform = band5_src.window_transform(window)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        ndvi[np.isnan(ndvi)] = 0

    return ndvi, nir_transform


def ndvi_buffer_extract(buffer_gdf, ndvi, transform):
    """
    Extract average NDVI values within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    ndvi (ndarray): Array of NDVI values.
    transform (Affine): Affine transformation of the NDVI data.

    Returns:
    result_df (pd.DataFrame): DataFrame with average NDVI values for each buffer around nodes.
    """
    results = []

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)
            buffer_gdf = buffer_gdf.to_crs(epsg=32610)

            stats = zonal_stats(buffer_gdf, ndvi, affine=transform, stats=["mean"])
            avg_ndvi_values = [stat["mean"] for stat in stats]

            avg_ndvi_df = pd.DataFrame(
                {
                    "node_id": buffer_gdf["node_id"],
                    f'avg_ndvi_{buffer_column.split("_")[1]}': avg_ndvi_values,
                }
            )

            results.append(avg_ndvi_df)

    result_df = results[0]
    for df in results[1:]:
        result_df = result_df.merge(df, on="node_id")

    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df


def pop_dens_buffer_extract(
    buffer_gdf, population_density_gdf, pop_dens_column="Population density"
):
    """
    Extract average population density within buffers around nodes.

    Parameters:
    buffer_gdf (GeoDataFrame): GeoDataFrame with buffer geometries.
    population_density_gdf (GeoDataFrame): GeoDataFrame with population density data.
    pop_dens_column (str): Column name for population density values.

    Returns:
    result_df (pd.DataFrame): DataFrame with average population density for each buffer around nodes.
    """
    results = []

    assert (
        buffer_gdf.crs == population_density_gdf.crs
    ), "CRS mismatch between node and population density data"

    for buffer_column in buffer_gdf.columns:
        if buffer_column.startswith("buffer_"):
            buffer_gdf = buffer_gdf.set_geometry(buffer_column)

            intersections = gpd.overlay(
                buffer_gdf[["node_id", buffer_column]],
                population_density_gdf,
                how="intersection",
            )
            intersections["intersection_area_sq_meters"] = intersections.to_crs(
                epsg=3310
            ).area

            intersections["intersection_area_sq_kilometers"] = (
                intersections["intersection_area_sq_meters"] / 1_000_000
            )

            intersections["population_contribution"] = (
                intersections["intersection_area_sq_kilometers"]
                * intersections[pop_dens_column]
            )

            population_sums = intersections.groupby("node_id")[
                "population_contribution"
            ].sum()
            area_sums = intersections.groupby("node_id")[
                "intersection_area_sq_kilometers"
            ].sum()

            average_density = (population_sums / area_sums).rename(
                f'avg_pop_dens_{buffer_column.split("_")[1]}'
            )

            results.append(average_density)

    result_df = pd.concat(results, axis=1).reset_index()
    assert len(result_df) == len(
        buffer_gdf
    ), "Resulting DataFrame does not represent all sensor locations."
    return result_df
