import copy
import datetime as dt
import itertools
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import torch
import yaml
from rasterio.transform import xy
from shapely.geometry import MultiPoint, Point


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


class GridSearchList(list):
    pass


def gs(loader, node):
    seq = loader.construct_sequence(node)
    return GridSearchList(seq)


def do_not_parse(loader, node):
    return None


yaml.add_constructor("!join", join)
yaml.add_constructor("!gs", gs)
yaml.add_constructor("!do_not_parse", do_not_parse)


def load_params(path: str) -> Dict:
    with open(path, "r") as f:
        params = yaml.full_load(f)
    return params


def write_params(params: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(params, f)


Parameter = Tuple[str, Any]
ParameterCombination = List[Parameter]
ParameterPool = List[ParameterCombination]


def unpack_gs_parameters(params: Dict, prefix: str = None) -> ParameterPool:
    """
    Collects all grid search parameters in the parameter dict.
    Example:
    params = {
        'a': 1,
        'b: GridSearchList([1, 2]),
        'c': {
            'ca': [1, 2],
            'cb': GridSearchList([3, 4])
        }
    unpack_gs_parameters(params) = [
        [('c.cb', 3), ('c.cb', 4)],
        [('b', 1), ('b', 2)]
    ]

    Parameters
    ----------
    params Dict of parameters
    prefix Str only used recursively by this function

    Returns
    -------
    ParameterPool, i.e. List of parameter configurations, i.e. List of List of Tuple[ParameterName, ParameterValue]

    """
    gs_params = []
    for key, value in params.items():
        if isinstance(value, GridSearchList):
            if prefix is not None:
                key = ".".join([prefix, key])
            gs_params.append([(key, v) for v in value])
        elif isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_gs_parameters(value, prefix)
            if "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                gs_params.extend(param_pool)
        elif isinstance(value, Sequence) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_gs_parameters(v, prefix)
                    if "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        gs_params.extend(param_pool)
    return gs_params


def replace_list_by_value_in_params(params: Dict, keys: List[str], value: Any) -> None:
    """
    Replace the GridSearchLists in the parameter dict by the split values.
    Changes params dict in-place
    ----------
    params Dict of params
    keys List of str, nested dictionary keys
    value Value expanded from GridSearchList

    Returns
    -------
    None
    """
    node = params
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return params
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def expand_params(
    params: Dict, adjust_run_name: bool = False, run_name_key: str = "run"
) -> Tuple[List[Dict], List[Optional[ParameterCombination]]]:
    param_pool = unpack_gs_parameters(params)

    if not param_pool:
        return [params], [None]

    parameter_combinations: List[ParameterCombination] = []
    cv_params = []
    for parameter_combination in itertools.product(*param_pool):
        sub_params = copy.deepcopy(params)
        if adjust_run_name:
            name = sub_params[run_name_key]
        for nested_parameter_name, value in parameter_combination:
            replace_list_by_value_in_params(sub_params, nested_parameter_name.split("."), value)
            if adjust_run_name:
                name += "_" + nested_parameter_name + "_" + str(value)
        if adjust_run_name:
            sub_params[run_name_key] = name.replace(".args.", "_")
        cv_params.append(sub_params)
        parameter_combinations.append(parameter_combination)
    return cv_params, parameter_combinations


def calc_m_per_px(raster_meta: Dict) -> float:
    """Calculated the average raster resolution of given metadata in m/px.

    Args:
        raster_meta (Dict): Raster meta data.

    Returns:
        float: Average raster resolution in m/px.
    """
    # GPS coordinates of anchor point
    lon0, lat0 = xy(raster_meta["transform"], 0, 0)
    # calculate UTM zone
    utm_zone = int(np.floor((lon0 / 360) * 60 + 31))

    utm = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84")
    utm0_x, utm0_y = utm(*xy(raster_meta["transform"], 0, 0))
    utm1_x, utm1_y = utm(*xy(raster_meta["transform"], 0, 1))
    utm2_x, utm2_y = utm(*xy(raster_meta["transform"], 1, 0))

    # calculate unit pixel distances
    pxx = abs(utm1_x - utm0_x)
    pxy = abs(utm2_y - utm0_y)
    # take mean (assume quadratic pixels)
    m_per_px = np.mean([pxx, pxy])
    return m_per_px


def instantiate_module(root_module: ModuleType, config: Dict, kwargs: Optional[Dict] = None) -> object:
    """
    Instantiates a class object from a specified module within the root module,
    using configuration parameters provided.

    Args:
        root_module (ModuleType): The root module containing the desired module.
        config (Dict): A dictionary containing configuration parameters for
            instantiating the module's class. It should contain two keys:
            - "module": Name of the module within the root module.
            - "args": A dictionary containing keyword arguments required
                for initializing the class.
        kwargs (Optional[Dict]): Additional kwargs for the instance. Defaults to None.

    Returns:
        object: An instance of the class object instantiated with the provided
            configuration parameters.

    Raises:
        AttributeError: If the specified module or class does not exist.
        TypeError: If the provided module is not of type ModuleType or the configuration
            parameters are not correctly structured.
        KeyError: If the required keys "module" or "args" are missing in the config dictionary.
    """
    if config["args"]:
        args = config["args"]
    else:
        args = {}
    if not kwargs:
        kwargs = {}
    return getattr(root_module, config["module"])(**args, **kwargs)


def convert_to_pointz(geom):
    """
    Converts a MultiPoint geometry to a single Point Z by taking the first point's coordinates.

    Parameters:
        geom (MultiPoint or Geometry): The geometry to convert.

    Returns:
        Point: If geom is a MultiPoint, returns a Point Z object representing the first point.
        Geometry: If geom is not a MultiPoint, returns the original geometry unchanged.
    """
    if isinstance(geom, MultiPoint):
        return Point(geom.geoms[0].x, geom.geoms[0].y, geom.geoms[0].z)
    else:
        return geom


def nested_dict_to_device(
    nested_dict: Dict[str, Union[str, torch.Tensor]], device: Union[str, torch.device]
) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Recursively moves all torch.Tensor objects in a nested dictionary to the specified device.

    Parameters:
        nested_dict (Dict[str, Union[str, torch.Tensor]]): The dictionary containing the tensors to move.
        device (Union[str, torch.device]): The device to move the tensors to. Can be a string or torch.device.

    Returns:
        Dict[str, Union[str, torch.Tensor]]: The dictionary with all tensors moved to the specified device.
    """
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            nested_dict_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            nested_dict[key] = value.to(device)
    return nested_dict


def read_weather_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a TSV (Tab-Separated Values) file containing weather data into a pandas DataFrame.

    Parameters:
        path (Union[str, Path]): The file path to the TSV file.

    Returns:
        pd.DataFrame: A DataFrame where the 'Timestamp' column has been converted to a DatetimeIndex.
    """
    df = pd.read_csv(path)
    df.Timestamp = pd.DatetimeIndex(df.Timestamp)
    return df


def read_ds_annotation_json(path: Union[str, Path], fake_annotations: bool = True) -> gpd.GeoDataFrame:
    """
    Reads a GeoJSON file containing dataset annotations into a GeoDataFrame, processes it by renaming specified columns,
    converting them to numeric types, and optionally filtering out fake annotations.

    It specifically targets columns that start with '20', which are assumed to be date columns, renames them to a date format,
    and converts these columns to numeric type while treating 'None' values as NaN. Additionally, it removes any empty geometries and,
    if `fake_annotations` is set to False, removes rows where all the date columns have a value of -1.

    Parameters:
        path (Union[str, Path]): The file path to the GeoJSON file.
        fake_annotations (bool): A flag to indicate whether rows containing fake annotations (represented by -1 in date columns)
                                 should be included in the returned GeoDataFrame. Defaults to True, which includes these rows.

    Returns:
        gpd.GeoDataFrame: A processed GeoDataFrame with specified columns renamed and converted, and optionally filtered.
    """
    df = gpd.read_file(path)
    date_cols = {d: pd.to_datetime(d).date() for d in df.columns if d.startswith("20")}

    df.rename(columns=date_cols, inplace=True)

    # delete empty geometries
    df = df[~df.geometry.is_empty]

    # convert values to numerics
    for column in date_cols.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # optionally remove fake annotations (-1)
    if not fake_annotations:
        df = df[(df[date_cols.values()] != -1).any(axis=1)]

    return df
