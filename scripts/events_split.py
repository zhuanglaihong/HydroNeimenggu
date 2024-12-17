"""
Author: silencesoup silencesoup@outlook.com
Date: 2024-12-14 13:27:00
LastEditors: silencesoup silencesoup@outlook.com
LastEditTime: 2024-12-14 13:38:54
FilePath: /HydroNeimeng/events.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import numpy as np
import pandas as pd
import re
from pint import UnitRegistry
from sklearn.model_selection import KFold
import xarray as xr
from hydrodatasource.cleaner.dmca_esr import rainfall_runoff_event_identify
from definitions import DATASET_DIR, PROJECT_DIR, RESULT_DIR
import xarray as xr
import os


def extract_number_and_unit(unit_str):
    """
    从字符串中提取数字和单位
    :param unit_str: 包含数字和单位的字符串，例如 '3h', '1D'
    :return: (数字, 单位) 的元组，例如 (3, 'h') 或 (1, 'D')
    """
    match = re.match(r"(\d+)([a-zA-Z]+)", unit_str)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        return number, unit
    return None, None  # 如果没有匹配到，返回 (None, None)


def get_rr_events(rain, flow, basin_name):

    if not (match := re.match(r"mm/(\d+)(h|d|D)", flow.units)):
        raise ValueError(f"Invalid unit format: {flow.units}")

    num, unit = match.groups()
    num = int(num)
    if unit == "h":
        multiple = num
    elif unit == "D":
        multiple = num * 24
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    print(f"flow.units = {flow.units}, multiple = {multiple}")

    rr_events = {}
    try:
        rr_event = rainfall_runoff_event_identify(
            rain.to_series(),
            flow.to_series(),
        )
    except Exception as e:
        print(f"Error processing {basin_name}: {e}")
        return None
    rr_events[basin_name] = rr_event

    return rr_events


def read_data_from_csv(csv_file_path, units):

    basename = os.path.basename(csv_file_path)
    basin_name = os.path.splitext(basename)[0]
    df = pd.read_csv(csv_file_path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # 去掉流量为空的点
    df_with_flow = df.dropna(subset=["streamflow"])

    rain = xr.DataArray(
        df_with_flow["total_precipitation_hourly"].values,
        dims="time",
        coords={"time": df_with_flow.index},
        attrs={"units": units, "basin": basin_name},  # 降雨单位为 mm/h
    )

    flow = xr.DataArray(
        df_with_flow["streamflow"].values,
        dims="time",
        coords={"time": df_with_flow.index},
        attrs={"units": units, "basin": basin_name},  # 流量单位为 mm/h
    )

    return rain, flow, basin_name


def split_events_based_on_time_units(basin_ids, time_unit="1h"):
    # 定义数据文件路径
    units = "mm/" + time_unit
    csv_folder_path = os.path.join(DATASET_DIR, "timeseries", time_unit)
    csv_file_names = os.listdir(csv_folder_path)
    selected_files = [
        csv_file_name
        for csv_file_name in csv_file_names
        if os.path.splitext(csv_file_name)[0] in basin_ids
        and csv_file_name.endswith(".csv")
    ]
    print(basin_ids)
    for csv_file_name in selected_files:
        basin_name = os.path.splitext(csv_file_name)[0]
        csv_file_path = os.path.join(csv_folder_path, csv_file_name)
        rain, flow, basin_name = read_data_from_csv(csv_file_path, units)
        if rain.size == 0:
            print(f"Skipping {csv_file_name}: no data")
            continue

        rr_events = get_rr_events(rain, flow, basin_name)
        if rr_events is None:
            continue

        all_events_df_list = []

        # rr_events 是一个字典
        for basin, events_df in rr_events.items():
            event_times_df = events_df[["BEGINNING_RAIN", "END_RAIN"]]
            event_times_df["BASIN"] = basin
            all_events_df_list.append(event_times_df)

        all_events_df = pd.concat(all_events_df_list, ignore_index=True)
        basin = event_times_df["BASIN"] = basin
        output_folder = os.path.join(RESULT_DIR, "events", basin)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, f"{basin}_{time_unit}_events.csv")
        print(f"Writing {output_file}")

        all_events_df.to_csv(output_file)


if __name__ == "__main__":
    basin_ids = pd.read_csv(
        os.path.join(PROJECT_DIR, "gage_ids/basin_neimenggu.csv"),
        dtype={"id": str},
    )["id"].values.tolist()
    split_events_based_on_time_units(basin_ids=basin_ids, time_unit="1D")
