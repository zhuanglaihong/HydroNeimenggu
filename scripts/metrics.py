import pathlib
import xarray as xr
from definitions import PROJECT_DIR, DATASET_DIR, RESULT_DIR
from torchhydro import CACHE_DIR
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def compute_flow_metrics(
    basin_info,
    nc_file,
    basin_columns,
    precip_var,
    flow_var_obs,
    flow_var_pred,
    target_basin_id,
    time_style,
    time_start=None,
    time_end=None,
    station_dict=None,
):
    """
    计算指定流域在特定时间范围内的流量指标，包括RMSE、相关系数、NSE和径流系数。
    """
    if time_style == "3h":
        time_end = pd.to_datetime(time_end) + pd.Timedelta(hours=1)
        time_start = pd.to_datetime(time_start) + pd.Timedelta(hours=1)
    else:
        time_start = pd.to_datetime(time_start)
        time_end = pd.to_datetime(time_end)
    try:
        ds = xr.open_dataset(nc_file)

        # 检查目标流域ID是否存在于数据集中
        if target_basin_id in ds[basin_columns].values:
            # 选择指定流域ID的数据
            basin_data = ds.sel({basin_columns: target_basin_id})
            flow_obs = xr.open_dataset(flow_var_obs).sel(
                {basin_columns: target_basin_id}
            )["streamflow"]

            flow_pred = xr.open_dataset(flow_var_pred).sel(
                {basin_columns: target_basin_id}
            )["streamflow"]

            # 应用时间范围过滤（如果指定）
            if time_start and time_end:
                try:
                    flow_obs_time_range = flow_obs.time
                    flow_pred_time_range = flow_pred.time

                    # 确定观测和预测流量的时间交集
                    flow_time_start = max(
                        time_start,
                        pd.to_datetime(flow_obs_time_range.min().values),
                        pd.to_datetime(flow_pred_time_range.min().values),
                    )
                    flow_time_end = min(
                        time_end,
                        pd.to_datetime(flow_obs_time_range.max().values),
                        pd.to_datetime(flow_pred_time_range.max().values),
                    )

                    # 选择在调整后的时间范围内的数据
                    basin_data = basin_data.sel(
                        time=slice(flow_time_start, flow_time_end)
                    )

                    if basin_data.time.size == 0:
                        print(f"流域ID {target_basin_id} 在{nc_file}时间范围内没有数据")
                        return None

                except Exception as e:
                    print(f"时间切片错误: {e}")
                    return None

            # 提取时间序列数据
            time = basin_data["time"]
            precip = basin_data[precip_var]

            # 读取在时间范围内的观测和预测流量数据
            flow_obs = (
                xr.open_dataset(flow_var_obs)
                .sel({basin_columns: target_basin_id})
                .sel(time=slice(flow_time_start, flow_time_end))["streamflow"]
            )
            flow_pred = (
                xr.open_dataset(flow_var_pred)
                .sel({basin_columns: target_basin_id})
                .sel(time=slice(flow_time_start, flow_time_end))["streamflow"]
            )

            # 检索流域信息
            if target_basin_id in station_dict:
                basin_info_entry = station_dict[target_basin_id]
                basin_name = basin_info_entry["name"]
                basin_area = basin_info_entry["basin_area"]
            else:
                print(f"流域ID {target_basin_id} 未在 station_dict 中找到")
                return None

            # 对齐观测和预测流量数据，确保时间坐标一致
            flow_obs_aligned, flow_pred_aligned = xr.align(
                flow_obs, flow_pred, join="inner"
            )
            precip_aligned = precip.sel(time=flow_obs_aligned.time)

            # 将对齐后的数据转换为NumPy数组
            flow_obs_values = flow_obs_aligned.values
            flow_pred_values = flow_pred_aligned.values
            precip_values = precip_aligned.values

            # 计算径流系数
            # 仅在 flow_obs 有值且降水不为零的时间点计算 flow_obs_coeff
            valid_indices = ~np.isnan(flow_obs_values) & ~np.isnan(precip_values)
            flow_obs_coeff_total = (
                np.sum(flow_obs_values[valid_indices])
                / np.sum(precip_values[valid_indices])
                if np.sum(precip_values[valid_indices]) != 0
                else np.nan
            )

            # 对于 flow_pred_coeff，可以在所有有效时间点计算
            valid_pred_indices = ~np.isnan(flow_pred_values) & ~np.isnan(precip_values)
            # valid_pred_indices = valid_indices
            flow_pred_coeff_total = (
                np.sum(flow_pred_values[valid_pred_indices])
                / np.sum(precip_values[valid_pred_indices])
                if np.sum(precip_values[valid_pred_indices]) != 0
                else np.nan
            )

            print(f"flow_obs_coeff_total: {flow_obs_coeff_total}")
            print(f"flow_pred_coeff_total: {flow_pred_coeff_total}")

            # 计算RMSE
            rmse = np.sqrt(np.mean((flow_pred_values - flow_obs_values) ** 2))

            # 计算皮尔逊相关系数
            if len(flow_obs_values) > 1:
                correlation = np.corrcoef(flow_obs_values, flow_pred_values)[0, 1]
            else:
                correlation = np.nan

            # 计算纳什-萨特克利夫效率系数 (NSE)
            numerator = np.sum((flow_obs_values - flow_pred_values) ** 2)
            denominator = np.sum((flow_obs_values - np.mean(flow_obs_values)) ** 2)
            if denominator != 0:
                nse = 1 - (numerator / denominator)
            else:
                nse = np.nan

            metrics = {
                "basin_id": target_basin_id,
                "basin_name": basin_name,
                "event_start": str(flow_time_start),
                "event_end": str(flow_time_end),
                "rmse": rmse,
                "correlation": correlation,
                "nse": nse,
                "flow_obs_coeff_total": flow_obs_coeff_total,
                "flow_pred_coeff_total": flow_pred_coeff_total,
            }

            # 关闭数据集
            ds.close()

            return metrics
        else:
            print(f"流域ID {target_basin_id} 未在文件 {nc_file} 中找到")
            return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def read_rainfall_events_summary(csv_file_path):
    """
    读取降雨事件总结CSV文件，并返回一个字典，键为流域ID，值为事件列表。
    """
    summary_df = pd.read_csv(csv_file_path)
    events_dict = {}
    for index, row in summary_df.iterrows():
        basin = row["BASIN"]
        start_time = row["BEGINNING_RAIN"]
        end_time = row["END_RAIN"]
        if basin not in events_dict:
            events_dict[basin] = []
        events_dict[basin].append({"Start_Time": start_time, "End_Time": end_time})
    return events_dict


def get_nc_files(target_basin_id, time_unit):
    """
    在CACHE_DIR目录中查找包含目标流域ID和时间单位的.nc文件。
    """
    for file_name in os.listdir(CACHE_DIR):
        if file_name.endswith(".nc") and time_unit in file_name:  # 仅处理 .nc 文件
            file_path = os.path.join(CACHE_DIR, file_name)
            try:
                dataset = xr.open_dataset(file_path)
                if "basin" in dataset:
                    basin_ids = dataset["basin"].values
                    if target_basin_id in basin_ids:
                        print(f"在文件 {file_name} 中找到流域ID {target_basin_id}")
                        dataset.close()
                        return file_path
                    else:
                        dataset.close()
                else:
                    dataset.close()
            except Exception as e:
                print(f"读取文件 {file_name} 时出错: {e}")
    return None


def compute_metrics_based_on_events(time_unit, project_name, metrics_list):
    """
    根据指定的时间单位和项目名称，计算所有流域和事件的流量指标，并将结果添加到metrics_list中。
    """
    events_folder_path = os.path.join(RESULT_DIR, "events")
    basin_ids = [
        folder
        for folder in os.listdir(events_folder_path)
        if os.path.isdir(os.path.join(events_folder_path, folder))
    ]
    basin_info = pd.read_csv(
        os.path.join(pathlib.Path(__file__).parent.parent, "gage_ids/basin_info.csv")
    )
    station_dict = basin_info.set_index("basin_id")[["name", "basin_area"]].to_dict(
        orient="index"
    )
    for basin_id in basin_ids:
        nc_file = get_nc_files(basin_id, time_unit)
        if nc_file is None:
            print(f"未找到流域ID {basin_id} 的 .nc 文件")
            continue
        events_path = os.path.join(
            events_folder_path, basin_id, f"{basin_id}_1D_events.csv"
        )
        if not os.path.exists(events_path):
            print(f"事件文件 {events_path} 不存在")
            continue
        events_dict = read_rainfall_events_summary(events_path)
        for event in events_dict.get(basin_id, []):
            start_time = event["Start_Time"]
            end_time = event["End_Time"]

            metrics = compute_flow_metrics(
                basin_info,
                nc_file,
                "basin",
                "total_precipitation_hourly",
                os.path.join(
                    RESULT_DIR, project_name, "epochbest_model.pthflow_obs.nc"
                ),
                os.path.join(
                    RESULT_DIR, project_name, "epochbest_model.pthflow_pred.nc"
                ),
                basin_id,
                time_unit,
                start_time,
                end_time,
                station_dict,
            )
            if metrics:
                metrics_list.append(metrics)


def main():
    """
    主函数，遍历RESULT_DIR中的所有项目文件夹，根据时间单位计算流量指标，并为每个项目生成单独的CSV文件。
    """
    for folder_name in os.listdir(RESULT_DIR):
        folder_path = os.path.join(RESULT_DIR, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("test_with_"):
            # 初始化一个字典来存储不同时间单位的指标
            project_metrics = {}
            for time_unit in ["1D", "3h"]:
                if time_unit in folder_name:
                    print(f"正在处理项目 {folder_name} 的{time_unit}指标")
                    metrics_list = []
                    compute_metrics_based_on_events(
                        time_unit, folder_name, metrics_list
                    )
                    project_metrics[time_unit] = metrics_list
            # 为每个时间单位保存一个单独的CSV文件
            for time_unit, metrics in project_metrics.items():
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    output_folder = os.path.join(
                        RESULT_DIR, "flow_metrics", folder_name
                    )
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_csv = os.path.join(
                        output_folder, f"{folder_name}_{time_unit}_flow_metrics.csv"
                    )
                    metrics_df.to_csv(output_csv, index=False)
                    print(f"流量指标已保存到 {output_csv}")
                else:
                    print(f"项目 {folder_name} 的{time_unit}没有可保存的指标数据。")


if __name__ == "__main__":
    main()
