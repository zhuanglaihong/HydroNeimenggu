import pathlib
from matplotlib.font_manager import FontProperties
import xarray as xr
import matplotlib.pyplot as plt
from definitions import PROJECT_DIR, DATASET_DIR, RESULT_DIR
from torchhydro import CACHE_DIR
import os
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import rcParams

user_home = os.path.expanduser("~")
font_path = os.path.join(user_home, ".fonts/SimHei.ttf")


def plot_precip_flow(
    basin_info,
    output_folder,
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
    font_prop = FontProperties(fname=font_path)
    rcParams["font.family"] = font_prop.get_name()
    rcParams["axes.unicode_minus"] = False
    if time_style == "3h":
        time_end = pd.to_datetime(time_end) + pd.Timedelta(hours=1)
        time_start = pd.to_datetime(time_start) + pd.Timedelta(hours=1)
    else:
        time_start = pd.to_datetime(time_start)
        time_end = pd.to_datetime(time_end)
    try:
        ds = xr.open_dataset(nc_file)

        # 确保输出文件夹存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 检查是否存在指定的basin_columns
        if target_basin_id in ds[basin_columns].values:
            # print(f"Processing {nc_file} with basin_id {target_basin_id}")

            # 提取指定basin_id的数据
            basin_data = ds.sel({basin_columns: target_basin_id})
            flow_obs = xr.open_dataset(flow_var_obs).sel(
                {basin_columns: target_basin_id}
            )["streamflow"]

            flow_pred = xr.open_dataset(flow_var_pred).sel(
                {basin_columns: target_basin_id}
            )["streamflow"]

            # 如果有时间范围，进行时间筛选
            if time_start and time_end:
                try:
                    # 获取 flow_obs 和 flow_pred 的时间范围
                    flow_obs_time_range = flow_obs.time
                    flow_pred_time_range = flow_pred.time

                    # 确定 flow_obs 和 flow_pred 时间范围的交集
                    flow_time_start = max(
                        time_start,
                        flow_obs_time_range.min().values,
                        flow_pred_time_range.min().values,
                    )
                    flow_time_end = min(
                        time_end,
                        flow_obs_time_range.max().values,
                        flow_pred_time_range.max().values,
                    )

                    # 使用调整后的时间范围选择 basin_data
                    basin_data = basin_data.sel(
                        time=slice(flow_time_start, flow_time_end)
                    )

                except Exception as e:
                    print(f"Error: {e}")
                    return None

            # 提取时间序列数据
            time = basin_data["time"]
            precip = basin_data[precip_var]

            # 读取流量预测和实测数据
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

            # 如果有时间范围，进行时间筛选

            station_dict = basin_info.set_index("basin_id")[
                ["name", "basin_area"]
            ].to_dict(orient="index")
            # 字典获取流域
            if target_basin_id in station_dict:

                basin_info = station_dict[target_basin_id]

                target_basin_id = basin_info["name"]

                basin_area = basin_info["basin_area"]
            else:

                print(f"{target_basin_id} not found in station_dict")

            # 创建图表
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 降水图（柱状图，宽度调整为0.5）
            ax1.bar(
                time, precip, width=0.1, color="blue", alpha=0.6, label="Precipitation"
            )
            ax1.set_ylabel("降雨值 (mm)", color="blue", fontproperties=font_prop)
            ax1.tick_params(axis="y", labelcolor="blue")

            # 只显示最大降水量的1/3
            ax1.set_ylim(0, precip.max() * 5)
            ax1.invert_yaxis()  # 降水量图表倒置显示

            # 添加第二个y轴用于流量图
            ax2 = ax1.twinx()

            # 单位转化
            if time_style == "1D":
                flow_obs_hourly = flow_obs / 24
                flow_obs = flow_obs_hourly * basin_area / 3.6
                flow_pred_hourly = flow_pred / 24
                flow_pred = flow_pred_hourly * basin_area / 3.6

            elif time_style == "3h":
                flow_obs_hourly = flow_obs / 3
                flow_obs = flow_obs_hourly * basin_area / 3.6
                flow_pred_hourly = flow_pred / 3
                flow_pred = flow_pred_hourly * basin_area / 3.6

            else:  # time_style == '1h'
                flow_obs = flow_obs * basin_area / 3.6
                flow_pred = flow_pred * basin_area / 3.6

            ax2.plot(
                time,
                flow_obs,
                color="green",
                linestyle="-",
                label="观测值",
            )
            ax2.plot(
                time,
                flow_pred,
                color="red",
                linestyle="--",
                label="预测值",
            )
            ax2.set_ylabel("径流值 (m^3/s)", color="red", fontproperties=font_prop)
            ax2.tick_params(axis="y", labelcolor="red")

            # 设置标题和图例
            plt.title(
                f"{target_basin_id}水文站 降雨与径流时序图", fontproperties=font_prop
            )

            plt.legend(loc="upper left")

            plt.savefig(
                f"{output_folder}/{target_basin_id}_{time_start}-{time_end}.png"
            )

            # 关闭数据集
            ds.close()
        else:
            print(f"{target_basin_id} not found in {nc_file}")
    except Exception as e:
        print(f"An error occurred  {e}")


# 读取场次数据
def read_rainfall_events_summary(csv_file_path):
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


# 筛选要画图的nc文件
def get_nc_files(target_basin_id, time_unit):

    for file_name in os.listdir(CACHE_DIR):
        if file_name.endswith(".nc") and time_unit in file_name:  # 只处理 .nc 文件
            file_path = os.path.join(CACHE_DIR, file_name)

            try:
                # 使用 xarray 打开 .nc 文件
                dataset = xr.open_dataset(file_path)

                # 检查文件中是否包含 basin_id
                if "basin" in dataset:
                    # 获取 basin_id 数据并检查是否包含目标 basin_id
                    basin_ids = dataset["basin"].values

                    if target_basin_id in basin_ids:
                        print(f"Found basin {target_basin_id} in file: {file_name}")
                        dataset.close()
                        return file_path
                    else:
                        dataset.close()
                        # print(f"basin {target_basin_id} not found in file: {file_name}")
                else:
                    dataset.close()
                    # print(f"basin not found in file: {file_name}")

            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return None


def plot_based_on_events(time_unit, project_name):
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
            continue
        events_path = os.path.join(
            events_folder_path, basin_id, f"{basin_id}_1D_events.csv"
        )
        events_dict = read_rainfall_events_summary(events_path)
        for event in events_dict[basin_id]:
            start_time = event["Start_Time"]
            end_time = event["End_Time"]

            plot_precip_flow(
                basin_info,
                os.path.join(RESULT_DIR, "events", basin_id, project_name),
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


if __name__ == "__main__":
    for folder_name in os.listdir(RESULT_DIR):
        if folder_name.startswith("test_with_") and "1D" in folder_name:
            print("plotting 1D")
            plot_based_on_events("1D", folder_name)
        elif folder_name.startswith("test_with_") and "3h" in folder_name:
            print("plotting 3h")
            plot_based_on_events("3h", folder_name)
