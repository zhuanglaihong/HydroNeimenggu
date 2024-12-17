import pathlib
import re
from matplotlib.font_manager import FontProperties
import xarray as xr
import matplotlib.pyplot as plt
from definitions import PROJECT_DIR, DATASET_DIR, RESULT_DIR
from torchhydro import CACHE_DIR
import os
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import rcParams
import geopandas as gpd

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
            )["sm_surface"]

            flow_pred = xr.open_dataset(flow_var_pred).sel(
                {basin_columns: target_basin_id}
            )["sm_surface"]

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
                .sel(time=slice(flow_time_start, flow_time_end))["sm_surface"]
            )
            flow_pred = (
                xr.open_dataset(flow_var_pred)
                .sel({basin_columns: target_basin_id})
                .sel(time=slice(flow_time_start, flow_time_end))["sm_surface"]
            )

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
            ax2.set_ylabel(
                "土壤含水量（m^3/m^3）", color="red", fontproperties=font_prop
            )
            ax2.tick_params(axis="y", labelcolor="red")

            # 设置标题和图例
            plt.title(
                f"{target_basin_id}水文站 降雨与土壤含水量时序图",
                fontproperties=font_prop,
            )

            plt.legend(loc="upper left")

            plt.savefig(f"{output_folder}/{target_basin_id}_sm_surface.png")

            # 关闭数据集
            ds.close()
        else:
            print(f"{target_basin_id} not found in {nc_file}")
    except Exception as e:
        print(f"An error occurred  {e}")


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


def plt_by_year(time_unit, project_name, year):
    year = str(year)
    output_folder = os.path.join(RESULT_DIR, "year", project_name, year)
    basin_colunms = "basin"
    precip_var = "total_precipitation_hourly"
    basin_ids = gpd.read_file(os.path.join(DATASET_DIR, "shapes", "basins.shp"))[
        "BASIN_ID"
    ].values.tolist()
    basins_with_no_data = [item for item in basin_ids if item.startswith("neimeng")]
    flow_var_obs = os.path.join(
        RESULT_DIR, project_name, "epochbest_model.pthflow_obs.nc"
    )
    flow_var_pred = os.path.join(
        RESULT_DIR, project_name, "epochbest_model.pthflow_pred.nc"
    )
    basin_info = pd.read_csv(
        os.path.join(pathlib.Path(__file__).parent.parent, "gage_ids/basin_info.csv")
    )

    for basin_id in basins_with_no_data:
        nc_file = get_nc_files(basin_id, time_unit)
        plot_precip_flow(
            basin_info,
            output_folder,
            nc_file,
            basin_colunms,
            precip_var,
            flow_var_obs,
            flow_var_pred,
            basin_id,
            time_unit,
            time_start=f"{year}-01-01",
            time_end=f"{year}-10-31",
        )


if __name__ == "__main__":
    for folder_name in os.listdir(RESULT_DIR):
        # 正则表达式匹配四个连续数字结尾的文件夹
        match = re.search(r"(\d{4})$", folder_name)

        if match:
            # 如果匹配到了四个数字结尾，提取年份并跳出循环
            year = int(match.group(1))
            print(f"Year extracted: {year}")

            if "1D" in folder_name and "mtl" in folder_name:
                print("plotting 1D")
                plt_by_year("1D", folder_name, year)
            elif "3h" in folder_name and "mtl" in folder_name:
                print("plotting 3h")
                plt_by_year("3h", folder_name, year)

            # 运行完后跳出循环
            continue

        # 如果没有匹配到年份，继续检查下一个文件夹
        for year in range(2015, 2021):
            if (
                folder_name.startswith("test_with_")
                and "1D" in folder_name
                and "mtl" in folder_name
            ):
                print("plotting 1D")
                plt_by_year("1D", folder_name, year)
            elif (
                folder_name.startswith("test_with_")
                and "3h" in folder_name
                and "mtl" in folder_name
            ):
                print("plotting 3h")
                plt_by_year("3h", folder_name, year)
