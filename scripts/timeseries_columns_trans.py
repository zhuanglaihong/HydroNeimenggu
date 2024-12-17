"""
Author: silencesoup silencesoup@outlook.com
Date: 2024-12-12 11:51:27
LastEditors: silencesoup silencesoup@outlook.com
LastEditTime: 2024-12-12 11:51:31
FilePath: /HydroNeimeng/scripts/timeseries_columns_trans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import pandas as pd
import os

# 指定文件夹路径
folder_path = r"C:\Programming\test\test_neimenggu_csv"

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 重命名列
        if "surface_net_solar_radiation_hourly" in df.columns:
            df = df.rename(
                columns={
                    "surface_net_solar_radiation_hourly": "surface_net_solar_radiation"
                }
            )

        # 删除指定的列
        columns_to_drop = [
            "node1_flow(m^3/s)",
            "pet(mm/day)",
            "et(mm/day)",
            "prcp(mm/day)",
        ]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )

        # 保存修改后的文件，文件名不变
        df.to_csv(file_path, index=False)
