"""
Author: Shuolong Xu
Date: 2024-04-17 12:55:24
LastEditTime: 2024-11-05 11:40:28
LastEditors: Wenyu Ouyang
Description:
FilePath: \torchhydro\experiments\train_with_era5land.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os.path
import pathlib

import pandas as pd
import sys
import hydrodatasource.configs.config as hdscc
import xarray as xr
import torch.multiprocessing as mp

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import train_worker
from torchhydro.trainers.trainer import train_and_evaluate

# Get the project directory of the py file
project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
from definitions import DATASET_DIR, RESULT_DIR

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

show = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent.parent, "gage_ids/basin_neimenggu.csv"),
    dtype={"id": str},
)
gage_id = show["id"].values.tolist()
# gage_id = ["songliao_21401550", "songliao_21401050"]
DEVICE = 0


def config():
    # 设置测试所需的项目名称和默认配置文件
    project_name = os.path.join(
        "test_with_era5land", "test_with_nmg_1D_era5land_mtlflowssm"
    )
    config_data = default_config_file()

    # 填充测试所需的命令行参数
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": DATASET_DIR,
            "other_settings": {
                "time_unit": ["1D"],
            },
        },
        ctx=[DEVICE],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 17,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 1,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=256,
        forecast_history=365,
        forecast_length=1,
        min_time_unit="D",
        min_time_interval=1,
        var_t=[
            # "precipitationCal",
            "total_precipitation_hourly",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        sampler="BasinBatchSampler",
        scaler="DapengScaler",
        train_epoch=100,
        save_epoch=1,
        # train_period=["2015-06-01", "2022-06-01"],
        test_period=["2019-06-01", "2020-12-01"],
        train_mode=False,
        stat_dict_file=os.path.join(
            RESULT_DIR,
            "train_with_camels_1D_era5land_mtlflowssm",
            "dapengscaler_stat.json",
        ),
        # valid_period=["2022-06-01", "2023-12-01"],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [DEVICE],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="batch",
        calc_metrics=False,
        early_stopping=True,
        rolling=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        model_type="MTL",
        weight_path=os.path.join(
            RESULT_DIR, "train_with_camels_1D_era5land_mtlflowssm", "best_model.pth"
        ),
        continue_train=False,
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


configs = config()
train_and_evaluate(configs)
