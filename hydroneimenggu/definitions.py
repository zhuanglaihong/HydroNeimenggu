"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:03:43
LastEditTime: 2024-12-12 10:25:38
LastEditors: Wenyu Ouyang
Description: some global variables used in this project
FilePath: /HydroNeimeng/definitions.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# NOTE: create a file in root directory -- definitions_private.py,
# then copy the code after 'except ImportError:' to definitions_private.py
# and modify the paths as your own paths in definitions_private.py
import os

from torchhydro import SETTING

try:
    import definitions_private

    PROJECT_DIR = definitions_private.PROJECT_DIR
    RESULT_DIR = definitions_private.RESULT_DIR
    DATASET_DIR = definitions_private.DATASET_DIR
except ImportError:
    PROJECT_DIR = os.getcwd()
    RESULT_DIR = "/mnt/disk1/owen/code/HydroNeimeng/"
    DATASET_DIR = SETTING["local_data_path"]["basins-interim"]
