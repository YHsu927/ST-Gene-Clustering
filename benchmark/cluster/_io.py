# -*- coding: utf-8 -*-
# @Time : 2023/2/6 22:42
# @Author : Tory Deng
# @File : _io.py
# @Software: PyCharm
import os

import numpy as np
from loguru import logger
import pandas as pd


def write_clusters_as_cache(cl_data: pd.DataFrame, data_name: str, cl_method: str, run: int, k: int):
    clusters_dir = f"./cache/clustering_result/{data_name}/{cl_method}/"
    if not os.path.exists(clusters_dir):
        os.makedirs(clusters_dir)
    cl_data.to_csv(os.path.join(clusters_dir, f"{run_k}.csv"), index=False)
    logger.opt(colors=True).info(f"<magenta>{cl_method}</magenta> clustering results have been cached.")


def read_clusters_from_cache(data_name: str, cl_method: str, run: int, k: int):
    clusters_dir = f"./cache/clustering_result/{data_name}/{cl_method}/{run_k}.csv"
    if os.path.exists(clusters_dir):
        logger.opt(colors=True).info(f"Loading cached <magenta>{cl_method}</magenta> clustering results...")
        return pd.read_csv(clusters_dir)
    else:
        return None
