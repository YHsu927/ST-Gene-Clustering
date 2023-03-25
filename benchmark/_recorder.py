# -*- coding: utf-8 -*-
# @Time : 2023/2/7 1:03
# @Author : Tory Deng
# @File : _recorder.py
# @Software: PyCharm
import os
from datetime import datetime
from itertools import product
from typing import Dict, Union, List, Literal, Callable

import numpy as np
import pandas as pd
from loguru import logger


def create_records(
        data_cfg: Dict[str, Dict[str, Union[os.PathLike, str, List, Dict[str, Union[List, str]]]]],
        cl_cfg: Dict[str, int],
        metrics: List[Literal['DB-Index(Pearson)','DB-Index(Spatial)','Average_Negative_Log_Minimum_P-Value','Average_Negative_Log_Minimum_P-Value_Fast']] 
):

    row_tuples = [
        tup
        for cl_method, n_runs in cl_cfg.items()
        for tup in product(data_cfg.keys(), (cl_method.__name__ if callable(cl_method) else cl_method,), range(n_runs))
    ]
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['dataset', 'clustering_method', 'run'])
    single_record = pd.DataFrame(
        np.full(len(row_tuples), fill_value=np.nan, dtype=float), index=row_index, columns=['values']
    )
    return {metric: single_record.copy() for metric in metrics}


def store_metrics_to_records(
        records: Dict[str, pd.DataFrame],
        metric: Literal['DB-Index(Pearson)','DB-Index(Spatial)','Average_Negative_Log_Minimum_P-Value','Average_Negative_Log_Minimum_P-Value_Fast'], 
        value: float,
        data_name: str,
        cl_method: Union[str, Callable],
        run: int
):
    if callable(cl_method):
        cl_method = cl_method.__name__
    records[metric].loc[data_name, cl_method, run] = value


def write_records(records: Dict[str, pd.DataFrame]):
    record_name = f"{datetime.now().strftime('%Y-%m %H_%M_%S')}"
    writer = pd.ExcelWriter(f'{record_name}.xlsx')
    for metric, record in records.items():
        record.to_excel(writer, sheet_name=metric, index=True)
    writer.close()
    logger.info(f"records have been saved into './{record_name}.xlsx'.")
