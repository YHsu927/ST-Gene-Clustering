# -*- coding: utf-8 -*-
# @Time : 2023/2/6 13:43
# @Author : Tory Deng
# @File : run_benchmark.py
# @Software: PyCharm
import os
from typing import List, Dict, Union, Literal, Callable, Optional

from loguru import logger

from ._metrics import compute_gene_clustering_metrics
from ._recorder import create_records, write_records, store_metrics_to_records
from ._utils import rm_cache, set_logger
from .cluster import generally_cluster_obs
from .dataset import load_data


@logger.catch
def run_bench(
        data_cfg: Dict[str, Dict[str, Union[os.PathLike, str, List, Dict[str, Union[List, str]]]]],
        cl_cfg: Dict[str, int],
        metrics: List[Literal['DB-Index(Pearson)','DB-Index(Spatial)','Average_Negative_Log_Minimum_P-Value','Average_Negative_Log_Minimum_P-Value_Fast']],
        species: Literal['mouse', 'human'],
        cl_kwarg: Optional[Dict] = None,
        preprocess: bool = True,
        clean_cache: bool = False,
        verbosity: Literal[0, 1, 2] = 2,
        log_path: Optional[Union[os.PathLike, str]] = None,
        random_state: int = 0, 
        k: int = 0
):
    set_logger(verbosity, log_path)
    if cl_kwarg is None:
        cl_kwarg = dict()
    if clean_cache:
        rm_cache("./cache")
    records = create_records(data_cfg, cl_cfg, metrics)

    for data_name, data_props in data_cfg.items():
        adata, img = load_data(data_name, data_props, preprocess, denoise=False)
        
        for cl_method, n_runs in cl_cfg.items():
            for run in range(n_runs):
                cl_data = generally_cluster_obs(adata, img, species, cl_method, random_state, run, k, **cl_kwarg)
                for metric in metrics:
                    value = compute_gene_clustering_metrics(cl_data, adata, metric, species)
                    store_metrics_to_records(records, metric, value, data_name, cl_method, run)
        write_records(records)
