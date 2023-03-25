# -*- coding: utf-8 -*-
# @Time : 2023/2/7 19:17
# @Author : Tory Deng
# @File : _general.py
# @Software: PyCharm
from typing import Union, Callable, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd

from ._io import read_clusters_from_cache, write_clusters_as_cache
from .spatial.functions import cluster_genes


def generally_cluster_obs(
        adata: ad.AnnData,
        img: Optional[np.ndarray],
        species: Literal['mouse', 'human'],
        cl_method: Union[str, Callable],
        random_state: int,
        run: int,
        k: int,
        **kwargs
):
    data_name = adata.uns['data_name']
    if isinstance(cl_method, str):
        # load cached clustering results or run clustering
        cl_data = read_clusters_from_cache(data_name, cl_method, run, k)
        if cl_data is None:
            n_clusters = k
            cl_data = cluster_genes(adata, img, species, cl_method, n_clusters, random_state=random_state+run)
            write_clusters_as_cache(cl_data, data_name, cl_method, run, k)
    elif callable(cl_method):  # cl_method is a custom functions
        cl_data = read_clusters_from_cache(adata, cl_method.__name__, run, k)
        if cl_data is None:
            cl_data = cl_method(adata, img, **kwargs) # cl_method must return AnnData
            write_clusters_as_cache(cl_data, cl_method.__name__, run)
    else:
        raise NotImplementedError(f"`cl_method` should be an valid string or a function, got {type(cl_method)}.")
    return cl_data

