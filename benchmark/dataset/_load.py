# -*- coding: utf-8 -*-
# @Time : 2023/2/6 16:55
# @Author : Tory Deng
# @File : _load.py
# @Software: PyCharm
import os
from math import e
from typing import Dict, Literal
from typing import Union

import numpy as np
import anndata as ad
import cv2
import scanpy as sc
from loguru import logger
from scipy.sparse import csr_matrix

from ._io import read_h5ad, write_adata_as_cache, read_adata_from_cache
from ._preprocess import make_unique, prefilter_special_genes, clean_var_names, clean_annotations
from ._denoise import denoising
from .._utils import console


def quality_control(adata: ad.AnnData, data_props:  Dict[str, Union[os.PathLike, str]]):
    clean_var_names(adata)
    make_unique(adata)
    if 'annot_key' in data_props.keys():
        to_remove = data_props['to_remove'] if 'to_remove' in data_props.keys() else None
        to_replace = data_props['to_replace'] if 'to_replace' in data_props.keys() else None
        clean_annotations(adata, data_props['annot_key'], to_remove, to_replace)
        # quality control
        prefilter_special_genes(adata)
        sc.pp.filter_genes(adata, min_cells=10)
    
    return adata


def _load_adata(
        data_name: str,
        data_props: Dict[str, Union[os.PathLike, str]],
        preprocess: bool = True,
        denoise: bool = False):
    logger.info("Reading adata...")
    adata = read_adata_from_cache(data_name)
    if isinstance(adata, ad.AnnData):
        console.print(f"Using cached adata and skip preprocessing. "
                      f"Shape: [yellow]{adata.n_obs}[/yellow] cells, [yellow]{adata.n_vars}[/yellow] genes.")
        return adata
    elif adata is None:
        logger.opt(colors=True).info(
            f"No cache for <magenta>{data_name}</magenta>. Trying to read h5ad from the given path in config..."
        )
        assert 'adata_path' in data_props.keys(), "Not found path to adata."
        adata = read_h5ad(data_props['adata_path'])


        if preprocess:
            console.print(f"Before QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
            adata = quality_control(adata, data_props)
            console.print(f"After QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
            # store to adata.raw
            adata.raw = adata

            if denoise:
                adata = denoising(adata, data_name, plot=False)
                adata = quality_control(adata, data_props)

            # normalization
            sc.pp.normalize_per_cell(adata)
            adata.layers['normalized'] = adata.X.copy()
            sc.pp.log1p(adata, base=e)
            # store data name
            adata.uns['data_name'] = data_name
            # store annot_key and batch_key
            if 'annot_key' in data_props.keys():
                adata.uns['annot_key'] = data_props['annot_key']
            if 'batch_key' in data_props.keys():
                adata.uns['batch_key'] = data_props['batch_key']
            # store the spot shape for spatial transcriptomics
            if 'shape' in data_props.keys():
                adata.uns['shape'] = data_props['shape']
            # Reload gene activity maps
            console.print("Reload gene activity maps...")
            adata.obs['array_x']=np.ceil((adata.obs['array_col']-adata.obs['array_col'].min())/2).astype(int)
            adata.obs['array_y']=(adata.obs['array_row']-adata.obs['array_row'].min()).astype(int)
            adata = adata[adata.obs['in_tissue']==1]
            
            ####convert gene expression into a matrix of n_row x n_col
            all_gene_exp_matrices = {}
            shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
            for gene in adata.var.index.values:
                g_matrix = np.zeros(shape=shape)
                g = adata[:,gene].X.tolist()
       
                for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
                    row_ix,col_ix = row_col
            
                    g_matrix[row_ix,col_ix] = g[i][0]
                all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
            all_gmat = {k:all_gene_exp_matrices[k].todense() for k in list(all_gene_exp_matrices.keys())}  
            gene_data=np.array(list(all_gmat.values()))
            adata.varm['gene_map']=gene_data.reshape(gene_data.shape[0],gene_data.shape[1]*gene_data.shape[2])
            console.print("Reload successfully!")
            # save the cache
            write_adata_as_cache(adata, data_name)
        else:
            console.print(f"Skip preprocessing [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
        return adata


def load_data(
        data_name: str,
        data_props: Dict[str, Union[os.PathLike, str]],
        preprocess: bool = True,
        denoise: bool = False):
    console.rule('[bold red]' + data_name)
    adata = _load_adata(data_name, data_props, preprocess, denoise)
    # read image
    if 'image_path' in data_props.keys():
        img = cv2.imread(data_props['image_path'])
        logger.info("Image has been loaded.")
        return adata, img
    else:
        logger.info("'image_path' is not given. Image data not found.")
        return adata, None

