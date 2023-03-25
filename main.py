# -*- coding: utf-8 -*-
# @Time : 2023/2/6 2:08
# @Author : Tory Deng
# @File : main.py
# @Software: PyCharm


# TODO: add Seurat clustering for SRT
# TODO: check GeneClust dependency

from benchmark.run_benchmark import run_bench

data_cfg = {
    'DLPFC151507': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151507_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151507_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151508': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151508_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151508_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151509': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151509_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151509_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151510': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151510_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151510_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151669': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151669_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151669_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151670': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151670_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151670_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151671': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151671_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151671_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151672': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151672_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151672_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151673': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151673_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151673_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151674': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151674_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151674_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151675': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151675_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151675_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
    'DLPFC151676': {
        'adata_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/h5ad/151676_10xvisium.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/human_dorsolateral_prefrontal_cortex/image/151676_full_image.tif',
        'annot_key': 'spatialLIBD',
    },
}

cl_cfg = {'Giotto': 1,
          'CNN-PReg': 1}
#cl_cfg = {'CNN-PReg': 1}

metrics = ['DB-Index(Pearson)', 'DB-Index(Spatial)','Average_Neg_Log_Min_P']

run_bench(
    data_cfg, cl_cfg, metrics, species='human', clean_cache=False, log_path='loguru.log', random_state=100, k=100
)
# rm_cache("./cache")
data_cfg = {
    'V1_Adult_Mouse_Brain': {
        'adata_path': '/volume2/bioinfo/SRT/visium/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain.h5ad',
        'image_path': '/volume2/bioinfo/SRT/visium/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_image.tif',
        
    }
    
}

cl_cfg = {'CNN-PReg': 1,
          'Giotto': 1,}
#cl_cfg={'CNN-PReg':1}

metrics = ['DB-Index(Pearson)','DB-Index(Spatial)' ,'Average_Neg_Log_Min_P']
    

run_bench(
    data_cfg, cl_cfg, metrics, species='mouse', clean_cache=False, log_path='loguru.log', random_state=100, k=100
)
