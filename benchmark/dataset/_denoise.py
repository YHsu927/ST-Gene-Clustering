import os
import anndata as ad
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import STAGATE_pyG as STAGATE
import torch
from .._utils import HiddenPrints
from loguru import logger


def denoising(adata: ad.AnnData,
              data_name: str,
              plot: bool = False,
              gene_num: int = 0):
    work_dir = f"./cache/denoise/{data_name}/"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if os.path.exists(os.path.join(work_dir, "ReX.npy")):
        adata.layers['STAGATE_ReX'] = np.load(os.path.join(work_dir, "ReX.npy"))
        logger.info("Loaded denoising result from cache.")
    else:
        logger.info("Begin denoising by STAGATE...")
        with HiddenPrints():
            STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
            adata = STAGATE.train_STAGATE(adata, save_reconstrction=True, n_epochs=100,
                                          device=torch.device('cpu'), verbose=False)
            np.save(os.path.join(work_dir, "ReX.npy"), adata.layers['STAGATE_ReX'])
        logger.info("save denoising result in cache.")

    if plot:
        # all_gene = adata.var.index.values
        # gene_num = [int(i) for i in np.linspace(0, 10000, 41)]

        # for i in range(3000):
        #     plot_gene = all_gene[i]
        #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        #     sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False,
        #                   ax=axs[0], title='RAW_'+plot_gene, vmax='p99')
        #     sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False,
        #                   ax=axs[1], title='STAGATE_'+plot_gene, layer='STAGATE_ReX', vmax='p99')
        #     # plt.savefig(os.path.join(work_dir, f"{plot_gene}.png"))
        #     plt.show()

        gene_name = ['UBE2J2', 'AGRN', 'ID3', 'HDAC1']
        fig, axs = plt.subplots(2, 4, figsize=(9, 4))
        for i in range(len(gene_name)):
            plot_gene = gene_name[i]
            sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False,
                          ax=axs[0, i], vmax='p99')
            axs[0, i].set_xlabel(None)
            axs[0, i].set_ylabel(None)
            sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False,
                          ax=axs[1, i], layer='STAGATE_ReX', vmax='p99')
            axs[1, i].set_xlabel(None)
            axs[1, i].set_ylabel(None)
            axs[1, i].set_title(None)

        axs[0, 0].set_ylabel('Raw', fontsize=14)
        axs[1, 0].set_ylabel('Denoise', fontsize=14)

        plt.savefig(os.path.join(work_dir, "denoise.png"))

    adata.X = adata.layers['STAGATE_ReX']
    adata.layers = None

    return adata