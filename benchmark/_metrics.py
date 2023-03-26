# -*- coding: utf-8 -*-
# @Time : 2023/2/6 22:56
# @Author : Tory Deng
# @File : _metrics.py
# @Software: PyCharm
from typing import Literal, Union
import anndata
import numba
import gseapy as gp
import rpy2.robjects as robjects
from rpy2.robjects import packages
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from multiprocessing import Pool



def compute_clustering_metrics(true_labels: Union[np.ndarray, pd.Series], pred_labels: np.ndarray, metric: Literal['ARI', 'NMI']):
    valid_label_mask = ~pd.isna(true_labels).to_numpy()  # np.isnan doesn't support strings
    if (n_nan_labels := true_labels.shape[0] - valid_label_mask.sum()) > 0:
        logger.opt(colors=True).info(
            f"Ignoring <yellow>{n_nan_labels}</yellow> NaNs in annotations during the {metric} computation."
        )
    if metric == 'ARI':
        return adjusted_rand_score(true_labels[valid_label_mask], pred_labels[valid_label_mask])
    elif metric == 'NMI':
        return normalized_mutual_info_score(true_labels[valid_label_mask], pred_labels[valid_label_mask])
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")


def compute_gene_clustering_metrics(
        data:pd.DataFrame, 
        adata:anndata.AnnData,
        metric: Literal['DB-Index(Pearson)', 'DB-Index(Spatial)','Average_Neg_Log_Min_P','Average_Neg_Log_Min_P_Fast'], #'Average_Neg_Log_Min_P_Fast' is not suggested and stable
        species:Literal['mouse', 'human']
):
    # adata = adata.raw.to_adata()
    valid_label_mask = ~pd.isna(data['cluster_id']).to_numpy()  # np.isnan doesn't support strings
    if (n_nan_labels := adata.shape[1] - valid_label_mask.sum()) > 0:
        logger.opt(colors=True).info(
            f"Ignoring <yellow>{n_nan_labels}</yellow> NaNs in gene clustering during the {metric} computation."
        )
    gene_fil=[]
    for i in range(adata.shape[1]):
        if adata.var.index[i] in list(data['gene_id']):
            gene_fil.append(i)
    adata=adata[:,gene_fil]
    adata.var['cluster_id']=data['cluster_id'].values
    if species!='mouse' and species!='human':
        raise NotImplementedError(f"Species {species} is not implemented.")
    if metric == 'DB-Index(Pearson)':
        return DB_INDEX_PEARSON_DISTANCE(adata)
    elif metric == 'DB-Index(Spatial)':
        return DB_INDEX_SPATIAL_EUCLIDEAN_DISTANCE(adata)
    elif metric == 'Average_Neg_Log_Min_P':
        return cofunction(adata,species,fast=False)
    elif metric == 'Average_Neg_Log_Min_P_Fast':
        return cofunction(adata,species,fast=True)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")

        
        
def DB_INDEX_PEARSON_DISTANCE(adata):
    nc=np.unique(adata.var['cluster_id'])
    lc=len(np.unique(adata.var['cluster_id']))
    cell=adata.shape[0]
    d=np.zeros((lc,1))
    central1=np.zeros((lc,cell))
    q=0
    for i in nc:
        n=np.sum(adata.var['cluster_id']==i)
        genemat=adata[:,adata.var['cluster_id']==i].X.T
        if n == 1:
            d[q,0]=0
        else:
            d[q,0]=(n*n-np.sum(np.corrcoef(genemat)))*2/(n*(n-1))
        central1[q,:]=np.mean(genemat,axis=0)
        q+=1
    mat=np.zeros((central1.shape[0],central1.shape[0]))
    for i in range(central1.shape[0]):
        for j in  range(central1.shape[0]):
            if i!=j:
                mat[i,j]=(d[i,:]+d[j,:])/(1-np.corrcoef(central1[i,:],central1[j,:])[0,1])
    return np.mean(np.max(mat,axis=0))


@numba.jit(nopython=True)
def spatial_corr(genedata,graph):
    mat=np.zeros((genedata.shape[0],genedata.shape[0]))
    for i in range(genedata.shape[0]):
        for j in np.arange(0,i,1):
            mat[i,j]=np.sqrt(np.dot(np.dot((genedata[i,:]-genedata[j,:]),graph),(genedata[i,:]-genedata[j,:]).T))
    return mat
def calculate(i,graph,adata):
    
    n=np.sum(adata.var['cluster_id']==i)
    genemat=adata[:,adata.var['cluster_id']==i].X.T
    if n == 1:
        a=0
    else:
        a=2*np.sum(spatial_corr(genemat,graph))*2/(n*(n-1))
    
    return a
def calculate1(i,graph,adata):
    
    genemat=adata[:,adata.var['cluster_id']==i].X.T
    b=np.mean(genemat,axis=0)
    return b
   
def DB_INDEX_SPATIAL_EUCLIDEAN_DISTANCE(adata):
    #gene_data=adata.varm['gene_map']
    #spot=gene_data.shape[1]
    spot=adata.shape[0]
    graph=np.zeros((spot,spot))
    for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
        row_ix,col_ix = row_col
        for j,row_col2 in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
            row_jx,col_jx = row_col2
            graph[i,j] =np.exp((np.square(row_ix-row_jx)+np.square(col_ix-col_jx))*(-1))
    graph=graph.astype('float32')
    nc=np.unique(adata.var['cluster_id'])
    lc=len(np.unique(adata.var['cluster_id']))
    d=np.zeros((lc,1))
    central=np.zeros((lc,spot))
    pool = Pool(processes=60)
    result=[]
    result1=[]
       
    for i in nc:
        '''
       for循环执行流程：
       （1）添加子进程到pool，并将这个对象（子进程）添加到result这个列表中。（此时子进程并没有运行）
       （2）执行子进程（同时执行48个）
       '''
        #calculate(i,graph,q)
        a=pool.apply_async(calculate, args=(i,graph,adata))#维持执行的进程总数为10，当一个进程执行完后添加新进程.
        b=pool.apply_async(calculate1, args=(i,graph,adata))
        result.append(a)
        result1.append(b)
    pool.close()
    pool.join()
        #n=np.sum(adata.var['cluster_id']==i)
        #genemat=adata[:,adata.var['cluster_id']==i].X.T
        #d[q,0]=np.sum(spatial_corr(genemat,graph))*2/(n*(n-1))
        #central[q,:]=np.mean(genemat,axis=0)
       
   
    q=0
    for i in result:
        d[q,0]=i.get()
        q+=1
    q=0
    for i in result1:
        central[q,:]=i.get()
        q+=1
    mat=np.zeros((central.shape[0],central.shape[0]))
    d=d.astype('float32')
    central=central.astype('float32')
    for i in range(central.shape[0]):
        for j in  range(central.shape[0]):
            if i!=j:
                mat[i,j]=(d[i,:]+d[j,:])/spatial_corr(central[[i,j],:],graph)[1,0]
    return np.mean(np.max(mat,axis=0)) 


def cluster_min_logp_fast(gene_list,species='mouse'):

    
    if species=='mouse':
        enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                 gene_sets=['GO_Biological_Process_2021'],
                 organism='mouse', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
    if species=='human':
        enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                 gene_sets=['GO_Biological_Process_2021'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )

# obj.results stores all results
#enr.results.head(5)
    return -np.log(enr.results['Adjusted P-value'].min())

pandas2ri.activate()
packages.importr('clusterProfiler')
packages.importr('org.Mm.eg.db')
packages.importr('org.Hs.eg.db')
def cluster_min_logp(gene_list,species='mouse'):

    gene_list=tuple(gene_list)
    if species=='mouse':
        rscript = f"""
        eg <- bitr(c{gene_list},
           fromType="SYMBOL",
           toType=c("ENTREZID"),
           OrgDb="org.Mm.eg.db")

        GO_enrich <- enrichGO(eg$SYMBOL,
                      OrgDb = org.Mm.eg.db,
                      ont='ALL',
                      keyType = "SYMBOL",
                      pAdjustMethod = 'BH',
                      pvalueCutoff = 0.5,
                      qvalueCutoff = 0.5)
        GO_res <- GO_enrich@result

        """
    if species=='human':
        rscript = f"""
        eg <- bitr(c{gene_list},
           fromType="SYMBOL",
           toType=c("ENTREZID"),
           OrgDb="org.Hs.eg.db")

        GO_enrich <- enrichGO(eg$SYMBOL,
                      OrgDb = org.Hs.eg.db,
                      ont='ALL',
                      keyType = "SYMBOL",
                      pAdjustMethod = 'BH',
                      pvalueCutoff = 0.5,
                      qvalueCutoff = 0.5)
        GO_res <- GO_enrich@result

        """
    result=robjects.r(rscript)
    return -np.log(result['p.adjust'].min())
def cofunction(adata,species='mouse',fast=True):
    nc=np.unique(adata.var['cluster_id'])
    lc=len(np.unique(adata.var['cluster_id']))
    d=np.zeros((lc,1))
    q=0
    for i in nc:
        if fast:
            d[q,0]=cluster_min_logp_fast(list(adata.var.index[adata.var['cluster_id']==i]),species)
        else:
            try:
                d[q,0]=cluster_min_logp(list(adata.var.index[adata.var['cluster_id']==i]),species)
            except:
                try:
                    d[q,0]=cluster_min_logp_fast(list(adata.var.index[adata.var['cluster_id']==i]),species)
                except:
                    d[q,0]=np.nan
        q+=1
    return np.nanmean(d)    

