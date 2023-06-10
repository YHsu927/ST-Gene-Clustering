# -*- coding: utf-8 -*-
# @Time : 2023/2/7 10:37
# @Author : Tory Deng
# @File : functions.py
# @Software: PyCharm
import contextlib
import os
import random
from pathlib import Path
from multiprocessing import Process

from typing import Literal
import anndata as ad
import anndata2ri
import numpy as np
import pandas as pd
from loguru import logger
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr

import sys
import cv2

from scipy.sparse import csr_matrix
from scipy.io import mmread


from ..._utils import HiddenPrints
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def cluster_genes(adata: ad.AnnData, image: np.ndarray, species: Literal['mouse', 'human'], method: str, k: int, random_state: int = 0):
    logger.opt(colors=True).info(f"Running <magenta>{method}</magenta> clustering with <yellow>{k}</yellow> clusters "
                                 f"and random state <yellow>{random_state}</yellow>...")
    if method == 'Giotto':
        return Giotto_clustering(adata, k, random_state=random_state)
    if method == 'STUtility':
        return STUtility_clustering(adata, k, random_state=random_state)
    if method == 'SPARK':
        return SPARK_clustering(adata, k, random_state=random_state)
    if method == 'CNN-PReg':
        return run_gene_clustering(adata, species=species, n_cluster=k, random_state=random_state,pre_train=False)
    if method == 'CNN-PReg-Pre':
        return run_gene_clustering(adata, species=species, n_cluster=k, random_state=random_state,pre_train=True)
    else:
        raise NotImplementedError(f"{method} has not been implemented.")


def Giotto_save_rds(adata: ad.AnnData):
    RDS_PATH = Path("./cache/temp")
    RDS_PATH.mkdir(parents=True, exist_ok=True)
    with HiddenPrints():
        anndata2ri.activate()
        importr("Giotto")
        globalenv['spe'] = adata
        r("""
        gobj <- createGiottoObject(expression = assay(spe, 'X'),
                                   spatial_locs = reducedDim(spe, 'spatial'),
                                   cell_metadata = colData(spe),
                                   feat_metadata = rowData(spe))
        saveRDS(gobj, './cache/temp/giotto_obj.rds')
        """)
        anndata2ri.deactivate()


def Giotto_clustering(adata: ad.AnnData, k: int, random_state: int = 0):
    '''
    Giotto provides a comprehensive toolbox for spatial analysis by R
    More details can be seen at:
    https://github.com/drieslab/Giotto
    https://giottosuite.readthedocs.io/en/latest/#
    '''
    p = Process(target=Giotto_save_rds, args=(adata,))
    p.start()
    p.join()
    with HiddenPrints():
        anndata2ri.activate()
        importr("Giotto")
        globalenv['n_cluster'] = k
        r("""
        gobj <- readRDS('./cache/temp/giotto_obj.rds')
        gobj <- normalizeGiotto(gobject = gobj, verbose = F)
        gobj <- createSpatialNetwork(gobject = gobj, method = 'kNN', k = 8, name = 'spatial_network')
        ranktest <- binSpect(gobject = gobj, bin_method='rank', cores=10, spatial_network_name = 'spatial_network')

        ext_spatial_feats <- ranktest$feats
        spat_cor_netw_DT <- detectSpatialCorFeats(gobj, method = 'network',
                                                  spatial_network_name = 'spatial_network',
                                                  subset_feats = ext_spatial_feats)
        spat_cor_netw_DT <- clusterSpatialCorFeats(spat_cor_netw_DT, name = 'spat_netw_clus', k = n_cluster)
        """)
        result = list(r('spat_cor_netw_DT$cor_clusters$spat_netw_clus'))
        names = list(r('names(spat_cor_netw_DT$cor_clusters$spat_netw_clu)'))
        anndata2ri.deactivate()

    data = pd.DataFrame({'gene_id': names, 'cluster_id': result}, columns=['gene_id', 'cluster_id'])
    return data


def STUtility_clustering(adata: ad.AnnData, k: int, random_state: int = 0):
    with HiddenPrints():
        anndata2ri.activate()
        importr("STutility")
        importr("Seurat")
        globalenv['n_cluster'] = k
        globalenv['mtx'] = csr_matrix(adata.X.T)
        globalenv['cellinfo'] = adata.obs
        globalenv['gene_name'] = adata.var_names
        globalenv['seed'] = random_state
        r("""
        set.seed(seed)

        colnames(mtx) <- rownames(cellinfo)
        rownames(mtx) <- gene_name

        se <- CreateSeuratObject(mtx, min.cells=5, min.features=5,
        meta.data = cellinfo[,!colnames(cellinfo)%in%c("n_genes","n_counts"),drop=F])

        gene_num <- 5000
        se <- SCTransform(se, variable.features.n = gene_num)
        se <- RunNMF(se, nfactors=n_cluster, n.cores=40)

        label <- c()
        sele_name <- rownames(se@reductions$NMF@feature.loadings)
        for (i in 1:gene_num){
        t <- which.max(se@reductions$NMF@feature.loadings[sele_name[i], ])
        label <- c(label, t)
        }
        options(warn=-1)
        """)
        result = list(r("label"))
        names = list(r("sele_name"))
        anndata2ri.deactivate()

    data = pd.DataFrame({'gene_id': names, 'cluster_id': result}, columns=['gene_id', 'cluster_id'])
    return data


def SPARK_clustering(adata: ad.AnnData, k: int, random_state: int = 0):
    with HiddenPrints():
        anndata2ri.activate()
        importr("SPARK")
        importr("amap")
        globalenv['n_cluster'] = k
        globalenv['mtx'] = adata.X.T
        globalenv['info'] = adata.obsm['spatial']
        globalenv['gene_name'] = adata.var_names
        globalenv['seed'] = random_state
        r("""
        LMReg <- function(ct, T) {
        return(lm(ct ~ T)$residuals)
        }

        set.seed(seed)
        spark <- sparkx(mtx, info, verbose=F)
        idx <- which(spark$res_mtest$combinedPval < 0.1)
        lib_size <- apply(mtx, 2, sum)
        vst_count <- log(mtx+0.01)
        vst_count <- vst_count[idx, ]
        sele_name <- gene_name[idx]
        vst_res <- t(apply(vst_count, 1, LMReg, T = log(lib_size)))

        hc <- hcluster(vst_res, method = "euc", link = "ward", nbproc = 20,
               doubleprecision = TRUE)
        memb <- cutree(hc, k = n_cluster)
        options(warn=-1)
        """)
        result = list(r("memb"))
        names = list(r("sele_name"))
        anndata2ri.deactivate()

    data = pd.DataFrame({'gene_id': names, 'cluster_id': result}, columns=['gene_id', 'cluster_id'])
    return data


def mnistNetwork(imgs, num_cluster: int, name='mnistNetwork', reuse=None):
    
    """
    Same network structure on MNIST used in the paper:
    Deep Adaptive Image Clustering
    https://doi.org/10.1109/ICCV.2017.626
    https://github.com/vector-1127/DAC/blob/master/MNIST/mnist.py
    """
    
    with tf.variable_scope(name, reuse=reuse):
        
        # convolutional layer 1
        conv1 = tf.layers.conv2d(imgs, 64, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv1 = tf.layers.batch_normalization(conv1, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv1 = tf.nn.relu(conv1)
        
        # convolutional layer 2
        conv2 = tf.layers.conv2d(conv1, 64, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv2 = tf.layers.batch_normalization(conv2, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv2 = tf.nn.relu(conv2)
        
        # convolutional layer 3
        conv3 = tf.layers.conv2d(conv2, 64, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.max_pooling2d(conv3, [2,2], [2,2])
        conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
        
        # convolutional layer 4
        conv4 = tf.layers.conv2d(conv3, 128, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv4 = tf.layers.batch_normalization(conv4, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv4 = tf.nn.relu(conv4)
        
        # convolutional layer 5
        conv5 = tf.layers.conv2d(conv4, 128, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv5 = tf.layers.batch_normalization(conv5, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv5 = tf.nn.relu(conv5)
        
        # convolutional layer 6
        conv6 = tf.layers.conv2d(conv5, 128, [3,3], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv6 = tf.nn.relu(conv6)
        conv6 = tf.layers.max_pooling2d(conv6, [2,2], [2,2])
        conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
        
        # convolutional layer 7
        conv7 = tf.layers.conv2d(conv6, num_cluster, [1,1], [1,1], padding='valid', activation=None,
                                 kernel_initializer=tf.keras.initializers.he_normal())
        conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv7 = tf.nn.relu(conv7)
        conv7 = tf.layers.average_pooling2d(conv7, [2,2], [2,2])
        conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv7_flat = tf.layers.flatten(conv7)
        
        # fully connected layer 8
        fc8 = tf.layers.dense(conv7_flat, num_cluster, kernel_initializer=tf.initializers.identity())
        fc8 = tf.layers.batch_normalization(fc8, axis=-1, epsilon=1e-5, training=True, trainable=False)
        fc8 = tf.nn.relu(fc8)
        
        # fully connected layer 9
        fc9 = tf.layers.dense(fc8, num_cluster, kernel_initializer=tf.initializers.identity())
        fc9 = tf.layers.batch_normalization(fc9, axis=-1, epsilon=1e-5, training=True, trainable=False)
        fc9 = tf.nn.relu(fc9)
        
        embs = tf.nn.softmax(fc9)

    return embs

def get_mnist_batch(batch_size, n_sample, w, h, mnist_data):
    
    """
    Generate a batch on MNIST hand written dataset
    """
    
    batch_index = random.sample(range(n_sample), batch_size)

    batch_data = np.empty([batch_size, w, h, 1], dtype=np.float32)
    for n, i in enumerate(batch_index):
        batch_data[n, ...] = mnist_data[i, ...]

    return batch_data, batch_index

def get_gene_batch_approx_reg(batch_size, n_sample, w, h, gene_data, ppi_mat):
    
    """
    Generate a batch on Visium spatial transcriptomics data 
    and Laplacian matrix on corresponding sub-PPI graph
    """
    
    batch_index = np.array(random.sample(range(n_sample), batch_size))

    batch_data = gene_data[batch_index, ...]

    A = ppi_mat[np.ix_(batch_index, batch_index)]

    d = 1.0/np.sqrt(A.sum(axis=1))
    D_inv = np.diag(np.where(np.isinf(d), 0, d))
    batch_ppi_lap_mat = np.identity(batch_size) - D_inv@A@D_inv

    return batch_data, batch_ppi_lap_mat, batch_index

def get_gene_batch_graph_reg(batch_size, n_sample, w, h, gene_data, ppi_mat):
    
    """
    Generate a batch on Visium spatial transcriptomics data 
    and Laplacian matrix on PPI graph after reordering 
    """
    
    random_index = np.array(random.sample(range(n_sample), n_sample))
    
    batch_index = random_index[0:batch_size]
    rest_index = random_index[batch_size:]

    batch_data = gene_data[batch_index, ...]

    A = ppi_mat[np.ix_(random_index, random_index)]

    d = 1.0/np.sqrt(A.sum(axis=1))
    D_inv = np.diag(np.where(np.isinf(d), 0, d))
    ppi_lap_mat = np.identity(n_sample) - D_inv@A@D_inv

    return batch_data, ppi_lap_mat, batch_index, rest_index

def complete_restore(session, checkpoint_path):
    
    """
    Restore weights for pretrained model on MNIST
    """
    
    #saver = tf.train.Saver()
    #saver.restore(session, checkpoint_path)
    model_dir = os.path.join(os.getcwd(), "model")
    saver = tf.train.import_meta_graph(checkpoint_path)
    saver.restore(session,tf.train.latest_checkpoint(model_dir))
    
    
    #return saver
def ge_id_trans(gene_ids,df):
    
    for i in range(df.shape[0]):
        k=list(gene_ids).index(df[i,0])
        gene_ids[k]=df[i,1]
        
        
    import os


def run_pretraining(n_cluster=100,random_state=0):
    random.seed(random_state)
    runepoch=50
    tf.reset_default_graph()
    eps = 1e-10
    base_lr = 0.001 # learning rate
    
    # GPU
    if True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU 0 will be used")
   
    
    # Model
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("Start pre-training on MNIST data...")
    
    # Load MNIST dataset
    (mnist_train_data, mnist_train_labels), (mnist_test_data, mnist_test_labels) = tf.keras.datasets.mnist.load_data()
    mnist_train_data = np.expand_dims(mnist_train_data, axis=-1)
    mnist_test_data = np.expand_dims(mnist_test_data, axis=-1)

    mnist_data = np.concatenate([mnist_train_data, mnist_test_data], axis=0)
    mnist_labels = np.concatenate([mnist_train_labels, mnist_test_labels], axis=0)
    
    n_sample, w, h, _ = np.shape(mnist_data)
    
    # Normalization
    mnist_data_norm = mnist_data.reshape(mnist_data.shape[0], -1)
    mnist_data_norm = mnist_data_norm/np.amax(mnist_data_norm, axis=1)[:, None]
    mnist_data_norm = mnist_data_norm.reshape(mnist_data.shape)

    imgs = tf.placeholder(shape=[None, w, h, 1], dtype=tf.float32, name='images')
    
    u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
    l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
    lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    
    label_feat = mnistNetwork(imgs, n_cluster, name="mnistNetwork", reuse=False)
    label_feat_norm = tf.nn.l2_normalize(label_feat, dim=1)
    # Compute similarity matrix based on embeddings from CNN encoder
    sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)
    
    pos_loc = tf.greater(sim_mat, u_thres, name='greater')
    neg_loc = tf.less(sim_mat, l_thres, name='less')
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)
    
    pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
    neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)
    
    # Construct loss function based on similarity matrix
    loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)

    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        eta = 0 # step size
        epoch = 1
        u = 0.95 # upper threshold
        l = 0.455 # lower threshold
        
        while u > l:

            print("Epoch %d" % epoch)
            
            # Update upper and lower thresholds
            u = 0.95 - eta
            l = 0.455 + 0.1*eta
            
            for i in range(1, int(runepoch + 1)):
                mnist_batch, batch_index = get_mnist_batch(128, n_sample, w, h, mnist_data_norm)
                feed_dict={imgs: mnist_batch,
                        u_thres: u,
                        l_thres: l,
                        lr: base_lr}

                train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
                if i % 5 == 0:
                    print('training loss at iter %d is %f' % (i, train_loss))
                    
            # Update step size
            eta += 1.1 * 0.009
            
            # Create checkpoint every 5 epochs 
            if epoch % 5 == 0: 
                model_name = 'CNN_MNIST_ep_' + str(epoch) + '.ckpt'
                save_path = saver.save(sess, os.path.join(model_dir, model_name))
                print("Checkpoint created in file: %s" % save_path)

            epoch += 1

def run_gene_clustering(adata: ad.AnnData, species:Literal['mouse', 'human'], n_cluster: int =100, random_state: int =0, pre_train: bool = False): 
#(gene_ids,A,gene_data):
    runepoch=50
    random.seed(random_state)
    if pre_train:
        run_pretraining(n_cluster=n_cluster,random_state=random_state)
    eps = 1e-10
    base_lr = 0.001 # learning rate 
    tf.reset_default_graph()
    
    # GPU
    if True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU 0 will be used")
    
    
    # Checkpoint
    checkpoint = os.path.join(os.getcwd(), "model", "CNN_MNIST_ep_45.ckpt.meta")
    if not os.path.exists(checkpoint):
        sys.exit("Pre-trained model does not exist")
    
    # Model 
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Clustering results
    gene_cluster_dir = os.path.join(os.getcwd(), "clustering")
    if not os.path.exists(gene_cluster_dir):
        os.makedirs(gene_cluster_dir)
    
    # Load Mouse or Human PPI graph
    if species=='mouse':
        print("Load mus musculus PPI network")
        A = mmread('Mmusculus_PPI_80.mtx').toarray()
        gene_ids = np.loadtxt('Mmusculus_gene_list_80.txt', dtype=np.str)
        df=pd.read_csv('mart_export.txt',sep='\t')
    elif species=='human':
        print("Load homo sapiens PPI network")
        A = mmread('Hsapiens_PPI_80.mtx').toarray()
        gene_ids = np.loadtxt('Hsapiens_gene_list_80.txt', dtype=np.str)
        df=pd.read_csv('mart_export (1).txt',sep='\t')
    else:sys.exit("PPI graph does not exist")
    
    #df=np.zeros((adata.shape[1],2),dtype=object)
    #df[:,0]=np.array(adata.var['gene_ids'])
    #df[:,1]=np.array(adata.var.index)
    df=np.array(df)
    all_genes = adata.var.index.values
    #Ensembl to MGI/HGNC Symbol
    ge_id_trans(gene_ids,df)
    j=[i for i in list(gene_ids) if i in list(all_genes)]
    selected=[]
    for i in range(all_genes.shape[0]):
        if all_genes[i] in j:
            selected.append(i)
    all_genes=all_genes[selected]
    adata=adata[:,selected]
    selected1=[]
    for i in range(gene_ids.shape[0]):
        if gene_ids[i] in j:
            selected1.append(i)
    A=A[selected1,:]
    A=A[:,selected1]
    gene_ids=gene_ids[selected1]
   
    print("Start clustering on all genes  ...")
        
    # Reload gene activity maps
    adata.obs['array_x']=np.ceil((adata.obs['array_col']-adata.obs['array_col'].min())/2).astype(int)
    adata.obs['array_y']=(adata.obs['array_row']-adata.obs['array_row'].min()).astype(int)
    adata = adata[adata.obs['in_tissue']==1]

    ####convert gene expression into a matrix of n_row x n_col
    #all_gene_exp_matrices = {}
    #shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
    #for gene in all_genes:
     #   g_matrix = np.zeros(shape=shape)
      #  g = adata[:,gene].X.tolist()
       
       # for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
        #    row_ix,col_ix = row_col
           
            #g_matrix[row_ix,col_ix] = g[i][0]
        #all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
    #all_gmat = {k:np.exp(all_gene_exp_matrices[k].todense())-1 for k in list(all_gene_exp_matrices.keys())}  
    #gene_data=np.array(list(all_gmat.values()))
    gene_data=adata.varm['gene_map'].reshape(adata.shape[1],adata.obs['array_y'].max()+1,adata.obs['array_x'].max()+1)
    reset=[]
    for i in range(gene_ids.shape[0]):
        k=list(all_genes).index(gene_ids[i])
        reset.append(k)
    gene_data=gene_data[reset]
    adata=adata[:,reset]
    gene_data_norm = gene_data.reshape(gene_data.shape[0], -1)
    gene_data_norm = gene_data_norm/np.amax(gene_data_norm, axis=1)[:, None]
    gene_data_norm = gene_data_norm.reshape(gene_data.shape)
    
    # Process gene activity maps (padding or resize)
    
    gene_data_norm = np.stack([cv2.resize(gene_data_norm[i, ...], dsize=(28,28)) 
                                   for i in range(gene_data_norm.shape[0])],axis=0)
   
        
    gene_data_norm = np.expand_dims(gene_data_norm, axis=-1)

    n_gene, w, h, _ = np.shape(gene_data_norm)

    gene_maps = tf.placeholder(shape=[None, w, h, 1], dtype=tf.float32, name='gene_maps')
    lap_mat = tf.placeholder(shape=None, dtype=tf.float32, name='lap_mat')
    
    u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
    l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
    lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    alpha = tf.placeholder(shape=[], dtype=tf.float32, name='alpha')
    
    # Prepend additional convolutional to CNN encoder according to gene activity maps processing 
    gene_embs = mnistNetwork(gene_maps, n_cluster, name="mnistNetwork", reuse=tf.AUTO_REUSE)
   
    
    gene_embs_norm = tf.nn.l2_normalize(gene_embs, dim = 1)
    
    # Use exact or approximated PPI graph regularization based on the number of genes involved in the clustering
  
        # Approximated PPI graph regularization
        # Compute similarity matrix and PPI graph regularization based on gene embeddings in the batch
    sim_mat = tf.matmul(gene_embs_norm, gene_embs_norm, transpose_b=True)
    graph_reg = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(gene_embs_norm), lap_mat), gene_embs_norm))
    
    pos_loc = tf.greater(sim_mat, u_thres, name='greater')
    neg_loc = tf.less(sim_mat, l_thres, name='less')
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)
    
    pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
    neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)
    
    graph_reg = tf.math.divide(graph_reg, n_cluster)
    
    # Construct combined loss (clustering loss and PPI graph regularization)
    loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy) + tf.multiply(graph_reg, alpha)
   
    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)
    
    # Infer gene cluster membership based on gene embeddings
    gene_clusters = tf.argmax(gene_embs, axis=1)
    
    saver = tf.train.Saver()
    
    
    with tf.Session() as sess:
        
        # Load pre-trained CNN
        complete_restore(sess, checkpoint)
       
   
        print('Pre-trained model restored!')

        eta = 0 # step size
        epoch = 1
        u = 0.95 # threshold for similar gene selection
        l = 0.455 # threshold for dissimilar gene selection
        
        # Create gene embedding matrix when fewer genes involved in the clustering
        
        while u > l:

            print("Epoch %d" % epoch)
            
            # Update thresholds for both similar and dissimilar gene selection
            u = 0.95 - eta
            l = 0.455 + 0.1*eta

            for i in range(1, int(runepoch + 1)):
                
                
                    
                if True:
                    
                    gene_batch, ppi_lap_mat, batch_index = get_gene_batch_approx_reg(128, n_gene, 
                                                                                          w, h, gene_data_norm, A)

                    feed_dict={gene_maps: gene_batch,
                            lap_mat: ppi_lap_mat,
                            alpha: 0.01,
                            u_thres: u,
                            l_thres: l,
                            lr: base_lr}

                train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
                
                # Update gene embedding matrix when fewer genes involved in the clustering
                

                if i % 20 == 0:
                    print('training loss at iter %d is %f' % (i, train_loss))
                    
            # Update step size
            eta += 1.1 * 0.009
            
            # Create checkpoint every 5 epochs
            if epoch % 5 == 0:  # save model at every 5 epochs
                model_name = 'CNN_PReg_ep_' + str(epoch) + '.ckpt'
                save_path = saver.save(sess, os.path.join(model_dir, model_name))
                print("Checkpoint created in file: %s" % save_path)

            epoch += 1

    with tf.Session() as sess:
        
        # Load the most recent checkpoint
        saver.restore(sess, os.path.join(model_dir, 'CNN_PReg_ep_45.ckpt'))
        
        # Infer gene memberships
        all_gene_clusters = np.zeros([n_gene], dtype=np.float32)
        for j in range(int(np.ceil(n_gene/128))):
            gene_batch = np.copy(gene_data_norm[128*j:128*(j+1), ...])
            feed_dict={gene_maps: gene_batch}
            all_gene_clusters[j*128:(j+1)*128] = sess.run(gene_clusters, feed_dict=feed_dict)

    data = pd.DataFrame({'gene_id': gene_ids, 'cluster_id': all_gene_clusters}, columns=['gene_id', 'cluster_id'])
   # data.to_csv(os.path.join(gene_cluster_dir,   'gene_clusters.csv'), index=False)
    #adata.varm['gene_map']=gene_data.reshape(gene_data.shape[0],gene_data.shape[1]*gene_data.shape[2])
    #adata.var['cluster_id']=data['cluster_id'].values
    return data
    #return data,gene_data,adata
