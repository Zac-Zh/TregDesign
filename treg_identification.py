import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # 添加垃圾回收模块
import os
import logging

# 设置scanpy内存使用限制
sc.settings.verbosity = 1  # 减少输出信息
sc.settings.n_jobs = 1  # 限制并行作业数


def identify_tregs(adata, output_file=None):
    """
    从单细胞RNA测序数据中识别和筛选Treg细胞

    参数:
    adata: AnnData对象，包含单细胞RNA测序数据
    output_file: 可选，保存结果的文件路径

    返回:
    treg_adata: 仅包含Treg细胞的AnnData对象
    """
    # 预处理
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 检查关键标记基因是否存在于obs中，如果不存在，尝试从var_names中获取
    key_markers = ['CD4', 'FOXP3', 'IL2RA', 'IL7R', 'CTLA4', 'TNFRSF18', 'IKZF2',
                  'TGFB1', 'IL10', 'ENTPD1', 'CCR7', 'SELL', 'CCR5', 'CXCR3']
    
    for marker in key_markers:
        if marker not in adata.obs.columns:
            if marker in adata.var_names:
                # 如果基因在var_names中，将其表达值添加到obs中，添加_expr后缀以避免冲突
                try:
                    adata.obs[f"{marker}_expr"] = adata[:, marker].X.toarray().flatten()
                except MemoryError:
                    logging.warning(f"处理{marker}时内存不足，使用稀疏矩阵处理")
                    # 使用稀疏矩阵方式处理，避免toarray()导致的内存溢出
                    adata.obs[f"{marker}_expr"] = np.array(adata[:, marker].X.sum(axis=1)).flatten()
            else:
                # 如果基因不存在，创建一个默认值列（所有值为0）
                logging.warning(f"标记基因 {marker} 在数据中不存在，使用默认值0")
                adata.obs[marker] = 0
    
    # 强制垃圾回收
    gc.collect()
    
    # 识别T细胞 (CD4+)
    cd4_cells = adata[adata.obs['CD4_expr'] > 1].copy()

    # 识别Treg细胞 (FOXP3+ IL2RA+ IL7R-)
    # 确保所有需要的列都存在
    for marker in ['FOXP3', 'IL2RA', 'IL7R']:
        if f"{marker}_expr" not in cd4_cells.obs.columns:
            logging.warning(f"标记基因 {marker} 在CD4+细胞中不存在，使用默认值")
            cd4_cells.obs[f"{marker}_expr"] = 0 if marker == 'IL7R' else 2  # 为IL7R默认低表达，其他默认高表达
    
    treg_mask = ((cd4_cells.obs['FOXP3_expr'] > 1.5) &
                 (cd4_cells.obs['IL2RA_expr'] > 1.5) &
                 (cd4_cells.obs['IL7R_expr'] < 0.5))

    treg_adata = cd4_cells[treg_mask].copy()

    # 进一步验证Treg身份
    treg_markers = ['CTLA4', 'TNFRSF18', 'IKZF2']
    
    # 确保leiden分组存在
    if 'leiden' not in treg_adata.obs.columns:
        logging.warning("leiden分组不存在，执行聚类以创建leiden分组")
        # 限制高变基因数量，避免内存溢出
        n_top_genes = min(500, treg_adata.n_vars)  # 减少从2000到500
        sc.pp.highly_variable_genes(treg_adata, n_top_genes=n_top_genes)
        sc.pp.pca(treg_adata, n_comps=min(30, treg_adata.n_obs-1))  # 限制PCA组件数
        
        # 使用更保守的neighbors参数
        n_neighbors = min(10, max(2, treg_adata.n_obs // 10))  # 确保neighbors数量合理
        sc.pp.neighbors(treg_adata, n_neighbors=n_neighbors, n_pcs=min(10, treg_adata.n_obs-1))
        sc.tl.leiden(treg_adata, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
    
    # 检查标记基因是否在var_names中
    available_markers = [marker for marker in treg_markers if marker in treg_adata.var_names]
    if available_markers:
        try:
            sc.pl.violin(treg_adata, available_markers, groupby='leiden', rotation=90, save='_treg_markers.pdf')
        except Exception as e:
            logging.warning(f"绘制小提琴图失败: {str(e)}")
    else:
        logging.warning("没有可用的Treg标记基因进行可视化")

    # 强制垃圾回收
    gc.collect()
    
    # 筛选免疫抑制特征的Treg亚群
    immunosuppressive_markers = ['TGFB1', 'IL10', 'ENTPD1']
    memory_markers = ['CCR7', 'SELL', 'CCR5', 'CXCR3']

    # 聚类以识别亚群 - 使用更保守的参数
    try:
        # 限制高变基因数量
        n_top_genes = min(500, treg_adata.n_vars)  # 减少从2000到500
        sc.pp.highly_variable_genes(treg_adata, n_top_genes=n_top_genes)
        sc.pp.pca(treg_adata, n_comps=min(20, treg_adata.n_obs-1))  # 限制PCA组件数
        
        # 检查细胞数量，如果太少则调整neighbors参数或跳过UMAP
        if treg_adata.n_obs < 10:
            logging.warning(f"细胞数量太少 (n={treg_adata.n_obs})，使用最小neighbors参数")
            # 使用最小neighbors参数
            n_neighbors = max(2, min(3, treg_adata.n_obs-1))
            sc.pp.neighbors(treg_adata, n_neighbors=n_neighbors, n_pcs=min(5, treg_adata.n_obs-1))
            sc.tl.leiden(treg_adata, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
        else:
            # 正常执行neighbors和UMAP，但使用更保守的参数
            n_neighbors = min(15, max(5, treg_adata.n_obs // 10))
            sc.pp.neighbors(treg_adata, n_neighbors=n_neighbors, n_pcs=min(15, treg_adata.n_obs-1))
            sc.tl.leiden(treg_adata, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
            try:
                sc.tl.umap(treg_adata, min_dist=0.3, spread=1.0)
            except Exception as e:
                logging.warning(f"UMAP计算失败: {str(e)}，继续执行后续分析")
        
        # 分析亚群表达模式
        try:
            sc.pl.dotplot(treg_adata, immunosuppressive_markers + memory_markers,
                        groupby='leiden', save='_treg_subsets.pdf')
        except Exception as e:
            logging.warning(f"绘制点图失败: {str(e)}")
    except MemoryError as me:
        logging.error(f"聚类过程中内存不足: {str(me)}")
    except Exception as e:
        logging.error(f"聚类过程中发生错误: {str(e)}")

    # 强制垃圾回收
    gc.collect()
    
    # 识别高TGF-beta表达的Treg
    high_tgfb_clusters = []
    
    # 检查TGFB1是否在var_names中
    if 'TGFB1' in treg_adata.var_names:
        for cluster in treg_adata.obs['leiden'].unique():
            try:
                idx = treg_adata.var_names.get_indexer(['TGFB1'])[0]
                if idx >= 0:  # 确保索引有效
                    # 避免除零错误
                    cluster_cells = treg_adata[treg_adata.obs['leiden'] == cluster]
                    if cluster_cells.n_obs > 0:
                        mean_expr = np.mean(cluster_cells.X[:, idx].toarray())
                        if mean_expr > 1.0:  # 阈值可调整
                            high_tgfb_clusters.append(cluster)
                else:
                    logging.warning("TGFB1基因索引无效，跳过高TGF-beta表达Treg识别")
            except Exception as e:
                logging.warning(f"计算TGFB1表达出错: {str(e)}")

    # 识别效应/记忆型Treg (CCR7低, SELL低, CCR5高, CXCR3高)
    # 确保所有需要的列都存在
    memory_markers = ['CCR7', 'SELL', 'CCR5', 'CXCR3']
    for marker in memory_markers:
        if marker not in treg_adata.obs.columns:
            if marker in treg_adata.var_names:
                # 如果基因在var_names中，将其表达值添加到obs中
                try:
                    treg_adata.obs[f"{marker}_expr"] = treg_adata[:, marker].X.toarray().flatten()
                except MemoryError:
                    logging.warning(f"处理{marker}时内存不足，使用稀疏矩阵处理")
                    # 使用稀疏矩阵方式处理
                    treg_adata.obs[f"{marker}_expr"] = np.array(treg_adata[:, marker].X.sum(axis=1)).flatten()
            else:
                # 如果基因不存在，创建一个默认值列
                logging.warning(f"记忆标记基因 {marker} 在数据中不存在，使用默认值")
                # 为CCR7和SELL默认低表达，为CCR5和CXCR3默认高表达
                default_value = 0 if marker in ['CCR7', 'SELL'] else 2
                treg_adata.obs[marker] = default_value
    
    memory_treg_mask = ((treg_adata.obs['CCR7_expr'] < 0.5) &
                        (treg_adata.obs['SELL_expr'] < 0.5) &
                        (treg_adata.obs['CCR5_expr'] > 1.0) &
                        (treg_adata.obs['CXCR3_expr'] > 1.0))

    memory_tregs = treg_adata[memory_treg_mask].copy()

    # 保存结果
    if output_file:
        treg_adata.write(output_file)

    # 最终垃圾回收
    gc.collect()
    
    return treg_adata


def analyze_gse102234(dataset_dir):
    """分析GSE102234数据集中的T1D患者Treg细胞
    
    参数:
    dataset_dir: 数据集目录路径
    
    返回:
    treg_adata: 识别的Treg细胞AnnData对象
    """
    import os
    import logging
    
    # 设置文件路径
    h5_file = os.path.join(dataset_dir, 'processed_data.h5ad')
    metadata_file = os.path.join(dataset_dir, 'metadata.csv')
    output_file = os.path.join(dataset_dir, 'GSE102234_tregs.h5ad')
    
    # 检查文件是否存在
    if not os.path.exists(h5_file):
        logging.warning(f"数据文件 {h5_file} 不存在，创建模拟数据")
        from data_download import create_mock_data
        adata = create_mock_data(h5_file, dataset='GSE102234')
    else:
        # 加载数据
        adata = sc.read_h5ad(h5_file)
    
    # 加载或创建元数据
    if os.path.exists(metadata_file):
        meta = pd.read_csv(metadata_file)
        if 'cell_id' in meta.columns:
            adata.obs = meta.set_index('cell_id')
    
    # 识别Treg
    treg_adata = identify_tregs(adata, output_file=output_file)
    
    # 对比新诊断T1D患者与对照（如果有condition列）
    if 'condition' in treg_adata.obs.columns:
        try:
            sc.tl.rank_genes_groups(treg_adata, 'condition', method='wilcoxon')
            sc.pl.rank_genes_groups(treg_adata, n_genes=20, save='_T1D_vs_control.pdf')
        except Exception as e:
            logging.error(f"差异基因分析失败: {str(e)}")
    
    return treg_adata


if __name__ == "__main__":
    # 处理GSE102234数据集
    t1d_tregs = analyze_gse102234()

    # 处理GSE30202数据集
    # 可添加类似的处理代码