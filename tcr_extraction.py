import os
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO, Seq


def run_trapes(fastq_file, output_dir, genome_ref, tcr_ref):
    """
    使用TRAPeS从单细胞RNA-seq数据中重建TCR序列

    参数:
    fastq_file: 输入的FASTQ文件
    output_dir: 输出目录
    genome_ref: 参考基因组路径
    tcr_ref: TCR基因片段注释路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 构建TRAPeS命令
    cmd = [
        "python", "TRAPeS.py",
        "--fastq", fastq_file,
        "--single-end",  # 或 --paired-end，取决于数据
        "--genome", genome_ref,
        "--tcr-reference", tcr_ref,
        "--output", output_dir
    ]

    # 运行TRAPeS
    subprocess.run(cmd, check=True)

    # 返回结果文件路径
    return os.path.join(output_dir, "reconstructed_tcrs.tsv")


def extract_tcr_sequences(scrnaseq_data, output_dir):
    """
    从单细胞RNA-seq数据中提取TCR序列

    参数:
    scrnaseq_data: 包含单细胞数据的目录
    output_dir: 输出目录

    返回:
    tcr_df: 包含细胞ID和对应TCR序列的DataFrame
    """
    # 获取所有细胞FASTQ文件
    cells = [f.split('.')[0] for f in os.listdir(scrnaseq_data) if f.endswith('.fastq.gz')]

    # 设置参考文件路径
    genome_ref = "reference/GRCh38.primary_assembly.genome.fa"
    tcr_ref = "reference/tcr_reference.fa"

    # 存储结果
    tcr_results = []

    # 对每个细胞运行TRAPeS
    for cell in cells:
        cell_fastq = os.path.join(scrnaseq_data, f"{cell}.fastq.gz")
        cell_output = os.path.join(output_dir, cell)

        try:
            tcr_file = run_trapes(cell_fastq, cell_output, genome_ref, tcr_ref)

            # 解析TCR结果
            if os.path.exists(tcr_file):
                tcr_data = pd.read_csv(tcr_file, sep='\t')

                # 提取TCR信息
                tcr_alpha = tcr_data[tcr_data['chain'] == 'alpha']
                tcr_beta = tcr_data[tcr_data['chain'] == 'beta']

                if not tcr_alpha.empty and not tcr_beta.empty:
                    alpha_v = tcr_alpha['v_gene'].iloc[0]
                    alpha_j = tcr_alpha['j_gene'].iloc[0]
                    alpha_cdr3 = tcr_alpha['cdr3_aa'].iloc[0]

                    beta_v = tcr_beta['v_gene'].iloc[0]
                    beta_j = tcr_beta['j_gene'].iloc[0]
                    beta_cdr3 = tcr_beta['cdr3_aa'].iloc[0]

                    # 质量控制
                    if (alpha_cdr3 and beta_cdr3 and
                            len(alpha_cdr3) >= 8 and len(beta_cdr3) >= 8 and
                            '*' not in alpha_cdr3 and '*' not in beta_cdr3):
                        tcr_results.append({
                            'cell_id': cell,
                            'alpha_v': alpha_v,
                            'alpha_j': alpha_j,
                            'alpha_cdr3': alpha_cdr3,
                            'beta_v': beta_v,
                            'beta_j': beta_j,
                            'beta_cdr3': beta_cdr3
                        })

        except Exception as e:
            print(f"Error processing cell {cell}: {str(e)}")

    # 创建DataFrame并保存结果
    tcr_df = pd.DataFrame(tcr_results)
    tcr_df.to_csv(os.path.join(output_dir, "all_tcr_sequences.csv"), index=False)

    return tcr_df


def merge_tcr_with_treg_data(tcr_df, treg_adata, output_file):
    """
    将TCR序列数据与Treg表型数据合并

    参数:
    tcr_df: 包含TCR序列的DataFrame
    treg_adata: 包含Treg细胞的AnnData对象
    output_file: 输出文件路径

    返回:
    combined_df: 合并后的DataFrame
    """
    # 提取Treg表型数据
    treg_df = pd.DataFrame({
        'cell_id': treg_adata.obs.index,
        'leiden_cluster': treg_adata.obs['leiden'],
        'FOXP3': treg_adata[:, treg_adata.var_names.get_indexer(['FOXP3'])[0]].X.toarray().flatten(),
        'IL2RA': treg_adata[:, treg_adata.var_names.get_indexer(['IL2RA'])[0]].X.toarray().flatten(),
        'TGFB1': treg_adata[:, treg_adata.var_names.get_indexer(['TGFB1'])[0]].X.toarray().flatten(),
        'IL10': treg_adata[:, treg_adata.var_names.get_indexer(['IL10'])[0]].X.toarray().flatten(),
        'ENTPD1': treg_adata[:, treg_adata.var_names.get_indexer(['ENTPD1'])[0]].X.toarray().flatten()
    })

    # 合并数据
    combined_df = pd.merge(treg_df, tcr_df, on='cell_id', how='inner')

    # 保存结果
    combined_df.to_csv(output_file, index=False)

    return combined_df


if __name__ == "__main__":
    # 提取TCR序列
    tcr_df = extract_tcr_sequences("data/GSE102234_fastq", "results/tcr_output")

    # 加载Treg数据
    import scanpy as sc

    treg_adata = sc.read_h5ad("results/GSE102234_tregs.h5ad")

    # 合并TCR和Treg数据
    combined_df = merge_tcr_with_treg_data(tcr_df, treg_adata, "results/treg_tcr_combined.csv")

    print(f"Found {len(combined_df)} Treg cells with paired TCR sequences")