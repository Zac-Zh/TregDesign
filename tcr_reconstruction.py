import os
import pandas as pd
import numpy as np
from Bio import SeqIO, Seq
import logging
import json
import requests
from io import StringIO

# IMGT参考序列数据
# 实际应用中，这些数据应该从IMGT数据库下载或从本地文件加载
# 这里提供一个简化版的参考数据结构
IMGT_REFERENCE = {
    # V基因参考序列 - 格式: {基因名: {"sequence": 核苷酸序列, "frame": 阅读框架}}
    "V_GENES": {},
    # J基因参考序列
    "J_GENES": {},
    # CDR3区域定义 - V基因保守的Cys位置和J基因保守的Phe/Trp位置
    "CDR3_POSITIONS": {
        "V": {"TRA": 104, "TRB": 104},  # 大致位置，实际应根据具体基因调整
        "J": {"TRA": 118, "TRB": 118}   # 大致位置，实际应根据具体基因调整
    }
}


def download_imgt_reference_data(output_dir):
    """
    从IMGT数据库下载TCR参考序列数据
    
    参数:
    output_dir: 输出目录，用于保存下载的参考数据
    
    返回:
    reference_data: 包含V和J基因参考序列的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    reference_file = os.path.join(output_dir, "imgt_reference_data.json")
    
    # 如果已经存在参考数据文件，直接加载
    if os.path.exists(reference_file):
        logging.info(f"加载现有IMGT参考数据: {reference_file}")
        with open(reference_file, 'r') as f:
            return json.load(f)
    
    # 初始化参考数据结构
    reference_data = {
        "V_GENES": {},
        "J_GENES": {},
        "CDR3_POSITIONS": {
            "V": {"TRA": 104, "TRB": 104},
            "J": {"TRA": 118, "TRB": 118}
        }
    }
    
    try:
        # 下载人类TCR alpha链V基因
        logging.info("下载人类TCR alpha链V基因参考序列...")
        tra_v_url = "https://www.imgt.org/genedb/GENElect?query=7.1+TRAV&species=Homo+sapiens"
        response = requests.get(tra_v_url)
        if response.status_code == 200:
            # 解析FASTA格式数据
            for record in SeqIO.parse(StringIO(response.text), "fasta"):
                gene_name = record.id.split('|')[1]  # 提取基因名
                reference_data["V_GENES"][gene_name] = {
                    "sequence": str(record.seq),
                    "frame": 0  # 默认阅读框架
                }
        
        # 下载人类TCR beta链V基因
        logging.info("下载人类TCR beta链V基因参考序列...")
        trb_v_url = "https://www.imgt.org/genedb/GENElect?query=7.2+TRBV&species=Homo+sapiens"
        response = requests.get(trb_v_url)
        if response.status_code == 200:
            for record in SeqIO.parse(StringIO(response.text), "fasta"):
                gene_name = record.id.split('|')[1]
                reference_data["V_GENES"][gene_name] = {
                    "sequence": str(record.seq),
                    "frame": 0
                }
        
        # 下载人类TCR alpha链J基因
        logging.info("下载人类TCR alpha链J基因参考序列...")
        tra_j_url = "https://www.imgt.org/genedb/GENElect?query=7.1+TRAJ&species=Homo+sapiens"
        response = requests.get(tra_j_url)
        if response.status_code == 200:
            for record in SeqIO.parse(StringIO(response.text), "fasta"):
                gene_name = record.id.split('|')[1]
                reference_data["J_GENES"][gene_name] = {
                    "sequence": str(record.seq)
                }
        
        # 下载人类TCR beta链J基因
        logging.info("下载人类TCR beta链J基因参考序列...")
        trb_j_url = "https://www.imgt.org/genedb/GENElect?query=7.2+TRBJ&species=Homo+sapiens"
        response = requests.get(trb_j_url)
        if response.status_code == 200:
            for record in SeqIO.parse(StringIO(response.text), "fasta"):
                gene_name = record.id.split('|')[1]
                reference_data["J_GENES"][gene_name] = {
                    "sequence": str(record.seq)
                }
        
        # 保存参考数据
        with open(reference_file, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        logging.info(f"IMGT参考数据已保存: {reference_file}")
        return reference_data
        
    except Exception as e:
        logging.error(f"下载IMGT参考数据失败: {str(e)}")
        # 如果下载失败，使用模拟数据
        logging.warning("使用模拟IMGT参考数据")
        
        # 创建一些模拟的V和J基因序列
        # Alpha链V基因
        for i in range(1, 11):
            gene_name = f"TRAV{i}"
            reference_data["V_GENES"][gene_name] = {
                "sequence": "ATGGAGAAGGTGCTGGTCACCTTCTTCCTCCTGGGAGCAGGCCCAGTGGAGCAGCCA" + \
                           "ACATGCAGTGGAGCAGCCTCCTGCAGGTGACGGTGTCACAGCCCGATTCACAGCTG" + \
                           "AACTATCGCTGCAAAGCCTCAGACTCCCAGCCCAGTGACTCCGCTCTCTACTTCTG",
                "frame": 0
            }
        
        # Beta链V基因
        for i in range(1, 11):
            gene_name = f"TRBV{i}"
            reference_data["V_GENES"][gene_name] = {
                "sequence": "ATGGGCTGCAGGCTGCTCTGCTGTGTGGCCTTTTGTCTCCTGGGAGCAGGCCCAGT" + \
                           "GGAGCAGACTCCACAATCCTGAGCTGCACTGTGACATCGGCCCAAAAGAACCCGAC" + \
                           "AGAGCTGAAGTGCAAGTCTAATGAAAACGACAAGTGGGTCAGCAGCACTGCCTACA",
                "frame": 0
            }
        
        # Alpha链J基因
        for i in range(1, 11):
            gene_name = f"TRAJ{i}"
            reference_data["J_GENES"][gene_name] = {
                "sequence": "TGAATTATGGAGGAAGCCAAGGAAATCTCATCTTTGGAAAAGGAACCCGTGTGACT" + \
                           "GTGGAACCAA"
            }
        
        # Beta链J基因
        for i in range(1, 11):
            gene_name = f"TRBJ{i}"
            reference_data["J_GENES"][gene_name] = {
                "sequence": "CTAACTATGGCTACACCTTCGGTTCGGGGACCAGGTTAACCGTTGTAGCGACCCGCT" + \
                           "GTCCA"
            }
        
        # 保存模拟参考数据
        with open(reference_file, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        logging.info(f"模拟IMGT参考数据已保存: {reference_file}")
        return reference_data


def reconstruct_tcr_sequence(v_gene, j_gene, cdr3_aa, chain_type, reference_data=None):
    """
    从V/J基因名称和CDR3氨基酸序列重建完整的TCR序列
    
    参数:
    v_gene: V基因名称 (例如 "TRAV1-2")
    j_gene: J基因名称 (例如 "TRAJ33")
    cdr3_aa: CDR3区域的氨基酸序列
    chain_type: 链类型 ("alpha" 或 "beta")
    reference_data: IMGT参考数据，如果为None则使用默认数据
    
    返回:
    tcr_nt: 重建的TCR核苷酸序列
    tcr_aa: 重建的TCR氨基酸序列
    """
    # 如果没有提供参考数据，使用默认数据
    if reference_data is None:
        reference_data = IMGT_REFERENCE
    
    # 确定链类型前缀
    chain_prefix = "TRA" if chain_type.lower() == "alpha" else "TRB"
    
    # 获取V基因序列
    if v_gene not in reference_data["V_GENES"]:
        logging.warning(f"V基因 {v_gene} 在参考数据中未找到，使用模拟序列")
        v_seq = "ATGGAGAAGGTGCTGGTCACCTTCTTCCTCCTGGGAGCAGGCCCAGTGGAGCAGCCA" + \
                "ACATGCAGTGGAGCAGCCTCCTGCAGGTGACGGTGTCACAGCCCGATTCACAGCTG" + \
                "AACTATCGCTGCAAAGCCTCAGACTCCCAGCCCAGTGACTCCGCTCTCTACTTCTG"
        v_frame = 0
    else:
        v_seq = reference_data["V_GENES"][v_gene]["sequence"]
        v_frame = reference_data["V_GENES"][v_gene]["frame"]
    
    # 获取J基因序列
    if j_gene not in reference_data["J_GENES"]:
        logging.warning(f"J基因 {j_gene} 在参考数据中未找到，使用模拟序列")
        j_seq = "TGAATTATGGAGGAAGCCAAGGAAATCTCATCTTTGGAAAAGGAACCCGTGTGACT" + \
                "GTGGAACCAA" if chain_type.lower() == "alpha" else \
                "CTAACTATGGCTACACCTTCGGTTCGGGGACCAGGTTAACCGTTGTAGCGACCCGCT" + \
                "GTCCA"
    else:
        j_seq = reference_data["J_GENES"][j_gene]["sequence"]
    
    # 获取CDR3区域的位置
    v_cdr3_pos = reference_data["CDR3_POSITIONS"]["V"][chain_prefix]
    j_cdr3_pos = reference_data["CDR3_POSITIONS"]["J"][chain_prefix]
    
    # 从CDR3氨基酸序列反向翻译为核苷酸序列
    # 这里使用最常见的密码子进行简化翻译
    codon_table = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
        'G': 'GGT', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTT',
        'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT',
        'S': 'TCT', 'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT'
    }
    
    cdr3_nt = ''
    for aa in cdr3_aa:
        cdr3_nt += codon_table.get(aa, 'NNN')  # 对于未知氨基酸，使用NNN
    
    # 构建完整的TCR序列
    # V区域直到CDR3开始
    v_part = v_seq[:v_cdr3_pos*3]  # 转换为核苷酸位置
    
    # J区域从CDR3结束开始
    j_part = j_seq[j_cdr3_pos*3:]  # 转换为核苷酸位置
    
    # 组合完整序列
    tcr_nt = v_part + cdr3_nt + j_part
    
    # 确保序列长度是3的倍数（完整的密码子）
    if len(tcr_nt) % 3 != 0:
        tcr_nt = tcr_nt[:-(len(tcr_nt) % 3)]  # 移除多余的核苷酸
    
    # 翻译为氨基酸序列
    tcr_aa = str(Seq.Seq(tcr_nt).translate())
    
    return tcr_nt, tcr_aa


def reconstruct_tcr_sequences_batch(tcr_df, reference_dir):
    """
    批量重建TCR序列
    
    参数:
    tcr_df: 包含TCR信息的DataFrame，必须包含alpha_v, alpha_j, alpha_cdr3, beta_v, beta_j, beta_cdr3列
    reference_dir: 参考数据目录
    
    返回:
    updated_df: 更新后的DataFrame，包含重建的TCR序列
    """
    # 下载或加载参考数据
    reference_data = download_imgt_reference_data(reference_dir)
    
    # 创建结果列
    tcr_df['alpha_nt'] = ''
    tcr_df['alpha_aa'] = ''
    tcr_df['beta_nt'] = ''
    tcr_df['beta_aa'] = ''
    
    # 对每个TCR进行序列重建
    for idx, row in tcr_df.iterrows():
        try:
            # 重建alpha链序列
            if pd.notna(row['alpha_v']) and pd.notna(row['alpha_j']) and pd.notna(row['alpha_cdr3']):
                alpha_nt, alpha_aa = reconstruct_tcr_sequence(
                    row['alpha_v'], row['alpha_j'], row['alpha_cdr3'], 'alpha', reference_data
                )
                tcr_df.at[idx, 'alpha_nt'] = alpha_nt
                tcr_df.at[idx, 'alpha_aa'] = alpha_aa
            
            # 重建beta链序列
            if pd.notna(row['beta_v']) and pd.notna(row['beta_j']) and pd.notna(row['beta_cdr3']):
                beta_nt, beta_aa = reconstruct_tcr_sequence(
                    row['beta_v'], row['beta_j'], row['beta_cdr3'], 'beta', reference_data
                )
                tcr_df.at[idx, 'beta_nt'] = beta_nt
                tcr_df.at[idx, 'beta_aa'] = beta_aa
        except Exception as e:
            logging.error(f"重建TCR序列出错 (ID: {row.get('cell_id', idx)}): {str(e)}")
    
    return tcr_df


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试重建功能
    test_data = {
        'cell_id': ['test1', 'test2'],
        'alpha_v': ['TRAV1', 'TRAV2'],
        'alpha_j': ['TRAJ1', 'TRAJ2'],
        'alpha_cdr3': ['CAVSESPFGNEKLTF', 'CAVNNNAGNMLTF'],
        'beta_v': ['TRBV1', 'TRBV2'],
        'beta_j': ['TRBJ1', 'TRBJ2'],
        'beta_cdr3': ['CASSVGVGAYEQYF', 'CASSQDSSYEQYF']
    }
    
    test_df = pd.DataFrame(test_data)
    reference_dir = 'data/reference'
    
    # 重建序列
    result_df = reconstruct_tcr_sequences_batch(test_df, reference_dir)
    
    # 输出结果
    print(result_df[['cell_id', 'alpha_cdr3', 'alpha_nt', 'beta_cdr3', 'beta_nt']])
    
    print("TCR序列重建测试完成")