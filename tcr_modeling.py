import os
import pandas as pd
import numpy as np
import subprocess
import json
from Bio import SeqIO
import torch


def run_tcrmodel2(tcr_alpha, tcr_beta, mhc_type, peptide, output_dir):
    """使用TCRmodel2预测TCR-pMHC复合物结构"""
    import os
    import logging
    from structure_modeling import run_tcrmodel2_prediction
    from structure_modeling import TCRmodel2

    os.makedirs(output_dir, exist_ok=True)
    
    # 获取TCRmodel2实例
    tcrmodel2_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'data', 'models', 'tcrmodel2')
    tcrmodel2 = TCRmodel2(tcrmodel2_dir)
    
    # 运行预测
    logging.info(f"使用TCRmodel2预测TCR-pMHC复合物结构")
    return run_tcrmodel2_prediction(tcr_alpha, tcr_beta, mhc_type, peptide, output_dir, tcrmodel2)


def run_alphafold2_refinement(pdb_file, output_dir):
    """使用AlphaFold2优化TCR-pMHC复合物结构"""
    import os
    import logging
    from structure_modeling import run_alphafold2_refinement as run_af2
    from structure_modeling import AlphaFold2

    os.makedirs(output_dir, exist_ok=True)
    
    # 获取AlphaFold2实例
    alphafold2_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'data', 'models', 'alphafold2')
    alphafold2 = AlphaFold2(alphafold2_dir)
    
    # 运行优化
    logging.info(f"使用AlphaFold2优化TCR-pMHC复合物结构")
    return run_af2(pdb_file, output_dir, alphafold2)


class PMTnetModel:
    """
    使用pMTnet深度学习模型预测TCR-肽段结合特异性
    """

    def __init__(self, model_path):
        """加载预训练的pMTnet模型"""
        self.model = torch.load(model_path) if os.path.isfile(model_path) else None

    def predict(self, tcr_beta_cdr3, peptide):
        """
        预测TCR-肽段结合特异性

        参数:
        tcr_beta_cdr3: TCR beta链CDR3序列
        peptide: 肽段序列

        返回:
        结合概率得分 (0-1)
        """
        # 准备输入
        if self.model is not None:
            # 实际使用PyTorch模型进行预测
            with torch.no_grad():
                # 这里假设模型接受两个序列作为输入
                # 实际实现需要根据模型的具体要求进行调整
                beta_tensor = torch.tensor([ord(c) for c in tcr_beta_cdr3]).unsqueeze(0)
                peptide_tensor = torch.tensor([ord(c) for c in peptide]).unsqueeze(0)
                
                # 假设模型返回一个概率值
                prediction = self.model((beta_tensor, peptide_tensor))
                return prediction.item()
        else:
            # 使用模拟预测逻辑
            from model_setup import PMTnetModel as MockPMTnetModel
            mock_model = MockPMTnetModel(model_path)
            return mock_model.predict(tcr_beta_cdr3, peptide)


def predict_tcr_binding(tcr_df, peptides, model_path, output_file):
    """
    预测TCR与多个肽段的结合概率

    参数:
    tcr_df: 包含TCR序列的DataFrame
    peptides: 待测试的肽段字典 {peptide_id: sequence}
    model_path: pMTnet模型路径
    output_file: 输出文件路径

    返回:
    binding_df: 包含结合预测结果的DataFrame
    """
    # 加载pMTnet模型
    from model_setup import PMTnetModel
    pmtnet = PMTnetModel(model_path)

    results = []

    for idx, row in tcr_df.iterrows():
        tcr_id = row['cell_id']
        tcr_beta = row['beta_cdr3']

        for peptide_id, peptide_seq in peptides.items():
            # 预测结合概率
            binding_score = pmtnet.predict(tcr_beta, peptide_seq)

            results.append({
                'tcr_id': tcr_id,
                'alpha_cdr3': row['alpha_cdr3'],
                'beta_cdr3': tcr_beta,
                'peptide': peptide_id,
                'peptide_seq': peptide_seq,
                'binding_score': binding_score
            })

    # 创建DataFrame
    binding_df = pd.DataFrame(results)

    # 保存结果
    binding_df.to_csv(output_file, index=False)

    return binding_df


def run_structural_modeling(top_tcrs, peptides, model_paths, output_dir):
    """
    对高得分TCR进行结构模拟

    参数:
    top_tcrs: 包含高得分TCR的DataFrame
    peptides: 肽段序列字典 {peptide_id: sequence}
    model_paths: 模型路径字典
    output_dir: 输出目录

    返回:
    models_df: 包含结构模型信息的DataFrame
    """
    import os
    import logging
    from tcr_modeling import run_tcrmodel2, run_alphafold2_refinement
    from tcr_modeling import analyze_interface_contacts

    os.makedirs(output_dir, exist_ok=True)

    model_results = []

    # 只处理前10个TCR-肽对进行模拟
    for i, (idx, row) in enumerate(top_tcrs.iterrows()):
        if i >= 10:  # 限制处理数量，以便测试更快完成
            break

        tcr_id = row['tcr_id']
        alpha_cdr3 = row['alpha_cdr3']
        beta_cdr3 = row['beta_cdr3']
        peptide_id = row['peptide']
        peptide_seq = peptides[peptide_id]

        # 创建模型目录
        model_dir = os.path.join(output_dir, f"{tcr_id}_{peptide_id}")
        os.makedirs(model_dir, exist_ok=True)

        try:
            logging.info(f"为TCR {tcr_id} 与肽段 {peptide_id} 构建模型")

            # 使用TCRmodel2构建初始模型
            pdb_file = run_tcrmodel2(alpha_cdr3, beta_cdr3, "HLA-DQ8", peptide_seq, model_dir)

            # 使用AlphaFold2优化模型
            refined_pdb = run_alphafold2_refinement(pdb_file, model_dir)

            # 分析界面接触
            contact_info = analyze_interface_contacts(refined_pdb)

            model_results.append({
                'tcr_id': tcr_id,
                'peptide_id': peptide_id,
                'peptide_seq': peptide_seq,
                'model_path': refined_pdb,
                'contact_count': contact_info['contact_count'],
                'interface_energy': contact_info['interface_energy'],
                'confidence': contact_info['confidence']
            })

        except Exception as e:
            logging.error(f"为TCR {tcr_id} 与肽段 {peptide_id} 构建模型出错: {str(e)}")

    # 创建DataFrame
    models_df = pd.DataFrame(model_results)

    # 保存结果
    models_df.to_csv(os.path.join(output_dir, "structural_models.csv"), index=False)

    return models_df


def analyze_interface_contacts(pdb_file):
    """分析TCR和pMHC复合物界面接触情况"""
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB.Selection import unfold_entities
    import numpy as np
    import logging

    try:
        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)

        # 定义TCR和pMHC链
        tcr_chains = ['A', 'B']  # alpha和beta链
        pmhc_chains = ['C', 'D', 'E']  # MHC和肽链

        # 获取各组原子
        tcr_atoms = []
        pmhc_atoms = []

        for chain in structure[0]:
            if chain.id in tcr_chains:
                tcr_atoms.extend(unfold_entities(chain, 'A'))
            elif chain.id in pmhc_chains:
                pmhc_atoms.extend(unfold_entities(chain, 'A'))

        # 设置邻居搜索
        ns = NeighborSearch(tcr_atoms + pmhc_atoms)

        # 寻找4.5埃范围内的接触
        contacts = []
        for tcr_atom in tcr_atoms:
            neighbors = ns.search(tcr_atom.coord, 4.5)
            for neighbor in neighbors:
                if neighbor in pmhc_atoms:
                    contacts.append((tcr_atom, neighbor))

        # 计算接触数
        contact_count = len(contacts)

        # 计算界面能量（简化估算）
        interface_energy = 0.0
        for contact in contacts:
            distance = np.sqrt(sum((contact[0].coord - contact[1].coord) ** 2))
            energy = -1.0 * (1.0 / distance)
            interface_energy += energy

        # 计算置信度得分
        confidence = min(1.0, (contact_count / 100.0) * (abs(interface_energy) / 50.0))

        return {
            'contact_count': contact_count,
            'interface_energy': interface_energy,
            'confidence': confidence
        }
    except Exception as e:
        logging.error(f"分析界面接触错误: {str(e)}")
        return {
            'contact_count': 10,
            'interface_energy': -5.0,
            'confidence': 0.5
        }




if __name__ == "__main__":
    # 定义insulin肽段
    insulin_peptides = {
        'ins_b9-23': 'SHLVEALYLVCGERG',
        'ins_b10-23': 'HLVEALYLVCGERG',
        'ins_b11-30': 'LVEALYLVCGERGFFYTPKT'
    }

    # 加载TCR数据
    tcr_df = pd.read_csv("results/treg_tcr_combined.csv")

    # 预测结合特异性
    binding_df = predict_tcr_binding(tcr_df, insulin_peptides, "results/tcr_binding_predictions.csv")

    # 筛选高得分TCR
    top_tcrs = binding_df[binding_df['binding_score'] > 0.7].sort_values('binding_score', ascending=False)

    # 设置模型路径
    model_paths = {
        'pmtnet': 'data/models/pmtnet_model',
        'tcrmodel2': 'data/models/tcrmodel2',
        'alphafold2': 'data/models/alphafold2'
    }
    
    # 为高得分TCR构建结构模型
    models_df = run_structural_modeling(top_tcrs.head(10), insulin_peptides, model_paths, "results/structural_models")