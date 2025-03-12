import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from tcr_reconstruction import reconstruct_tcr_sequences_batch


def integrate_tcr_predictions(binding_df, models_df, treg_phenotype_df, output_file, reference_dir='data/reference'):
    """
    整合不同预测结果，对TCR进行排序和选择

    参数:
    binding_df: 结合预测结果
    models_df: 结构模型结果
    treg_phenotype_df: Treg表型数据
    output_file: 输出文件路径
    reference_dir: IMGT参考数据目录，用于重建TCR序列

    返回:
    ranked_tcrs: 排序后的TCR候选清单
    """
    # 检查并确保binding_df中有tcr_id列
    if 'tcr_id' not in binding_df.columns and 'cell_id' in binding_df.columns:
        binding_df['tcr_id'] = binding_df['cell_id']
    
    # 检查并确保models_df中有tcr_id列
    if models_df is not None and len(models_df) > 0:
        if 'tcr_id' not in models_df.columns and 'cell_id' in models_df.columns:
            models_df['tcr_id'] = models_df['cell_id']
    
    # 检查并确保treg_phenotype_df中有cell_id列作为主键
    if 'cell_id' not in treg_phenotype_df.columns:
        raise ValueError("Treg表型数据中缺少'cell_id'列")
    
    # 合并结构模型和结合预测结果
    if models_df is not None and len(models_df) > 0:
        merged_df = pd.merge(
            binding_df,
            models_df,
            left_on=['tcr_id', 'peptide'],
            right_on=['tcr_id', 'peptide_id'],
            how='left'
        )
    else:
        merged_df = binding_df.copy()
    
    # 合并Treg表型数据 - 使用明确的left_on和right_on参数
    final_df = pd.merge(
        merged_df,
        treg_phenotype_df,
        left_on='tcr_id',
        right_on='cell_id',
        how='left'
    )

    # 确保final_df中有tcr_id列
    if 'tcr_id' not in final_df.columns and 'cell_id' in final_df.columns:
        final_df['tcr_id'] = final_df['cell_id']
    elif 'cell_id' in final_df.columns and 'tcr_id' in final_df.columns:
        # 如果两列都存在，确保tcr_id列有值
        final_df['tcr_id'] = final_df['tcr_id'].fillna(final_df['cell_id'])

    # 标准化评分指标
    scaler = MinMaxScaler()
    for column in ['binding_score', 'contact_count', 'interface_energy', 'confidence']:
        if column in final_df.columns and not final_df[column].isnull().all():
            # 确保列中有非空值
            final_df[f'{column}_scaled'] = scaler.fit_transform(final_df[[column]].fillna(0))
        else:
            # 如果列不存在或全为空值，创建默认值
            final_df[f'{column}_scaled'] = 0

    # 计算综合评分
    final_df['composite_score'] = (
            final_df['binding_score_scaled'] * 0.4 +
            final_df.get('confidence_scaled', 0) * 0.3 +
            final_df.get('contact_count_scaled', 0) * 0.15 +
            final_df.get('interface_energy_scaled', 0) * 0.15
    )

    # Treg表型加分
    if 'FOXP3' in final_df.columns and 'TGFB1' in final_df.columns and 'IL10' in final_df.columns:
        final_df['treg_quality'] = (
                final_df['FOXP3'] * 0.4 +
                final_df['TGFB1'] * 0.3 +
                final_df['IL10'] * 0.3
        )

        # 归一化Treg质量评分
        final_df['treg_quality_scaled'] = scaler.fit_transform(final_df[['treg_quality']])

        # 更新综合评分，加入Treg表型考量
        final_df['final_score'] = final_df['composite_score'] * 0.7 + final_df['treg_quality_scaled'] * 0.3
    else:
        final_df['final_score'] = final_df['composite_score']

    # 重建TCR序列
    logging.info("重建TCR完整序列...")
    # 确保DataFrame包含必要的列
    # 检查V和J基因列
    v_j_cols = ['alpha_v', 'alpha_j', 'beta_v', 'beta_j']
    missing_v_j_cols = [col for col in v_j_cols if col not in final_df.columns]
    
    # 检查CDR3列，考虑可能的列名变体
    has_alpha_cdr3 = any(col in final_df.columns for col in ['alpha_cdr3', 'alpha_cdr3_x', 'alpha_cdr3_y'])
    has_beta_cdr3 = any(col in final_df.columns for col in ['beta_cdr3', 'beta_cdr3_x', 'beta_cdr3_y'])
    
    if not missing_v_j_cols and has_alpha_cdr3 and has_beta_cdr3:
        try:
            # 重建TCR序列
            final_df = reconstruct_tcr_sequences_batch(final_df, reference_dir)
            logging.info("TCR序列重建完成")
            
            # 检查是否成功生成了核苷酸序列
            alpha_nt_count = final_df['alpha_nt'].notna().sum()
            beta_nt_count = final_df['beta_nt'].notna().sum()
            logging.info(f"成功重建 {alpha_nt_count} 个alpha链和 {beta_nt_count} 个beta链核苷酸序列")
            
        except Exception as e:
            logging.error(f"TCR序列重建失败: {str(e)}")
    else:
        if missing_v_j_cols:
            logging.warning(f"缺少重建TCR序列所需的V/J基因列: {missing_v_j_cols}")
        if not has_alpha_cdr3:
            logging.warning("缺少alpha链CDR3序列信息")
        if not has_beta_cdr3:
            logging.warning("缺少beta链CDR3序列信息")
        
        logging.info("尝试使用可用的CDR3信息进行序列重建...")
        try:
            # 即使缺少一些列，也尝试重建可能的序列
            final_df = reconstruct_tcr_sequences_batch(final_df, reference_dir)
            logging.info("部分TCR序列重建完成")
        except Exception as e:
            logging.error(f"部分TCR序列重建失败: {str(e)}")
    
    # 根据最终评分排序
    ranked_tcrs = final_df.sort_values('final_score', ascending=False)

    # 保存排序结果
    ranked_tcrs.to_csv(output_file, index=False)

    # 可视化顶部候选TCR
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tcr_id', y='final_score', data=ranked_tcrs.head(10))
    plt.title('Top 10 TCR Candidates')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file.replace('.csv', '_top10.pdf'))

    return ranked_tcrs


def generate_tcr_report(ranked_tcrs, output_file, top_n=20):
    """
    生成TCR候选报告

    参数:
    ranked_tcrs: 排序后的TCR候选清单
    output_file: 输出文件路径
    top_n: 报告中包含的顶部候选数量
    """
    # 选择顶部候选
    top_tcrs = ranked_tcrs.head(top_n)

    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TCR候选报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .score-high {{ color: #27ae60; font-weight: bold; }}
            .score-medium {{ color: #f39c12; }}
            .score-low {{ color: #e74c3c; }}
            .sequence {{ font-family: monospace; font-size: 0.85em; word-break: break-all; }}
        </style>
    </head>
    <body>
        <h1>HLA-DQ8-胰岛素特异性Treg TCR候选</h1>
        <p>以下是排名前{top_n}的TCR候选，按结合特异性和Treg功能排序：</p>
        <table>
            <tr>
                <th>排名</th>
                <th>TCR ID</th>
                <th>α链CDR3</th>
                <th>α链核苷酸序列</th>
                <th>β链CDR3</th>
                <th>β链核苷酸序列</th>
                <th>肽段</th>
                <th>结合评分</th>
                <th>结构评分</th>
                <th>Treg质量</th>
                <th>总评分</th>
            </tr>
    """

    # 添加每个TCR的行
    for i, (idx, row) in enumerate(top_tcrs.iterrows()):
        # 确定评分颜色类
        binding_class = "score-high" if row.get('binding_score', 0) > 0.7 else "score-medium" if row.get('binding_score', 0) > 0.4 else "score-low"
        final_class = "score-high" if row['final_score'] > 0.7 else "score-medium" if row['final_score'] > 0.4 else "score-low"
        
        # 获取CDR3序列，处理可能的列名不一致
        alpha_cdr3 = row.get('alpha_cdr3_y', row.get('alpha_cdr3_x', row.get('alpha_cdr3', 'N/A')))
        beta_cdr3 = row.get('beta_cdr3_y', row.get('beta_cdr3_x', row.get('beta_cdr3', 'N/A')))
        
        # 获取核苷酸序列
        alpha_nt = row.get('alpha_nt', 'N/A')
        beta_nt = row.get('beta_nt', 'N/A')
        
        # 如果序列太长，添加省略号
        if len(str(alpha_nt)) > 50 and alpha_nt != 'N/A':
            alpha_nt_display = f"{alpha_nt[:50]}...（共{len(alpha_nt)}个碱基）"
        else:
            alpha_nt_display = alpha_nt
            
        if len(str(beta_nt)) > 50 and beta_nt != 'N/A':
            beta_nt_display = f"{beta_nt[:50]}...（共{len(beta_nt)}个碱基）"
        else:
            beta_nt_display = beta_nt
        
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{row.get('tcr_id', 'N/A')}</td>
                <td>{alpha_cdr3}</td>
                <td class="sequence">{alpha_nt_display}</td>
                <td>{beta_cdr3}</td>
                <td class="sequence">{beta_nt_display}</td>
                <td>{row.get('peptide', 'N/A')}</td>
                <td class="{binding_class}">{row.get('binding_score', 0):.3f}</td>
                <td>{row.get('confidence', 0):.3f}</td>
                <td>{row.get('treg_quality', 0):.3f}</td>
                <td class="{final_class}">{row['final_score']:.3f}</td>
            </tr>
        """

    # 完成HTML
    html_content += """
        </table>
        <p>注：结合评分表示TCR与肽段的结合亲和力；结构评分表示TCR-pMHC复合物的结构质量；Treg质量表示调节性T细胞的功能特性。</p>
        <p>注：为了显示清晰，较长的核苷酸序列已被截断，完整序列可在CSV文件中查看。</p>
    </body>
    </html>
    """

    # 保存HTML报告
    with open(output_file, 'w') as f:
        f.write(html_content)

    return output_file


if __name__ == "__main__":
    # 加载各种预测结果
    binding_df = pd.read_csv("results/tcr_binding_predictions.csv")
    models_df = pd.read_csv("results/structural_models/structural_models.csv")
    treg_phenotype_df = pd.read_csv("results/treg_tcr_combined.csv")

    # 整合结果并排序
    ranked_tcrs = integrate_tcr_predictions(
        binding_df,
        models_df,
        treg_phenotype_df,
        "results/ranked_tcr_candidates.csv"
    )

    # 生成报告
    generate_tcr_report(ranked_tcrs, "results/tcr_candidates_report.html")

    print(f"Analysis complete. Found {len(ranked_tcrs)} candidate TCRs.")
    print(f"Top candidate: TCR {ranked_tcrs.iloc[0]['tcr_id']} with score {ranked_tcrs.iloc[0]['final_score']:.3f}")