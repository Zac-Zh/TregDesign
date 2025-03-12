import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def integrate_tcr_predictions(binding_df, models_df, treg_phenotype_df, output_file):
    """
    整合不同预测结果，对TCR进行排序和选择

    参数:
    binding_df: 结合预测结果
    models_df: 结构模型结果
    treg_phenotype_df: Treg表型数据
    output_file: 输出文件路径

    返回:
    ranked_tcrs: 排序后的TCR候选清单
    """
    # 合并结构模型和结合预测结果
    merged_df = pd.merge(
        binding_df,
        models_df,
        left_on=['tcr_id', 'peptide'],
        right_on=['tcr_id', 'peptide_id'],
        how='inner'
    )

    # 合并Treg表型数据
    final_df = pd.merge(
        merged_df,
        treg_phenotype_df,
        left_on='tcr_id',
        right_on='cell_id',
        how='inner'
    )

    # 标准化评分指标
    scaler = MinMaxScaler()
    for column in ['binding_score', 'contact_count', 'interface_energy', 'confidence']:
        if column in final_df.columns:
            final_df[f'{column}_scaled'] = scaler.fit_transform(final_df[[column]])

    # 计算综合评分
    final_df['composite_score'] = (
            final_df['binding_score_scaled'] * 0.4 +
            final_df['confidence_scaled'] * 0.3 +
            final_df['contact_count_scaled'] * 0.15 +
            final_df['interface_energy_scaled'] * 0.15
    )

    # Treg表型加分
    if 'TGFB1' in final_df.columns and 'IL10' in final_df.columns:
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
    # 选择前N个候选TCR
    top_tcrs = ranked_tcrs.head(top_n)

    # 创建HTML报告
    html = "<html><head><title>HLA-DQ8-Insulin Specific Treg TCR Candidates</title>"
    html += "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse;width:100%}"
    html += "th,td{border:1px solid #ddd;padding:8px} tr:nth-child(even){background-color:#f2f2f2}"
    html += "th{padding-top:12px;padding-bottom:12px;text-align:left;background-color:#4CAF50;color:white}</style></head>"
    html += "<body><h1>HLA-DQ8-Insulin Specific Treg TCR Candidates</h1>"

    # 添加摘要表格
    html += "<h2>Top " + str(top_n) + " TCR Candidates Summary</h2>"
    html += "<table><tr><th>Rank</th><th>TCR ID</th><th>Alpha CDR3</th><th>Beta CDR3</th>"
    html += "<th>Peptide</th><th>Binding Score</th><th>Final Score</th></tr>"

    for i, (_, row) in enumerate(top_tcrs.iterrows()):
        # 使用alpha_cdr3_x字段，如果不存在则尝试使用alpha_cdr3_y，如果都不存在则使用alpha_cdr3
        alpha_cdr3 = row.get('alpha_cdr3_x', row.get('alpha_cdr3_y', row.get('alpha_cdr3', 'N/A')))
        beta_cdr3 = row.get('beta_cdr3_x', row.get('beta_cdr3_y', row.get('beta_cdr3', 'N/A')))
        
        html += f"<tr><td>{i + 1}</td><td>{row['tcr_id']}</td><td>{alpha_cdr3}</td>"
        html += f"<td>{beta_cdr3}</td><td>{row['peptide']}</td>"
        html += f"<td>{row['binding_score']:.3f}</td><td>{row['final_score']:.3f}</td></tr>"

    html += "</table>"

    # 添加详细信息
    html += "<h2>Detailed Information for Top Candidates</h2>"

    for i, (_, row) in enumerate(top_tcrs.head(5).iterrows()):
        html += f"<h3>{i + 1}. TCR {row['tcr_id']}</h3>"
        html += "<h4>Sequence Information</h4>"
        # 使用alpha_cdr3_x字段，如果不存在则尝试使用alpha_cdr3_y，如果都不存在则使用alpha_cdr3
        alpha_cdr3 = row.get('alpha_cdr3_x', row.get('alpha_cdr3_y', row.get('alpha_cdr3', 'N/A')))
        beta_cdr3 = row.get('beta_cdr3_x', row.get('beta_cdr3_y', row.get('beta_cdr3', 'N/A')))
        
        html += f"<p><strong>Alpha Chain CDR3:</strong> {alpha_cdr3}</p>"
        html += f"<p><strong>Beta Chain CDR3:</strong> {beta_cdr3}</p>"

        html += "<h4>Binding Prediction</h4>"
        html += f"<p><strong>Peptide:</strong> {row['peptide']}</p>"
        html += f"<p><strong>Binding Score:</strong> {row['binding_score']:.3f}</p>"

        html += "<h4>Structural Model</h4>"
        html += f"<p><strong>Contact Count:</strong> {row['contact_count']}</p>"
        html += f"<p><strong>Interface Energy:</strong> {row['interface_energy']:.2f}</p>"
        html += f"<p><strong>Model Confidence:</strong> {row['confidence']:.3f}</p>"

        if 'FOXP3' in row:
            html += "<h4>Treg Phenotype</h4>"
            html += f"<p><strong>FOXP3 Expression:</strong> {row['FOXP3']:.2f}</p>"
            html += f"<p><strong>IL2RA (CD25) Expression:</strong> {row['IL2RA']:.2f}</p>"
            if 'TGFB1' in row:
                html += f"<p><strong>TGFB1 Expression:</strong> {row['TGFB1']:.2f}</p>"
            if 'IL10' in row:
                html += f"<p><strong>IL10 Expression:</strong> {row['IL10']:.2f}</p>"

    html += "</body></html>"

    # 保存HTML报告
    with open(output_file, 'w') as f:
        f.write(html)


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