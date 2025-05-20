#!/usr/bin/env python
# filepath: /workspaces/dicision_tree/analyze_credit_data.py
"""
此脚本用于分析信用卡数据集中的异常情况，特别是借款超过额度的情况，
并生成详细的分析报告和可视化图表。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_import import load_data, analyze_data_anomalies

def analyze_credit_limit_anomalies():
    """分析信用额度异常情况并生成报告"""
    # 创建输出目录
    output_dir = 'analysis_reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据（不使用预处理的数据）
    print("加载原始数据...")
    train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=False)
    
    # 分析训练数据
    print("\n分析训练数据中的信用额度异常情况...")
    analyze_dataset(train_data, train_labels, 'train', output_dir)
    
    # 分析测试数据
    print("\n分析测试数据中的信用额度异常情况...")
    analyze_dataset(test_data, None, 'test', output_dir)
    
    print(f"\n分析完成，报告已保存至 {output_dir} 目录")

def analyze_dataset(data, labels=None, dataset_name='dataset', output_dir='analysis_reports'):
    """分析单个数据集的信用额度异常情况"""
    if 'LIMIT_BAL' not in data.columns:
        print(f"错误: 数据集中没有找到LIMIT_BAL列")
        return
    
    # 找出所有账单金额列
    bill_cols = [col for col in data.columns if 'BILL_AMT' in col]
    if not bill_cols:
        print(f"错误: 数据集中没有找到账单金额列")
        return
    
    # 创建分析报告
    report = {
        'dataset_size': len(data),
        'bill_columns': bill_cols,
        'anomaly_counts': {}
    }
    
    # 检查每个账单列的异常情况
    all_over_limit_mask = pd.Series(False, index=data.index)
    for bill_col in bill_cols:
        over_limit_mask = data[bill_col] > data['LIMIT_BAL']
        over_limit_count = over_limit_mask.sum()
        
        # 更新所有超限记录的掩码
        all_over_limit_mask = all_over_limit_mask | over_limit_mask
        
        if over_limit_count > 0:
            # 计算超额比例
            over_limit_ratio = data.loc[over_limit_mask, bill_col] / data.loc[over_limit_mask, 'LIMIT_BAL']
            
            report['anomaly_counts'][bill_col] = {
                'count': int(over_limit_count),
                'percentage': float(over_limit_count / len(data) * 100),
                'min_ratio': float(over_limit_ratio.min()),
                'max_ratio': float(over_limit_ratio.max()),
                'mean_ratio': float(over_limit_ratio.mean()),
                'median_ratio': float(over_limit_ratio.median())
            }
            
            print(f"  {bill_col}: 发现 {over_limit_count} 条记录 ({over_limit_count / len(data) * 100:.2f}%) 借款超过额度")
            print(f"    超额比例: 最小 {over_limit_ratio.min():.2f}倍, 最大 {over_limit_ratio.max():.2f}倍, 平均 {over_limit_ratio.mean():.2f}倍")
    
    # 输出总体结果
    total_over_limit = all_over_limit_mask.sum()
    report['total_over_limit'] = {
        'count': int(total_over_limit),
        'percentage': float(total_over_limit / len(data) * 100)
    }
    
    print(f"\n总计: {total_over_limit} 条记录 ({total_over_limit / len(data) * 100:.2f}%) 至少在一个月中借款超过额度")
    
    # 如果有标签，分析超额借款对违约的影响
    if labels is not None and not labels.empty:
        over_limit_default_rate = labels[all_over_limit_mask].mean()
        normal_default_rate = labels[~all_over_limit_mask].mean()
        
        report['default_analysis'] = {
            'over_limit_default_rate': float(over_limit_default_rate),
            'normal_default_rate': float(normal_default_rate),
            'relative_risk': float(over_limit_default_rate / normal_default_rate) if normal_default_rate > 0 else None
        }
        
        print(f"\n违约率分析:")
        print(f"  超额借款客户的违约率: {over_limit_default_rate:.2%}")
        print(f"  正常借款客户的违约率: {normal_default_rate:.2%}")
        if normal_default_rate > 0:
            print(f"  相对风险比: {over_limit_default_rate / normal_default_rate:.2f}倍")
    
    # 生成可视化
    generate_visualizations(data, bill_cols, dataset_name, output_dir)
    
    # 保存报告
    import json
    report_path = os.path.join(output_dir, f'{dataset_name}_credit_limit_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def generate_visualizations(data, bill_cols, dataset_name, output_dir):
    """生成可视化图表"""
    # 1. 账单金额与信用额度的散点图
    plt.figure(figsize=(15, 10))
    
    for i, bill_col in enumerate(bill_cols):
        plt.subplot(2, 3, i+1)
        plt.scatter(data['LIMIT_BAL'], data[bill_col], alpha=0.5)
        plt.plot([0, data['LIMIT_BAL'].max()], [0, data['LIMIT_BAL'].max()], 'r--')
        plt.xlabel('信用额度')
        plt.ylabel(bill_col)
        plt.title(f'{bill_col} vs 信用额度')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_bill_vs_limit_scatter.png'))
    plt.close()
    
    # 2. 账单金额与信用额度比率的分布图
    plt.figure(figsize=(15, 10))
    
    for i, bill_col in enumerate(bill_cols):
        plt.subplot(2, 3, i+1)
        ratio = data[bill_col] / data['LIMIT_BAL'].replace(0, np.nan)
        ratio = ratio[~ratio.isna()]  # 移除除以零的结果
        
        sns.histplot(ratio[ratio <= 3], bins=50)  # 限制在3倍以内，以便查看主要分布
        plt.axvline(x=1, color='r', linestyle='--')
        plt.xlabel(f'{bill_col} / 信用额度')
        plt.ylabel('频数')
        plt.title(f'{bill_col} 与信用额度的比率分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_bill_to_limit_ratio_hist.png'))
    plt.close()
    
    # 3. 各月份超额借款比例对比图
    over_limit_percents = []
    for bill_col in bill_cols:
        over_limit_count = (data[bill_col] > data['LIMIT_BAL']).sum()
        over_limit_percent = over_limit_count / len(data) * 100
        over_limit_percents.append(over_limit_percent)
    
    plt.figure(figsize=(10, 6))
    plt.bar(bill_cols, over_limit_percents)
    plt.xlabel('账单月份')
    plt.ylabel('超额借款比例 (%)')
    plt.title('不同月份超额借款比例对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_over_limit_percent_by_month.png'))
    plt.close()

if __name__ == "__main__":
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    analyze_credit_limit_anomalies()
