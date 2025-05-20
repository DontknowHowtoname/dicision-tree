#!/usr/bin/env python
# filepath: d:\Files\OneDrive - email.sxu.edu.cn\School\DataMining\Midterm\codes\filter_negative_values.py
"""
此脚本用于过滤数据集中的负值行。
会删除除了PAY_0至PAY_6列以外的其他列中存在负值的行。
"""
import pandas as pd
import os
import sys

def filter_negative_values(input_file, output_file, is_test=False):
    """
    处理数据集中的负值：
    - 先将BILL_AMT和PAY_AMT列中的负值替换为0
    - 然后对于训练集，过滤除PAY_0至PAY_6列以外的其他列中存在负值的行
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        is_test: 是否为测试数据集，如果是，只替换负值而不删除任何行
    
    返回:
        处理后的数据框
    """
    try:
        print(f"正在读取文件: {input_file}")
        if not os.path.exists(input_file):
            print(f"错误: 文件 {input_file} 不存在!")
            return None
            
        df = pd.read_csv(input_file)
        
        # 保存原始行数用于比较
        original_rows = len(df)
        print(f"原始数据行数: {original_rows}")
        
        # 显示部分数据信息
        print("\n数据前5行:")
        print(df.head())
        print("\n数据列名:")
        print(df.columns.tolist())
        
        # 找出pay_0到pay_6的列名
        pay_columns = [col for col in df.columns if col.startswith('PAY_')]
        
        # 确认我们有pay_columns
        print(f"\n支付状态列: {pay_columns}")
        
        # 获取除了pay列之外的所有列
        other_columns = [col for col in df.columns if col not in pay_columns]
        print(f"其他列: {other_columns}")
        print(f"其他列数量: {len(other_columns)}")
          # 检查每列的负值情况
        print("\n各列负值统计:")
        for col in df.columns:
            if df[col].dtype.kind in 'bifc':  # 检查列是否为数值型
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    print(f"列 {col}: {neg_count} 行包含负值")
        
        # 找出所有的BILL_AMT和PAY_AMT列
        bill_amt_cols = [col for col in df.columns if col.startswith('BILL_AMT')]
        pay_amt_cols = [col for col in df.columns if col.startswith('PAY_AMT')]
        
        print("\n首先替换BILL_AMT和PAY_AMT列中的负值为0:")
        # 替换BILL_AMT列的负值为0
        replaced_count = 0
        for col in bill_amt_cols:
            if df[col].dtype.kind in 'bifc':  # 确认是数值类型
                neg_mask = df[col] < 0
                neg_count = neg_mask.sum()
                if neg_count > 0:
                    print(f"  替换列 {col} 中的 {neg_count} 个负值为0")
                    df.loc[neg_mask, col] = 0
                    replaced_count += neg_count
        
        # 替换PAY_AMT列的负值为0
        for col in pay_amt_cols:
            if df[col].dtype.kind in 'bifc':  # 确认是数值类型
                neg_mask = df[col] < 0
                neg_count = neg_mask.sum()
                if neg_count > 0:
                    print(f"  替换列 {col} 中的 {neg_count} 个负值为0")
                    df.loc[neg_mask, col] = 0
                    replaced_count += neg_count
        
        print(f"总共替换了 {replaced_count} 个负值为0")
        
        # 如果是测试数据集，只替换负值不删除行
        if is_test:
            # 保存替换后的数据
            df.to_csv(output_file, index=False)
            print(f"处理后的数据已保存至: {output_file}")
            return df
            
        # 对于训练集，继续找出除PAY_列以外其他列中存在负值的行
        negative_rows = set()
        # 排除BILL_AMT和PAY_AMT列，因为它们的负值已被替换为0
        other_columns = [col for col in other_columns if not (col.startswith('BILL_AMT') or col.startswith('PAY_AMT'))]
        
        print("\n继续检查其他列中的负值行:")
        for col in other_columns:
            if df[col].dtype.kind in 'bifc':  # 检查列是否为数值型
                # 过滤掉非数值行
                try:
                    neg_mask = df[col] < 0
                    negative_indices = df[neg_mask].index.tolist()
                    if negative_indices:
                        print(f"列 {col} 中包含 {len(negative_indices)} 个负值")
                        negative_rows.update(negative_indices)
                except Exception as e:
                    print(f"处理列 {col} 时出错: {e}")

        # 如果不是测试数据集，执行训练数据的逻辑
        else:
            # 转换为列表（训练集处理逻辑 - 删除行）
            negative_rows = list(negative_rows)
            print(f"\n包含负值的行数: {len(negative_rows)}")
            
            # 如果找到了负值行，则删除
            if negative_rows:
                # 显示部分被删除的行
                sample_size = min(5, len(negative_rows))
                sample_rows = [negative_rows[i] for i in range(sample_size)]
                print(f"\n被删除行的样本 (前{sample_size}行):")
                for idx in sample_rows:
                    print(f"行 {idx}:")
                    print(df.loc[idx])
                
                # 删除这些行
                df_filtered = df.drop(index=negative_rows)
                print(f"\n过滤后行数: {len(df_filtered)}")
                print(f"删除的行数: {original_rows - len(df_filtered)}")
                
                # 保存过滤后的数据
                df_filtered.to_csv(output_file, index=False)
                print(f"过滤后的数据已保存至: {output_file}")
                
                return df_filtered
            
        print("没有找到包含负值的行！")
        # 仍然保存原始数据
        df.to_csv(output_file, index=False)
        print(f"原始数据已保存至: {output_file}")
        return df
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 检查并创建输出目录
    output_dir = "filtered_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n===== 处理训练数据：删除负值行 =====")
    # 处理训练数据
    train_input = "data/train.csv"
    train_output = os.path.join(output_dir, "train_filtered.csv")
    train_filtered = filter_negative_values(train_input, train_output, is_test=False)
    
    # 处理训练标签数据 - 需要与过滤后的训练数据保持一致
    train_label_input = "data/train_label.csv"
    if os.path.exists(train_label_input):
        train_label_df = pd.read_csv(train_label_input)
        # 假设ID列在训练数据和标签数据中都存在
        if 'ID' in train_filtered.columns and 'ID' in train_label_df.columns:
            filtered_ids = train_filtered['ID'].values
            filtered_labels = train_label_df[train_label_df['ID'].isin(filtered_ids)]
            train_label_output = os.path.join(output_dir, "train_label_filtered.csv")
            filtered_labels.to_csv(train_label_output, index=False)
            print(f"过滤后的标签数据已保存至: {train_label_output}")
    
    print("\n===== 处理测试数据：替换BILL_AMT和PAY_AMT中的负值为0 =====")
    # 处理测试数据
    test_input = "data/test.csv"
    test_output = os.path.join(output_dir, "test_processed.csv")  # 更改文件名，表示使用了不同的处理方法
    if os.path.exists(test_input):
        test_filtered = filter_negative_values(test_input, test_output, is_test=True)