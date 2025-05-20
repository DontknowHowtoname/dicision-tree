import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import os

def load_data(use_preprocessed=False):
    """
    加载数据，可选择直接加载预处理后的数据
    
    参数:
        use_preprocessed: 是否使用预处理后的数据文件（如果存在）
    
    返回:
        train_data, train_labels, test_data, train_ids, test_ids
    """
    # 检查预处理后的文件是否存在
    preprocessed_dir = 'preprocessed_data'
    train_preprocessed_path = os.path.join(preprocessed_dir, 'train_preprocessed.csv')
    test_preprocessed_path = os.path.join(preprocessed_dir, 'test_preprocessed.csv')
    
    if use_preprocessed and os.path.exists(train_preprocessed_path) and os.path.exists(test_preprocessed_path):
        print("正在加载预处理后的数据...")
        train_data = pd.read_csv(train_preprocessed_path)
        test_data = pd.read_csv(test_preprocessed_path)
        train_labels_df = pd.read_csv('filtered_data/train_label_filtered.csv')

        # 处理标签
        train_labels = train_labels_df['Label']
        
        # 加载ID（如果已保存）
        train_ids_path = os.path.join(preprocessed_dir, 'train_ids.csv')
        test_ids_path = os.path.join(preprocessed_dir, 'test_ids.csv')
        
        train_ids = pd.read_csv(train_ids_path)['ID'] if os.path.exists(train_ids_path) else None
        test_ids = pd.read_csv(test_ids_path)['ID'] if os.path.exists(test_ids_path) else None
        
        print(f"预处理数据加载完成。训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}")
        
        return train_data, train_labels, test_data, train_ids, test_ids
    
    # 加载原始训练数据
    train_data = pd.read_csv('filtered_data/train_filtered.csv')
    train_labels_df = pd.read_csv('filtered_data/train_label_filtered.csv')
    
    # 存储ID列以供将来参考，但从训练数据中删除
    train_ids = None
    if 'ID' in train_data.columns:
        train_ids = train_data['ID'].copy()
        # 从训练数据中删除ID列，因为它不应该参与训练
        train_data = train_data.drop('ID', axis=1)
    
    # 确保train_labels_df只包含有效的行（与train_data匹配的行）
    if 'ID' in train_labels_df.columns:
        # 获取有效的ID
        valid_ids = train_ids.values if train_ids is not None else range(len(train_data))
        
        # 只保留这些ID对应的标签
        train_labels = train_labels_df[train_labels_df['ID'].isin(valid_ids)]
        # 确保标签顺序与训练数据一致
        train_labels = train_labels.set_index('ID').reindex(index=valid_ids).reset_index()
        # 只取Label列作为标签
        train_labels = train_labels['Label']
    else:
        # 如果没有ID列，则假设前N行是有效的
        train_labels = train_labels_df.iloc[:len(train_data)]['Label']
    
    print(f"训练数据样本数: {len(train_data)}, 标签数: {len(train_labels)}")
    print(f"训练数据特征数: {train_data.shape[1]}")
    
    # 处理测试数据
    test_data = pd.read_csv('filtered_data/test_processed.csv')
    test_ids = None
    if 'ID' in test_data.columns:
        test_ids = test_data['ID'].copy()
        # 从测试数据中删除ID列
        test_data = test_data.drop('ID', axis=1)
    
    return train_data, train_labels, test_data, train_ids, test_ids

# 数据预处理功能实现
def preprocess_data(data, is_train=True, scaler=None, imputer=None, save_path=None):
    """
    对数据进行预处理，包括缺失值处理、特征标准化和异常值处理
    
    参数:
        data: 待处理的数据
        is_train: 是否为训练数据
        scaler: 用于特征标准化的转换器，测试集使用与训练集相同的转换
        imputer: 用于填充缺失值的转换器，测试集使用与训练集相同的转换
        save_path: 如果提供，则将预处理后的数据保存到该路径
    
    返回:
        处理后的数据，以及用于转换的scaler和imputer（如果是训练集）
    """
    df = data.copy()
    
    # 删除完全为空的列
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        print(f"删除完全为空的列: {empty_cols}")
        df = df.drop(columns=empty_cols)
    
    # 1. 检查并处理缺失值
    print(f"处理前缺失值情况:\n{df.isnull().sum()}")
    
    if is_train:
        # 为训练集创建新的imputer
        imputer = SimpleImputer(strategy='median')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # 对分类特征，使用众数填充
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if not cat_cols.empty:
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    else:
        # 使用训练集的imputer转换测试集
        if imputer is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = imputer.transform(df[numeric_cols])
            
            # 对分类特征，使用众数填充
            cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            if not cat_cols.empty:
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    print(f"处理后缺失值情况:\n{df.isnull().sum()}")
    
    # 1.5 处理借款金额超过额度的异常情况并添加超额倍数特征
    # if 'LIMIT_BAL' in df.columns:
    #     # 找出所有包含账单金额的列
    #     bill_cols = [col for col in df.columns if 'BILL_AMT' in col]
        
    #     # 处理每个月的账单数据和超额情况
    #     for i, bill_col in enumerate(bill_cols, 1):
    #         # 计算超额倍数 (账单金额/信用额度)
    #         df[f'OVER_LIMIT_RATIO_{i}'] = df[bill_col] / df['LIMIT_BAL'].replace(0, 1)
            
    #         # 标记是否超额
    #         df[f'IS_OVER_LIMIT_{i}'] = (df[bill_col] > df['LIMIT_BAL']).astype(int)
            
    #         # 处理超额账单
    #         over_limit_mask = df[bill_col] > df['LIMIT_BAL']
    #         # 将超过额度的账单金额限制为额度值
    #         df.loc[over_limit_mask, bill_col] = df.loc[over_limit_mask, 'LIMIT_BAL']
        
    #     # 添加月度账单利用率特征（每月账单占信用额度的百分比）
    #     for i, bill_col in enumerate(bill_cols, 1):
    #         df[f'UTIL_RATE_{i}'] = df[bill_col] / df['LIMIT_BAL'].replace(0, 1)
        
    #     # 添加最大和平均月度利用率
    #     util_rate_cols = [f'UTIL_RATE_{i}' for i in range(1, len(bill_cols)+1)]
    #     df['MAX_UTIL_RATE'] = pd.concat([df[col] for col in util_rate_cols], axis=1).max(axis=1)
    #     df['AVG_UTIL_RATE'] = pd.concat([df[col] for col in util_rate_cols], axis=1).mean(axis=1)
        
    #     # 添加超额月份计数特征
    #     is_over_limit_cols = [f'IS_OVER_LIMIT_{i}' for i in range(1, len(bill_cols)+1)]
    #     df['OVER_LIMIT_MONTHS_COUNT'] = df[is_over_limit_cols].sum(axis=1)
    
    # 2. 处理异常值 - 使用分位数法检测并替换异常值
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值替换为边界值
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    # 3. 特征工程 - 添加新特征
    # 添加账单与付款比率特征
    for i in range(1, 7):
        bill_col = f'BILL_AMT{i}'
        pay_col = f'PAY_AMT{i}'
        if bill_col in df.columns and pay_col in df.columns:
            # 确保不除以零
            df[f'PAYMENT_RATIO_{i}'] = df[pay_col] / df[bill_col].replace(0, 1)
    
    # 添加累计账单和累计还款特征
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]
    
    # if bill_cols and pay_cols:
    #     # 初始化累计账单和累计还款列
    #     df['CUM_BILL_AMT1'] = df['BILL_AMT1']  # 第一个月的累计账单等于当月账单
    #     df['CUM_PAYMENT1'] = df['PAY_AMT1']    # 第一个月的累计还款等于当月还款
        
    #     # 计算累计账单和累计还款 (从第2个月开始)
    #     for i in range(2, 7):
    #         if f'BILL_AMT{i}' in df.columns and f'PAY_AMT{i-1}' in df.columns and f'CUM_BILL_AMT{i-1}' in df.columns:
    #             # 本月累计账单 = 上月累计账单 - 上月还款 + 本月账单
    #             # 注意：如果计算结果为负，则置为0（表示全部还清）
    #             df[f'CUM_BILL_AMT{i}'] = np.maximum(0, df[f'CUM_BILL_AMT{i-1}'] - df[f'PAY_AMT{i-1}'] + df[f'BILL_AMT{i}'])
                
    #             # 本月累计还款 = 上月累计还款 + 本月还款
    #             if f'PAY_AMT{i}' in df.columns and f'CUM_PAYMENT{i-1}' in df.columns:
    #                 df[f'CUM_PAYMENT{i}'] = df[f'CUM_PAYMENT{i-1}'] + df[f'PAY_AMT{i}']
                    
    #                 # 计算累计还款与累计账单比率
    #                 df[f'CUM_PAYMENT_RATIO_{i}'] = df[f'CUM_PAYMENT{i}'] / df[f'CUM_BILL_AMT{i}'].replace(0, 1)
                    
    #                 # 计算累计账单与信用额度的比率（累计超额倍数）
    #                 if 'LIMIT_BAL' in df.columns:
    #                     df[f'CUM_OVER_LIMIT_RATIO_{i}'] = df[f'CUM_BILL_AMT{i}'] / df['LIMIT_BAL'].replace(0, 1)
    #                     df[f'CUM_IS_OVER_LIMIT_{i}'] = (df[f'CUM_BILL_AMT{i}'] > df['LIMIT_BAL']).astype(int)
        
    #     # 添加最终月的累计特征
    #     df['FINAL_CUM_BILL'] = df['CUM_BILL_AMT6'] if 'CUM_BILL_AMT6' in df.columns else df['CUM_BILL_AMT5']
    #     df['FINAL_CUM_PAYMENT'] = df['CUM_PAYMENT6'] if 'CUM_PAYMENT6' in df.columns else df['CUM_PAYMENT5']
    #     df['FINAL_CUM_PAYMENT_RATIO'] = df['FINAL_CUM_PAYMENT'] / df['FINAL_CUM_BILL'].replace(0, 1)
        
    #     # 计算最终累计超额比率和累计超额月数
    #     if 'LIMIT_BAL' in df.columns:
    #         df['FINAL_CUM_OVER_LIMIT_RATIO'] = df['FINAL_CUM_BILL'] / df['LIMIT_BAL'].replace(0, 1)
    #         cum_over_limit_cols = [f'CUM_IS_OVER_LIMIT_{i}' for i in range(2, 7) if f'CUM_IS_OVER_LIMIT_{i}' in df.columns]
    #         if cum_over_limit_cols:
    #             df['CUM_OVER_LIMIT_MONTHS'] = df[cum_over_limit_cols].sum(axis=1)
    
    # 添加总账单和总付款特征
    bill_cols = [col for col in df.columns if 'BILL_AMT' in col]
    pay_cols = [col for col in df.columns if 'PAY_AMT' in col]
    
    if bill_cols and pay_cols:
        df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
        df['TOTAL_PAYMENT'] = df[pay_cols].sum(axis=1)
        df['TOTAL_PAYMENT_RATIO'] = df['TOTAL_PAYMENT'] / df['TOTAL_BILL'].replace(0, 1)
    
    # 添加延迟付款计数特征
    pay_status_cols = [col for col in df.columns if col.startswith('PAY_') and not col.startswith('PAY_AMT')]
    if pay_status_cols:
        df['DELAY_PAYMENT_COUNT'] = (df[pay_status_cols] > 0).sum(axis=1)
    
    # 4. 特征标准化 - 使用RobustScaler，对异常值不敏感
    if is_train:
        # 为训练集创建新的scaler
        scaler = RobustScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        # 使用训练集的scaler转换测试集
        if scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    print(f"预处理后特征数: {df.shape[1]}")
    
    # 如果提供了保存路径，则保存预处理后的数据
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"预处理后的数据已保存至: {save_path}")
    
    if is_train:
        return df, scaler, imputer
    else:
        return df

def save_preprocessed_data(train_data, test_data, train_ids=None, test_ids=None):
    """
    预处理数据并保存为CSV文件
    
    参数:
        train_data: 训练数据
        test_data: 测试数据
        train_ids: 训练数据ID
        test_ids: 测试数据ID
    
    返回:
        预处理后的训练数据、测试数据、scaler和imputer
    """
    # 创建保存目录
    preprocessed_dir = 'preprocessed_data'
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # 预处理训练数据并保存
    train_save_path = os.path.join(preprocessed_dir, 'train_preprocessed.csv')
    processed_train, scaler, imputer = preprocess_data(
        train_data, 
        is_train=True, 
        save_path=train_save_path
    )
    
    # 预处理测试数据并保存
    test_save_path = os.path.join(preprocessed_dir, 'test_preprocessed.csv')
    processed_test = preprocess_data(
        test_data, 
        is_train=False, 
        scaler=scaler,
        imputer=imputer,
        save_path=test_save_path
    )
    
    # 保存ID信息（如果有）
    if train_ids is not None:
        train_ids_path = os.path.join(preprocessed_dir, 'train_ids.csv')
        pd.DataFrame({'ID': train_ids}).to_csv(train_ids_path, index=False)
    
    if test_ids is not None:
        test_ids_path = os.path.join(preprocessed_dir, 'test_ids.csv')
        pd.DataFrame({'ID': test_ids}).to_csv(test_ids_path, index=False)
    
    print("所有预处理数据已保存完成。")
    return processed_train, processed_test, scaler, imputer

def analyze_data_anomalies(data):
    """
    分析数据中的异常情况，并返回分析报告
    
    参数:
        data: 待分析的数据
    
    返回:
        异常数据的统计信息
    """
    df = data.copy()
    report = {}
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    report['missing_values'] = missing_values[missing_values > 0].to_dict()
    report['missing_values_total'] = missing_values.sum()
    
    # 检查借款超过额度的情况
    if 'LIMIT_BAL' in df.columns:
        bill_cols = [col for col in df.columns if 'BILL_AMT' in col]
        over_limit_data = {}
        
        for bill_col in bill_cols:
            over_limit_mask = df[bill_col] > df['LIMIT_BAL']
            over_limit_count = over_limit_mask.sum()
            
            if over_limit_count > 0:
                over_limit_data[bill_col] = {
                    'count': int(over_limit_count),
                    'percentage': float(over_limit_count / len(df) * 100),
                    'max_ratio': float(df[over_limit_mask][bill_col].max() / df.loc[over_limit_mask, 'LIMIT_BAL'].min())
                }
        
        report['over_limit'] = over_limit_data
        report['over_limit_total'] = sum(item['count'] for item in over_limit_data.values())
    
    # 检查异常值 - 使用IQR方法
    outliers_data = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_lower = (df[column] < lower_bound).sum()
        outliers_upper = (df[column] > upper_bound).sum()
        
        if outliers_lower > 0 or outliers_upper > 0:
            outliers_data[column] = {
                'lower_outliers': int(outliers_lower),
                'upper_outliers': int(outliers_upper),
                'total_outliers': int(outliers_lower + outliers_upper),
                'percentage': float((outliers_lower + outliers_upper) / len(df) * 100)
            }
    
    report['outliers'] = outliers_data
    report['outliers_total'] = sum(item['total_outliers'] for item in outliers_data.values())
    
    return report

# 示例使用函数
if __name__ == "__main__":
    # 加载原始数据
    train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=False)
    
    # 预处理并保存数据
    processed_train, processed_test, scaler, imputer = save_preprocessed_data(
        train_data, test_data, train_ids, test_ids
    )
    
    print("预处理完成，数据已保存。下次可以直接加载预处理后的数据：")
    print("train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=True)")