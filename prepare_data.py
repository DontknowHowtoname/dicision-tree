#!/usr/bin/env python
# filepath: /workspaces/dicision_tree/prepare_data.py
"""
此脚本用于预处理数据并保存为CSV文件，供后续训练使用。
"""
from data_import import load_data, save_preprocessed_data

if __name__ == "__main__":
    print("开始数据预处理流程...")
    
    # 加载原始数据
    train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=False)
    
    # 预处理并保存数据
    processed_train, processed_test, scaler, imputer = save_preprocessed_data(
        train_data, test_data, train_ids, test_ids
    )
    
    print(f"训练数据形状: {processed_train.shape}")
    print(f"测试数据形状: {processed_test.shape}")
    print("预处理完成，数据已保存。现在可以运行train_decision_tree.py直接使用预处理后的数据进行训练。")
