from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from data_import import load_data, preprocess_data, save_preprocessed_data
import pandas as pd
import numpy as np
import os
import time

def train_and_predict():
    print("开始训练与预测过程...")
    start_time = time.time()
    
    # 加载数据 (直接使用预处理后的数据，如果存在)
    train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=True)
    
    # 验证样本和标签数量是否匹配
    if len(train_data) != len(train_labels):
        raise ValueError(f"Number of samples in train.csv ({len(train_data)}) does not match number of labels in train_label.csv ({len(train_labels)})")
        
    # 如果预处理后的数据不存在，则进行预处理并保存
    import os
    preprocessed_dir = 'preprocessed_data'
    train_preprocessed_path = os.path.join(preprocessed_dir, 'train_preprocessed.csv')
    
    if not os.path.exists(train_preprocessed_path):
        print("预处理后的数据不存在，现在进行数据预处理...")
        from data_import import save_preprocessed_data
        # 重新加载原始数据
        train_data, train_labels, test_data, train_ids, test_ids = load_data(use_preprocessed=False)
        # 预处理并保存
        train_data, test_data, _, _ = save_preprocessed_data(train_data, test_data, train_ids, test_ids)
    else:
        print("使用已预处理的数据...")

    # 将训练数据分割为训练集和验证集，用于评估模型性能
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    
    # 特征选择 - 使用随机森林评估特征重要性
    print("进行特征选择评估...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train, y_train)
    
    # 计算特征重要性
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("随机森林评估的前15个重要特征:")
    print(feature_importances.head(15))
    
    # 根据重要性选择特征
    selector = SelectFromModel(rf_selector, threshold="mean")
    selector.fit(X_train, y_train)
    
    # 获取所选特征
    selected_features = X_train.columns[selector.get_support()]
    print(f"选择了 {len(selected_features)} 个特征: {list(selected_features)}")
    
    # 使用所选特征
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    
    print(f"特征选择后: 训练集 {X_train_selected.shape}, 验证集 {X_val_selected.shape}")

    # 使用交叉验证和网格搜索找到最佳参数
    print("正在使用网格搜索找到最佳模型参数...")
    param_grid = {
        'max_depth': [1, 2, 3, 4],  # 将深度限制到最大4层
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # 使用特征选择后的数据进行网格搜索
    print("使用选择的特征进行网格搜索...")
    grid_search.fit(X_train_selected, y_train)
    
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")
    
    # 使用最佳参数创建决策树分类器，确保深度不超过4
    best_params['max_depth'] = min(best_params.get('max_depth', 4), 4)  # 确保最大深度为4
    best_clf = DecisionTreeClassifier(**best_params, random_state=42)
    
    print(f"最终使用的参数 (限制深度为4): {best_params}")
    
    # 在全部训练数据上训练最终模型
    print("正在训练最终模型...")
    
    # 对全部数据应用特征选择
    X_train_all_selected = selector.transform(train_data)
    best_clf.fit(X_train_all_selected, train_labels)
    
    # 保存选择的特征列表
    pd.DataFrame({'selected_features': list(selected_features)}).to_csv('selected_features.csv', index=False)
    
    # 输出训练准确率
    X_train_all_selected = selector.transform(train_data)
    train_predictions = best_clf.predict(X_train_all_selected)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"训练准确率: {train_accuracy:.4f}")
    
    # 在验证集上评估模型
    val_predictions = best_clf.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"验证集准确率: {val_accuracy:.4f}")
    
    # 计算交叉验证分数
    print("计算交叉验证分数...")
    # 对所有训练数据应用特征选择后进行交叉验证
    X_selected = selector.transform(train_data)
    cv_scores = cross_val_score(best_clf, X_selected, train_labels, cv=5)
    print(f"5折交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_val, val_predictions))
    
    # 导出决策树结构到文本文件
    tree_rules = export_text(best_clf, feature_names=list(selected_features))
    
    # 创建决策树文本文件
    tree_file_path = 'decision_tree_structure.txt'
    with open(tree_file_path, 'w', encoding='utf-8') as f:
        f.write(tree_rules)
    print(f"决策树结构已保存到文件: {os.path.abspath(tree_file_path)}")
    
    # 评估特征重要性
    feature_importances = pd.DataFrame({
        'feature': selected_features,
        'importance': best_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n所选特征的重要性:")
    print(feature_importances)
    
    # 保存特征重要性到文件
    feature_importances.to_csv('feature_importances.csv', index=False)
    
    # 保存完整的随机森林特征重要性
    all_feature_importances = pd.DataFrame({
        'feature': train_data.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    all_feature_importances.to_csv('all_feature_importances.csv', index=False)
    print("特征重要性已保存到 feature_importances.csv")

    # 预测测试数据
    print("正在预测测试数据...")
    # 对测试数据应用同样的特征选择
    test_data_selected = selector.transform(test_data)
    test_predictions = best_clf.predict(test_data_selected)

    # 保存预测结果，包含ID列（如果有）
    if test_ids is not None:
        result_df = pd.DataFrame({'ID': test_ids, 'label': test_predictions})
    else:
        result_df = pd.DataFrame({'label': test_predictions})
    
    result_df.to_csv('data/test_label.csv', index=False)
    print(f"已保存{len(test_predictions)}条预测结果到 data/test_label.csv")
    
    # 输出总运行时间
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    train_and_predict()