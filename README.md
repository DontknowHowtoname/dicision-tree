# dicision_tree

## 项目说明

这是一个基于决策树的分类项目，包含数据预处理、特征工程和模型训练等流程。

## 文件结构

- `data_import.py`: 包含数据加载和预处理功能
- `prepare_data.py`: 预处理数据并保存为CSV文件
- `train_decision_tree.py`: 训练决策树模型并评估性能
- `feature_importances.csv`: 特征重要性排序结果
- `decision_tree_structure.txt`: 决策树结构可视化
- `data/`: 存放原始数据的文件夹
- `preprocessed_data/`: 存放预处理后数据的文件夹（运行后自动生成）

## 使用流程

1. **预处理数据**：
   ```
   python prepare_data.py
   ```
   这将加载原始数据，进行预处理，并将结果保存到 `preprocessed_data/` 文件夹中。

2. **训练决策树模型**：
   ```
   python train_decision_tree.py
   ```
   这将自动使用预处理后的数据训练决策树模型，并生成评估结果。

## 数据预处理流程

- 处理缺失值：数值型特征使用中位数填充，分类特征使用众数填充
- 处理异常借款：检测并处理借款金额超过信用额度的异常情况
- 处理异常值：使用IQR方法检测并处理异常值
- 特征工程：创建额外的有意义特征
- 特征缩放：使用RobustScaler对数值型特征进行标准化

预处理后的数据会保存为CSV文件，避免每次训练时重复预处理，提高效率。同时会生成异常数据分析报告，存储在`preprocessed_data`文件夹中。

## 预处理数据列标签说明

### 原始特征

- `LIMIT_BAL`: 信用额度
- `SEX`: 性别 (1=男, 2=女)
- `EDUCATION`: 教育程度 (1=研究生, 2=大学, 3=高中, 4=其他)
- `MARRIAGE`: 婚姻状况 (1=已婚, 2=单身, 3=其他)
- `AGE`: 年龄
- `PAY_0` 到 `PAY_6`: 过去6个月的还款状态 (数值越大表示延迟天数越多)
- `BILL_AMT1` 到 `BILL_AMT6`: 过去6个月的账单金额
- `PAY_AMT1` 到 `PAY_AMT6`: 过去6个月的还款金额

### 信用额度利用率特征

1. **月度超额特征**:
   - `OVER_LIMIT_RATIO_i`: 第i个月的账单金额与信用额度比率
     ```
     OVER_LIMIT_RATIO_i = BILL_AMTi / LIMIT_BAL
     ```
   - `IS_OVER_LIMIT_i`: 第i个月是否超过信用额度 (1=是, 0=否)
     ```
     IS_OVER_LIMIT_i = (BILL_AMTi > LIMIT_BAL) ? 1 : 0
     ```
   - `OVER_LIMIT_MONTHS_COUNT`: 总共有几个月超过信用额度
     ```
     OVER_LIMIT_MONTHS_COUNT = SUM(IS_OVER_LIMIT_i) for i=1 to 6
     ```

2. **信用额度利用率特征**:
   - `UTIL_RATE_i`: 第i个月的信用卡利用率
     ```
     UTIL_RATE_i = BILL_AMTi / LIMIT_BAL
     ```
   - `MAX_UTIL_RATE`: 最大月度利用率
     ```
     MAX_UTIL_RATE = MAX(UTIL_RATE_i) for i=1 to 6
     ```
   - `AVG_UTIL_RATE`: 平均月度利用率
     ```
     AVG_UTIL_RATE = MEAN(UTIL_RATE_i) for i=1 to 6
     ```

### 支付比率特征

- `PAYMENT_RATIO_i`: 第i个月的还款比例
  ```
  PAYMENT_RATIO_i = PAY_AMTi / BILL_AMTi
  ```
- `TOTAL_BILL`: 6个月内总账单金额
  ```
  TOTAL_BILL = SUM(BILL_AMTi) for i=1 to 6
  ```
- `TOTAL_PAYMENT`: 6个月内总还款金额
  ```
  TOTAL_PAYMENT = SUM(PAY_AMTi) for i=1 to 6
  ```
- `TOTAL_PAYMENT_RATIO`: 总还款比例
  ```
  TOTAL_PAYMENT_RATIO = TOTAL_PAYMENT / TOTAL_BILL
  ```
- `DELAY_PAYMENT_COUNT`: 延迟付款次数
  ```
  DELAY_PAYMENT_COUNT = SUM(PAY_i > 0) for i=0 to 6
  ```

### 累计账单和累计还款特征

1. **累计账单和还款**:
   - `CUM_BILL_AMT1`: 第1个月的累计账单 = BILL_AMT1
   - `CUM_PAYMENT1`: 第1个月的累计还款 = PAY_AMT1
   - 对于i=2到6:
     ```
     CUM_BILL_AMTi = MAX(0, CUM_BILL_AMT(i-1) - PAY_AMT(i-1) + BILL_AMTi)
     CUM_PAYMENTi = CUM_PAYMENT(i-1) + PAY_AMTi
     ```

2. **累计还款比率**:
   - `CUM_PAYMENT_RATIO_i`: 第i个月的累计还款比率 (i=2到6)
     ```
     CUM_PAYMENT_RATIO_i = CUM_PAYMENTi / CUM_BILL_AMTi
     ```

3. **累计超额特征**:
   - `CUM_OVER_LIMIT_RATIO_i`: 第i个月的累计账单与信用额度比率 (i=2到6)
     ```
     CUM_OVER_LIMIT_RATIO_i = CUM_BILL_AMTi / LIMIT_BAL
     ```
   - `CUM_IS_OVER_LIMIT_i`: 第i个月的累计账单是否超过信用额度 (i=2到6)
     ```
     CUM_IS_OVER_LIMIT_i = (CUM_BILL_AMTi > LIMIT_BAL) ? 1 : 0
     ```
   - `CUM_OVER_LIMIT_MONTHS`: 累计账单超过信用额度的月数
     ```
     CUM_OVER_LIMIT_MONTHS = SUM(CUM_IS_OVER_LIMIT_i) for i=2 to 6
     ```

4. **最终累计特征**:
   - `FINAL_CUM_BILL`: 最终累计账单金额 (CUM_BILL_AMT6)
   - `FINAL_CUM_PAYMENT`: 最终累计还款金额 (CUM_PAYMENT6)
   - `FINAL_CUM_PAYMENT_RATIO`: 最终累计还款比率
     ```
     FINAL_CUM_PAYMENT_RATIO = FINAL_CUM_PAYMENT / FINAL_CUM_BILL
     ```
   - `FINAL_CUM_OVER_LIMIT_RATIO`: 最终累计账单与信用额度比率
     ```
     FINAL_CUM_OVER_LIMIT_RATIO = FINAL_CUM_BILL / LIMIT_BAL
     ```

## 数据分析工具

项目提供了专门的数据分析工具：

- `analyze_credit_data.py`：分析信用额度数据异常情况，生成详细报告和可视化图表
  ```
  python analyze_credit_data.py
  ```
  执行后会在`analysis_reports`文件夹中生成分析报告和图表。

## 特征重要性分析

基于决策树和随机森林的特征重要性评估结果显示，对信用风险预测最为重要的特征包括：

1. **顶级预测特征**:
   - `DELAY_PAYMENT_COUNT`（延迟付款次数）：占决策树分裂重要性的61.98%
   - `PAY_0`（最近一个月的还款状态）：占决策树分裂重要性的22.89%
   - `CUM_PAYMENT4`（第4个月的累计还款总额）：占决策树分裂重要性的3.86%

2. **累计类特征重要性**:
   模型显示，累计账单和累计还款特征的引入显著提升了预测能力，尤其是：
   - 累计还款（CUM_PAYMENT）系列特征
   - 累计还款比率（CUM_PAYMENT_RATIO）系列特征
   - 累计超额比率（CUM_OVER_LIMIT_RATIO）系列特征

3. **信用利用率特征重要性**:
   - `AVG_UTIL_RATE`（平均利用率）：在决策树中的重要性为1.83%
   - 与单月超额相比，累计超额对预测信用风险的贡献更大

完整的特征重要性排序详见 `feature_importances.csv` 和 `all_feature_importances.csv` 文件。

## 模型性能结果

在添加累计账单和累计还款特征后，模型性能有显著提升：

- 训练准确率: 74.80%
- 验证准确率: 74.69%
- 5折交叉验证准确率: 74.43% ± 0.85%

分类报告:
- 0类（无违约风险）:
  - 精确率: 76%
  - 召回率: 95%
  - F1分数: 84%
- 1类（有违约风险）:
  - 精确率: 63%
  - 召回率: 24%
  - F1分数: 35%

**模型参数**:
- 算法: 决策树
- 最优参数：
  - 标准: 熵 (entropy)
  - 最大深度: 4
  - 最小样本分裂: 2
  - 最小叶节点样本: 8
  - 最大特征数: None (使用全部特征)
  - 类别权重: None (类别平衡)

通过对数据进行更细致的特征工程，特别是添加累计账单和还款特征，模型对信用风险的识别能力得到了明显提升。