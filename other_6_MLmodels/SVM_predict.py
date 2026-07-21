import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# 加载模型和特征缩放器
model_filename = "./last/svm_model_GridSearchCV_best_20240718.pkl"
model = joblib.load(model_filename)
scaler_filename = "./last/svm_scaler_GridSearchCV_best_20240718.pkl"
scaler = joblib.load(scaler_filename)

# 读取新的CSV文件
new_data_file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_test_20240717.csv"
new_data = pd.read_csv(new_data_file_path)

# 确保新数据集中包含真实标签列
if 'Label' not in new_data.columns:
    raise ValueError("The new data must contain a 'Label' column with the true labels.")

# 使用RDKit计算新数据的Morgan指纹作为特征
new_data['Morgan_Features'] = new_data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(new_data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 应用特征缩放器到新数据
new_features_scaled = scaler.transform(new_features)

# 使用模型进行预测
predictions = model.predict(new_features_scaled)

# 获取真实标签
true_labels = new_data['Label']

# 计算准确率和F1分数
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')  # 使用加权平均F1分数

# 计算MCC指数
mcc = matthews_corrcoef(true_labels, predictions)

# 打印性能指标
print(f"Accuracy: {accuracy}")
print(f"Weighted F1 Score: {f1}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# 如果你想保存预测结果到CSV文件
output_csv_path = "./last/SVM_pred_0718.csv"
predictions_df = pd.DataFrame({'SMILES': new_data['SMILES'], 'Label': predictions})
predictions_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")