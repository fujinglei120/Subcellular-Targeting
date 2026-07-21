import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# 加载保存的KNN模型
model_filename = "./last/knn_model_GridSearchCV_best_20240721.pkl"
model = joblib.load(model_filename)

# 读取新的CSV文件
new_data_file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_test_20240717.csv"
new_data = pd.read_csv(new_data_file_path)

# 确保新数据集中包含真实标签列
if 'Label' not in new_data.columns:
    raise ValueError("The new data must contain a 'Label' column with the true labels.")

# 对新数据的SMILES使用RDKit计算Morgan指纹作为特征
new_data['Morgan_Features'] = new_data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(new_data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 使用加载的模型进行预测
predictions = model.predict(new_features)

# 获取真实标签
true_labels = new_data['Label']

# 计算准确率和加权平均F1分数
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')
# 计算MCC指数
mcc = matthews_corrcoef(true_labels, predictions)

# 打印性能指标
print(f"Accuracy: {accuracy}")
print(f"Weighted F1 Score: {f1}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# 创建一个新的DataFrame来保存预测结果
predictions_df = pd.DataFrame({'SMILES': new_data['SMILES'], 'Label': predictions})

# 将预测结果保存到新的CSV文件中
output_csv_path = "./last/KNN_pred_0721.csv"
predictions_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")