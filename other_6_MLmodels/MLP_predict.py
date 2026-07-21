import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# 加载保存的MLP模型
model_file = "./last/mlp_model_GridSearchCV_best_20240721.pkl"
model = joblib.load(model_file)

# 读取CSV文件
file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_test_20240717.csv"
data = pd.read_csv(file_path)

# 确保新数据集中包含真实标签列
if 'Label' not in data.columns:
    raise ValueError("The new data must contain a 'Label' column with the true labels.")

# 假设CSV文件中SMILES数据在名为'SMILES'的列中
smiles_list = data['SMILES'].tolist()

# 对SMILES数据进行预处理
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 使用加载的模型进行预测
predictions = model.predict(new_features)

# 获取真实标签
true_labels = data['Label'].tolist()

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
predictions_df = pd.DataFrame({'SMILES': data['SMILES'], 'Label': predictions})

# 将预测结果保存到新的CSV文件中
output_data = pd.DataFrame({'SMILES': smiles_list, 'Prediction': predictions})
output_file_path = "./last/MLP_pred_0721.csv"  # 你希望保存的输出文件路径
output_data.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")