import pandas as pd
import joblib  # 使用joblib代替pickle以支持更复杂的对象
from rdkit import Chem
from rdkit.Chem import AllChem

# 加载模型
model_filename = "xgb_model.pkl"
model = joblib.load(model_filename)

# 读取新的CSV文件
new_data_file_path = "molecules+mhq_c_4.csv"
new_data = pd.read_csv(new_data_file_path)

# 使用RDKit计算新数据的Morgan指纹作为特征
new_data['Morgan_Features'] = new_data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(new_data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 使用模型进行预测
predictions = model.predict(new_features)

# 创建一个新的DataFrame来保存预测结果
predictions_df = pd.DataFrame({'SMILES': new_data['SMILES'], 'Label': predictions})

# 将预测结果保存到新的CSV文件中
output_csv_path = "XGBoost_pred_canonical.csv"
predictions_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")