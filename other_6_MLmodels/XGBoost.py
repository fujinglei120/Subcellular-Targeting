import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

file_path = "./data/molecules+mhq_c_4.csv"
column_names = ["SMILES", "Label"]

# 读取数据
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 通过RDKit计算Morgan指纹作为特征
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 分割数据集为训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(
    features, data["Label"], test_size=0.2, random_state=42
)

# 模型训练
model = XGBClassifier(objective="multi:softmax", num_class=len(data["Label"].unique()))
model.fit(train_features, train_labels)

# 模型预测
predictions = model.predict(test_features)

# 评估模型性能
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

# 计算F1分数
f1 = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", classification_report(test_labels, predictions))

# 保存模型
model_filename = "xgb_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")