import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# 读取包含SMILES和标签的数据
file_path = "./data/molecules+mhq_c_4.csv"
column_names = ["SMILES", "Label"]
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 通过RDKit计算Morgan指纹作为特征
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))).values.astype(int))

# 分割数据集为训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(
    features, data["Label"], test_size=0.2, random_state=42
)

# 创建并训练MLP模型
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(train_features_scaled, train_labels)

# 在测试集上进行预测
predictions = model.predict(test_features_scaled)

# 评估模型性能
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

# 计算F1分数
f1 = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", classification_report(test_labels, predictions))

# 保存模型
model_filename = "mlp_model.pkl"
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")