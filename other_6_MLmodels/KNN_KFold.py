import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# 读取包含SMILES和标签的数据
file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_train_20240717.csv"
column_names = ["SMILES", "Label"]
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 通过RDKit计算Morgan指纹作为特征
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))).values.astype(int))

# 创建预处理和模型训练的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('knn', KNeighborsClassifier())
])

# 设置超参数网格
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],  # KNN中的K值，指K个临近样本
    'knn__weights': ['uniform', 'distance'],  # 权重类型
    'knn__p': [1, 2, 3]  # 距离度量方式，如1为曼哈顿距离，2为欧几里得距离，3为闵可夫斯基距离
}

# 初始化KFold进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV，使用交叉验证
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# 执行超参数优化
grid_search.fit(features, data["Label"])

# 打印最佳超参数组合
print("Best parameters found:")
print(grid_search.best_params_)

# 保存最佳模型
model_filename = "./last/knn_model_GridSearchCV_best_20240721.pkl"
joblib.dump(grid_search.best_estimator_, model_filename)

print(f"Best model saved to {model_filename}")