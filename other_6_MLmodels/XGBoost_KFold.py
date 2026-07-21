import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBClassifier
import pickle

# 定义文件路径和列名
file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_train_20240717.csv"
column_names = ["SMILES", "Label"]

# 读取数据
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 通过RDKit计算Morgan指纹作为特征
data['Morgan_Features'] = data['SMILES'].apply(
    lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 定义模型管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('clf', XGBClassifier(objective="multi:softmax", num_class=len(data["Label"].unique())))
])

# 定义超参数网格
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__learning_rate': [0.01, 0.1, 0.5],
    'clf__max_depth': [3, 5, 7],
    'clf__gamma': [0, 0.1, 0.5],
    'clf__colsample_bytree': [0.5, 0.8, 1.0]
    # 'clf__scale_pos_weight': [1, 2, 3]  # 正样本与负样本的权重比例。在不平衡的数据集中，调整这个参数可以帮助模型更好地学习
}

# 初始化KFold进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)
# f1_weighted

# 进行超参数优化
grid_search.fit(features, data["Label"])

# 打印最佳超参数组合
print("Best parameters found:")
print(grid_search.best_params_)

# 保存最佳模型
best_model = grid_search.best_estimator_
model_filename = "./last/xgb_model_GridSearchCV_best_20240718.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model saved to {model_filename}")