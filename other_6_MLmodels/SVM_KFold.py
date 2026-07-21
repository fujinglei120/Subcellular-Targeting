import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
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
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 创建预处理和模型训练的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('svm', SVC(probability=True))  # SVM分类器
])

# 设置超参数网格
param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly'],  # 添加多项式核
    'svm__C': [0.1, 1, 10, 100],  # 扩展C的范围
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # 扩展gamma的范围
    'svm__degree': [2, 3, 4],  # 多项式核的度数
    'svm__coef0': [0, 1, 2],  # 独立于特征的核函数系数
    'svm__tol': [1e-3, 1e-4],  # 优化过程中的容忍度
    # 'svm__max_iter': [100, 1000],  # 最大迭代次数
    'svm__class_weight': ['balanced', None],  # 类权重
    # 'svm__kernel': ['linear', 'rbf'],  # 核函数类型
    # 'svm__C': [0.1, 1, 10],  # 正则化参数
    # 'svm__gamma': ['scale', 'auto']  # RBF核函数的参数，它定义了单个训练样本的影响范围，可以看作是核函数的“带宽”。
    # gamma值较小意味着更远的影响范围，而较大的gamma值意味着更近的影响范围
}

# 初始化KFold进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV，使用交叉验证
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# 计算特征
features = features.values
labels = data["Label"].values

# 执行超参数优化
grid_search.fit(features, labels)

# 打印最佳参数和性能
print(f"Best parameters found: {grid_search.best_params_}")

# 保存最佳模型和特征缩放器
model_filename = "./last/svm_model_GridSearchCV_best_20240718.pkl"
scaler_filename = "./last/svm_scaler_GridSearchCV_best_20240718.pkl"

joblib.dump(grid_search.best_estimator_, model_filename)
joblib.dump(grid_search.best_estimator_.named_steps['scaler'], scaler_filename)

print(f"Best model and scaler saved to {model_filename} and {scaler_filename}")