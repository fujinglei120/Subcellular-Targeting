import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from rdkit import Chem
from rdkit.Chem import AllChem
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense
from keras.wrappers.scikit_learn import KerasClassifier


# 读取包含SMILES和标签的数据
file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_train_20240717.csv"
column_names = ["SMILES", "Label"]
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 将SMILES转换为分子指纹
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将SMILES转换为字符序列
smiles_sequences = data['SMILES'].apply(lambda x: ' '.join(list(x)))
max_sequence_length = max(len(seq) for seq in smiles_sequences)
print("max_sequence_length = ",max_sequence_length)

# 使用Keras的Tokenizer将字符转换为序列
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_sequences)
sequences = tokenizer.texts_to_sequences(smiles_sequences)

# 补齐序列以保证相同长度
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# 定义创建CNN模型的函数
def create_model(optimizer='adam', embedding_dim=128, num_filters=128, filter_sizes=[3, 4, 5], batch_size=32, epochs=50):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim))
    for filter_size in filter_sizes:
        model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=len(data["Label"].unique()), activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 创建KerasClassifier包装器
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=1)

# 定义超参数网格
# param_grid = {
#     'optimizer': ['adam', 'rmsprop'],  # 优化器
#     'embedding_dim': [128, 256],  # embedding_dim定义了词向量的维度
#     'num_filters': [64, 128, 256],  # 卷积层的过滤器数量
#     'batch_size': [10, 20, 40, 60, 80, 100],
#     'epochs': [10, 20, 30, 40, 50]
# }

param_grid = {
    'optimizer': ['rmsprop'],  # 优化器
    'embedding_dim': [256],  # embedding_dim定义了词向量的维度
    'num_filters': [256],  # 卷积层的过滤器数量
    'batch_size': [10],
    'epochs': [50]
}

# 初始化KFold进行交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 使用GridSearchCV进行交叉验证和超参数优化
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf)
grid_search.fit(padded_sequences, data["Label"])

# 输出最佳参数和分数
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# 训练最佳模型并保存
model_filename = "./last/cnn_model_GridSearchCV_best_20240720.h5"
best_model = create_model(**grid_search.best_params_)
best_model.fit(padded_sequences, data["Label"], epochs=100, batch_size=32, verbose=1)
best_model.save(model_filename)
print(f"Best model saved to {model_filename}")