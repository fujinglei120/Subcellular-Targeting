import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from rdkit import Chem
from rdkit.Chem import AllChem
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# 读取包含SMILES和标签的数据
file_path = "./data/molecules+mhq_c_4.csv"
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

# 使用Keras的Tokenizer将字符转换为序列
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_sequences)
sequences = tokenizer.texts_to_sequences(smiles_sequences)

# 补齐序列以保证相同长度
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# 分割数据集为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(
    padded_sequences, data["Label"], test_size=0.2, random_state=42
)

# 构建CNN模型
embedding_dim = 128
filter_sizes = [3, 4, 5]
num_filters = 128

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim))
for filter_size in filter_sizes:
    model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))  # 添加池化层
model.add(GlobalMaxPooling1D())
model.add(Dense(units=len(data["Label"].unique()), activation='softmax'))

# 定义学习率调度器
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5
    elif epoch > 50:
        lr *= 0.5
    elif epoch > 30:
        lr *= 0.5
    elif epoch > 10:
        lr *= 0.5
    return lr

# 创建Adam优化器并设置学习率调度器
optimizer = Adam(learning_rate=lr_schedule(0))
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 设置学习率调度器为回调函数
lr_scheduler = LearningRateScheduler(lr_schedule)

# 训练模型
model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_data=(test_data, test_labels), callbacks=[lr_scheduler])

# 在测试集上进行预测
predictions_prob = model.predict(test_data)

# 选择具有最高概率的类别作为预测结果
predictions = predictions_prob.argmax(axis=1)

# 评估模型性能
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

# 计算 F1 分数
precision, recall, f1, support = precision_recall_fscore_support(test_labels, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 保存模型
model_filename = "cnn_model.h5"
model.save(model_filename)

print(f"Model saved to {model_filename}")