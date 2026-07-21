import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

# 加载模型
model = load_model('./last/cnn_model_GridSearchCV_best_20240720.h5')
print("Model loaded from disk")

# 读取CSV文件
file_path = "C:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_test_20240717.csv"
column_names = ["SMILES", "Label"]  # 假设CSV文件中包含了正确的标签列
data = pd.read_csv(file_path, usecols=column_names)

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

# 使用加载的模型进行预测
predictions_prob = model.predict(padded_sequences)

# 选择具有最高概率的类别作为预测结果
predictions = predictions_prob.argmax(axis=1)

# 获取真实标签
true_labels = data['Label']

# 计算准确率
accuracy = accuracy_score(true_labels, predictions)

# 使用F1分数
f1 = f1_score(true_labels, predictions, average='weighted')  # 或者 'micro' 或 'weighted'

# 计算MCC指数
mcc = matthews_corrcoef(true_labels, predictions)

# 打印性能指标
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# 创建一个新的DataFrame来保存预测结果
predictions_df = pd.DataFrame({'SMILES': data['SMILES'], 'Predicted_Label': predictions})

# 保存预测结果到CSV文件
predictions_df.to_csv('./last/CNN_pred_0720.csv', index=False)
print("Predictions saved to 'CNN_pred_0720.csv'")
