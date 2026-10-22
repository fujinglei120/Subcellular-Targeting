from rdkit import Chem
import pandas as pd

# 读取CSV文件
data = pd.read_csv('E:/BIT/codes/chemprop/data/AIE_ACQ/TEST20250504.csv')

# 创建一个空列表来存储分子的 Canonical SMILES
canonical_smiles = []

# 遍历每个 SMILES，并将其转换为 Canonical SMILES
for smiles in data['SMILES']:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 生成标准SMILES
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            canonical_smiles.append(canonical)
        else:
            # 如果无法生成分子，则添加None
            canonical_smiles.append(None)
    except Exception as e:
        # 如果处理过程中出现异常，则添加None
        print(f"Error processing SMILES: {smiles}")
        print(e)
        canonical_smiles.append(None)

# 将 Canonical SMILES 添加到数据中
data['Canonical_SMILES'] = canonical_smiles

# 保存包含 Canonical SMILES 的新数据到CSV文件
data.to_csv('E:/BIT/codes/chemprop/data/AIE_ACQ/TEST20250504_canonical.csv', index=False)
print("Canonical SMILES have been saved to 'E:/BIT/codes/chemprop/data/AIE_ACQ/TEST20250504_canonical.csv'")