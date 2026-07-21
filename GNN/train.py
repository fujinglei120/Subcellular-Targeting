import chemprop

def main():
    arguments = [
        '--seed', '0',
        # 185 9988 9961 9955 0.8
        # 9965 0.833
        # 9952
        # 拆分数据集使用的随机种子
        '--pytorch_seed', '0',

        '--data_path', 'E:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_train.csv',
        '--dataset_type', 'multiclass',
        # 'regression', 'classification', 'multiclass', 'spectra'
        '--multiclass_num_classes', '4',
        '--num_workers', '4',
        # 并行加载工作线程数
        '--number_of_molecules', '1',
        '--split_sizes', '0.9', '0.1', '0',
        '--split_type', 'random',
        # 'random', 'scaffold_balanced'
        # 'predetermined', 'crossval'
        # 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'
        '--separate_test_path', 'E:/BIT/codes/chemprop/data/20220710/last/SMILES_Canonical_of_AIEgens_for_Organelles_Imaging_test2.1.csv',

        '--ensemble_size', '1',

        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling',
        # When using rdkit_2d_normalized features, --no_features_scaling must be specified.
        # '--features_path', 'C:/BIT/codes/chemprop/data/20220710/molecules_descriptor.csv',

        '--depth', '2',
        '--hidden_size', '128',
        '--ffn_num_layers', '3',
        '--ffn_hidden_size', '256',
        '--init_lr', '0.0015',
        '--max_lr', '0.01',
        '--final_lr', '0.002',
        '--dropout', '0.05',

        '--activation', 'ReLU',
        # 激活函数，默认ReLU
        # 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'
        '--loss_function', 'cross_entropy',
        # 'mse', 'bounded_mse'
        # 'binary_cross_entropy', 'cross_entropy'
        # 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'
        '--log_frequency', '50',
        # 每经过多少批次(batch)记录一次训练的损失(loss)，默认10
        '--metric', 'accuracy',
        '--extra_metric', 'f1', 'mcc',
        # 'auc', 'prc-auc'
        # 'rmse', 'mae', 'mse'
        # 'r2', 'accuracy'
        # 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein'
        # 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse'
        # 用于评估模型的额外指标，默认无
        # '--metrics', '',
        # # 评估模型的指标列表，其中第一个指标用于早停判定

        '--batch_size', '512',
        '--epochs', '150',
        '--warmup_epochs', '10',
        '--num_folds', '1',

        '--save_dir', '20220710_4_checkpoints_20240811_best_01'
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

if __name__ == '__main__':
    main()