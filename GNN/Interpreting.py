import chemprop

def main():
    arguments = [
        # '--num_workers', '0',
        # 并行加载工作线程数
        '--data_path', 'E:/BIT/codes/chemprop/data/20220710/interpret/molecules+mhq_c_4_3_i.csv',
        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling',
        # When using rdkit_2d_normalized features, --no_features_scaling must be specified.
        '--checkpoint_dir', '20220710_4_checkpoints_20240313_RDKit_cv_best_3',

        '--max_atoms', '20',
        '--min_atoms', '8',

        '--batch_size', '500',
        '--rollout', '5',

        '--prop_delta', '0.5',
        '--property_id', '1'
    ]
    args = chemprop.args.InterpretArgs().parse_args(arguments)
    chemprop.interpret.interpret(args=args)

if __name__ == '__main__':
    main()