import chemprop

def main():
    arguments = [
        '--test_path', 'E:/BIT/codes/chemprop/data/20220710/test/20220710design.csv',
        '--preds_path', 'E:/BIT/codes/chemprop/data/20220710/test/20220710design_pred.csv',
        '--checkpoint_dir', '20220710_4_checkpoints_20240721_best_01',
        '--num_workers', '4',

        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    chemprop.train.make_predictions(args=args)

if __name__ == '__main__':
    main()

