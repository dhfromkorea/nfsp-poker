from models.featurizer import FeaturizerManager

if __name__ == '__main__':
    # TODO: arg parser
    cuda = True
    verbose = False
    fm = FeaturizerManager(hdim=50, n_filters=10,
                           plot_freq=10000, batch_size=64,
                           checkpoint_freq=20000,
                           cuda=cuda, verbose=verbose)
    fm.train_featurizer1()

