from models.featurizer import FeaturizerManager

FEATURIZER_NAME = 'chs_h50xf10_model_e9b22079_best'
SAVED_FEATURIZER_PATH = 'data/hand_eval/2017_11_30/saved_models/' + FEATURIZER_NAME

if __name__ == '__main__':
    # TODO: arg parser
    cuda = True
    verbose = False
    #fm = FeaturizerManager(hdim=50, n_filters=10,
    #                       plot_freq=10000, batch_size=64,
    #                       checkpoint_freq=20000,
    #                       cuda=cuda, verbose=verbose)
    #fm.train_featurizer1()
    f = FeaturizerManager.load_model(SAVED_FEATURIZER_PATH, cuda=cuda)

