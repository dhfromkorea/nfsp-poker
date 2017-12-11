from models.featurizer import FeaturizerManager

from pycrayon import CrayonClient
import time

FEATURIZER_NAME = 'chs_h50xf10_model_e9b22079_best'
SAVED_FEATURIZER_PATH = 'data/hand_eval/2017_11_30/saved_models/' + FEATURIZER_NAME


def setup_tensorboard(exp_id, cur_t, hostname, port):
    exp_filename = '{}_{}'.format(cur_t, exp_id)
    tb = CrayonClient(hostname=hostname, port=port)
    try:
        tb_experiment = tb.create_experiment(exp_filename)
    except:
        # flush the data anew
        tb.remove_experiment(exp_filename)
        tb_experiment = tb.create_experiment(exp_filename)
    return tb_experiment, tb

if __name__ == '__main__':
    # TODO: arg parser
    cuda = True
    verbose = False
    exp_id = 'featurizer1_train'
    plot_freq = 100
    checkpoint_freq = 1000
    tb_experiment, _ = setup_tensorboard(exp_id, time.time(), 'localhost', '8889')

    fm = FeaturizerManager(hdim=50,
                           n_filters=10,
                           plot_freq=plot_freq,
                           batch_size=1024,
                           num_epochs=100,
                           checkpoint_freq=checkpoint_freq,
                           tensorboard=tb_experiment,
                           cuda=cuda,
                           verbose=verbose)
    fm.train_featurizer()

