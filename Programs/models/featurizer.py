# we are going to train featurizer here
# ideally we move all featurizer-related stuff here

import numpy as np
import torch as t
from IPython import display
import pickle
import os
import glob as g
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import time

from models.q_network import CardFeaturizer1
from game.utils import variable, moving_avg, initialize_save_folder
from game.game_utils import Card, cards_to_array
from game.errors import LoadModelError, NotImplementedError

# 60% 20% 20% picked arbitrarily
HS_DATA_PATH = 'data/hand_eval/'
HS_DATA_TRAIN_PATH = HS_DATA_PATH + 'train/'
HS_DATA_VAL_PATH = HS_DATA_PATH + 'valid/'
# should never touch this unless we feel good about the model
HS_DATA_TEST_PATH = HS_DATA_PATH + 'test/'
PLOT_PATH = 'img/'
MODEL_PATH = 'saved_models/'

# remove this later
def moving_avg(arr, window):
    return [np.mean(arr[i:i+window]) for i in range(len(arr) - window + 1)]

class FeaturizerManager():
    '''
    TODO: support checkpoint saving
    right now only Featurizer1 is supported
    '''
    def __init__(self, hdim, n_filters,
                 model_name=None,
                 cuda=False, lr=1e-4, batch_size=512,
                 num_epochs=50, weight_decay=1e-3, plot_freq=10000,
                 checkpoint_freq=20000,
                 verbose=False, tensorboard=None):
        self.verbose = verbose
        self.f = CardFeaturizer1(hdim, n_filters, cuda=cuda)
        self.data = {'train': {}, 'val': {}, 'test': {}}
        self.train_paths = g.glob(HS_DATA_TRAIN_PATH +'*')
        self.val_paths = g.glob(HS_DATA_VAL_PATH + '*')
        self.test_paths = g.glob(HS_DATA_TEST_PATH + '*')
        self.cuda = cuda
        self.save_path = initialize_save_folder(HS_DATA_PATH)
        self.global_step = 0
        self.tensorboard = tensorboard

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.plot_freq = plot_freq
        self.checkpoint_freq = checkpoint_freq

        if model_name is None:
            self.model_name = 'featurizer_{}'.format(time.time())
        else:
            self.model_name = model_name
        self.model_path = self.save_path + MODEL_PATH + self.model_name
        self.plot_path = self.save_path + PLOT_PATH
        self.train_losses = []
        self.val_losses = []

    def _load_data(self, paths):
        data = {}
        for filename in tqdm(paths):
            with open(filename, 'rb') as f:
                dataset_ = pickle.load(f)
                try:
                    # assuming all the data are in dict
                    data.update(dataset_)
                except:
                    data = dataset_
        return data


    def load_data(self, includes_val, includes_test):
        train_data = self._load_data(self.train_paths)
        val_data = None
        test_data = None
        if includes_val:
            val_data = self._load_data(self.val_paths)
        if includes_test:
            test_data = self._load_data(self.test_paths)
        return train_data, val_data, test_data


    def parse_dataset(self, dataset):
        x_hand = np.zeros((len(dataset), 13, 4))
        x_board = np.zeros((len(dataset), 3, 13, 4))
        y_hs = np.zeros((len(dataset), 1))
        y_probas_combi = np.zeros((len(dataset), 9))

        for i, (cards_, probas) in enumerate(dataset.items()):
            # if there are only two cards and nothing on the board then the keys don't have the same format.....
            if isinstance(cards_, tuple):
                hand_, board_ = cards_
            else:
                hand_ = cards_
                board_ = None

            hand = []
            board = []

            for card_ in hand_:
                if card_[0] != 'T':
                    try:
                        hand.append(Card(card_[0], card_[1]))
                    except:
                        print(hand_, board_)
                        raise Exception
                else:
                    hand.append(Card('10', card_[1]))

            if board_ is not None:
                for card_ in board_:
                    if card_[0] != 'T':
                        board.append(Card(card_[0], card_[1]))
                    else:
                        board.append(Card('10', card_[1]))

            x_hand[i] = cards_to_array(hand)
            x_board[i] = cards_to_array(board)
            hs = FeaturizerManager.clip(probas['player1winprob'])
            y_hs[i] = np.log(hs/(1-hs))
            y_probas_combi[i][0] = probas['High Card']
            y_probas_combi[i][1] = probas['Pair']
            y_probas_combi[i][2] = probas['Two Pair']
            y_probas_combi[i][3] = probas['Three of a Kind']
            y_probas_combi[i][4] = probas['Straight']
            y_probas_combi[i][5] = probas['Flush']
            y_probas_combi[i][6] = probas['Full House']
            y_probas_combi[i][7] = probas['Four of a Kind']
            y_probas_combi[i][8] = probas['Straight Flush']
        return x_hand, x_board, y_hs, y_probas_combi


    def _preprocess_data(self, includes_val, includes_test):
        train_dataset, val_dataset, test_dataset = self.load_data(includes_val, includes_test)

        x_hand_train, x_board_train, y_hs_train, y_probas_combi_train \
                = self.parse_dataset(train_dataset)

        scaler = StandardScaler()
        self.data['train']['x_hand'] = x_hand_train
        self.data['train']['x_board'] = x_board_train
        self.data['train']['y_hs'] = scaler.fit_transform(y_hs_train)
        self.data['train']['y_probs_combi'] = y_probas_combi_train

        if includes_val:
            x_hand_val, x_board_val, y_hs_val, y_probas_combi_val \
                    = self.parse_dataset(val_dataset)
            self.data['val']['x_hand'] = x_hand_val
            self.data['val']['x_board'] = x_board_val
            self.data['val']['y_hs'] = scaler.transform(y_hs_val)
            self.data['val']['y_probs_combi'] = y_probas_combi_val

        if includes_test:
            x_hand_test, x_board_test, y_hs_test, y_probas_combi_test \
                    = self.parse_dataset(test_dataset)
            self.data['test']['x_hand'] = x_hand_test
            self.data['test']['x_board'] = x_board_test
            self.data['test']['y_hs'] = scaler.transform(y_hs_test)
            self.data['test']['y_probs_combi'] = y_probas_combi_test

        return train_dataset, val_dataset, test_dataset

    def save_model(self, path, epoch_i, batch_i, is_best=False):
        path += '_e{}b{}'.format(epoch_i, batch_i)
        if is_best:
            uniq_path = path + '_best'
        else:
            i = 1
            uniq_path = path
            while os.path.isfile(uniq_path):
                uniq_path = path + str(i)
                i += 1
        t.save(self.f.state_dict(), uniq_path)


    @staticmethod
    def load_model(path, cuda=False):
        if os.path.isfile(path):
            # TODO: hardcoding hdim and nfliters
            f = CardFeaturizer1(hdim=50, n_filters=10, cuda=cuda)
            if cuda:
                f.load_state_dict(t.load(path, map_location=lambda storage, loc:storage.cuda(0)))
            else:
                f.load_state_dict(t.load(path))
            for param in f.parameters():
                # freeze weights
                param.requires_grad = False
            print('loaded gpu-enabled Featurizer? -> ', next(f.parameters()).is_cuda)
            return f
        else:
            raise LoadModelError('The path does not exist')

    @staticmethod
    def clip(x):
        try:
            # if x is in ByteTensor
            return x*(x>=1e-2).float()*(x<=.99).float() + .99*(x>.99).float() + .01*(x<.01).float()
        except:
            return x*(x>=1e-2)*(x<=.99) + .99*(x>.99) + .01*(x<.01)


    @staticmethod
    def inv_freq(x, count, bins):
        """
        Give a weight to each loss inversely proportional to their occurence frequency
        This way the model doesn't learn to just output the most common value
        """
        total = float(count.sum())
        weight = 0.0
        for k in range(count.shape[0]-1):
            c, b0,b1 = float(count[k]),float(bins[k]),float(bins[k+1])
            try:
                weight += (total/c)*((x>=b0).Managerfloat())*((x<b1).float())
            except:
                weight = (total/c)*((x>=b0).float())*((x<b1).float())
        return weight


    def train_featurizer(self):
        '''
        TODO: train_featurizer11 and train_featurizer1 will be merged!
        '''
        train_dataset, val_dataset, test_dataset =\
        self._preprocess_data(includes_val=True, includes_test=True)
        optimizer = t.optim.Adam(self.f.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch_i in range(self.num_epochs):
            d = self.data['train']
            x_hand_train, x_board_train, y_hs_train, y_probas_combi_train \
          = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])

            # check if weights are NaN
            start_t = time.time()
            for p in self.f.parameters():
                if np.isnan(p.data.cpu().numpy()).sum()>0:
                    raise ValueError('nan weights !')

            for i in range(0, len(train_dataset), self.batch_size):
                batch_i = i // self.batch_size

                optimizer.zero_grad()

                hand = variable(x_hand_train[i:i+self.batch_size], cuda=self.cuda)
                board = variable(x_board_train[i:i+self.batch_size], cuda=self.cuda)
                target = variable(y_hs_train[i:i+self.batch_size], cuda=self.cuda).squeeze()
                if len(hand) != self.batch_size:
                    break
                # pred
                HS_pred = self.f.forward(hand, board)[0].squeeze().float()

                HS = FeaturizerManager.clip(HS_pred)
                pred = t.log(HS/(1-HS))
                loss = (target - pred).pow(2).mean()
                raw_loss = float(round(loss.data.cpu().numpy()[0], 2))

                if self.tensorboard is not None:
                    self.tensorboard.add_scalar_value('train_loss', raw_loss)

                loss.backward()
                optimizer.step()
            print(time.time()-start_t, 'seconds per epoch')
            # epoch ended
            self.validate_featurizer(val_dataset, against_test_data=False)
            self.save_model(self.model_path, epoch_i, batch_i)
            self.tensorboard.to_zip('data/hand_eval/tensorboard/{}'.format(time.time()))
            print('saved model', 'epoch: ', epoch_i, 'batch: ', batch_i)

        # after training is over, we save the model
        self.validate_featurizer(test_dataset, against_test_data=True)
        self.save_model(self.model_path, epoch_i, batch_i, is_best=True)
        self.tensorboard.to_zip('data/hand_eval/tensorboard/{}'.format(time.time()))
        print('saved final model', 'epoch: ', epoch_i, 'batch: ', batch_i)


    def validate_featurizer(self, dataset, against_test_data=False):
        if against_test_data:
            d = self.data['test']
        else:
            d = self.data['val']

        # hardcode epochs
        x_hand, x_board, y_hs, y_probas_combi = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])

        for i in range(0, len(dataset), self.batch_size):
            batch_i = i // self.batch_size
            hand = variable(x_hand[i:i+self.batch_size], cuda=self.cuda)
            board = variable(x_board[i:i+self.batch_size], cuda=self.cuda)
            target = variable(y_hs[i:i+self.batch_size], cuda=self.cuda).squeeze()

            self.f.eval()
            if len(hand) != self.batch_size:
                break

            HS_pred = self.f.forward(hand, board)[0].squeeze().float()
            HS = FeaturizerManager.clip(HS_pred)

            self.f.train()
            # for numerical stable
            m = variable(np.array([1e-6]), cuda=self.cuda)
            pred = t.log(t.max(HS/(1-HS), m))
            loss = (target - pred).pow(2).mean()
            loss = float(round(loss.data.cpu().numpy()[0], 3))

            if self.tensorboard is not None:
                if against_test_data:
                    self.tensorboard.add_scalar_value('test_loss', loss)
                else:
                    self.tensorboard.add_scalar_value('validation_loss', loss)


    def plot(self):
        fig = plt.figure(figsize=(10,10))
        plt.plot(train_losses_, label='train loss (smoothed)')
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('{}train_loss_e{}b{}'.format(self.plot_path, epoch_i, batch_i), ppi=300, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(10,10))
        plt.plot(self.val_losses, label='validation loss')
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        fig.savefig('{}val_loss_e{}b{}'.format(self.plot_path, epoch_i, batch_i), ppi=300, bbox_inches='tight')
        plt.close()
