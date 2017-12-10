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

class FeaturizerManager():
    '''
    TODO: support checkpoint saving
    right now only Featurizer1 is supported
    '''
    def __init__(self, hdim, n_filters, model_name=None, featurizer_type='hs', cuda=False,
                lr=1e-4, batch_size=128, num_epochs=50, weight_decay=1e-3, plot_freq=10000,
                 checkpoint_freq=20000, verbose=False):
        self.verbose = verbose
        self.f = CardFeaturizer1(hdim, n_filters, cuda=cuda)
        self.data = {'train': {}, 'val': {}, 'test': {}}
        self.train_paths = g.glob(HS_DATA_TRAIN_PATH +'*')
        self.val_paths = g.glob(HS_DATA_VAL_PATH + '*')
        self.test_paths = g.glob(HS_DATA_TEST_PATH + '*')
        self.cuda = cuda
        self.save_path = initialize_save_folder(HS_DATA_PATH)
        self.global_step = 0

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.plot_freq = plot_freq
        self.checkpoint_freq = checkpoint_freq

        if model_name is None:
            self.model_name = 'c{}_h{}xf{}_model'.format(featurizer_type, hdim, n_filters)
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

        #print(x_hand_train, x_board_train, y_hs_train, y_probas_combi_train)
        #print(x_hand_val, x_board_val, y_hs_val, y_probas_combi_val)
        #print(x_hand_test, x_board_test, y_hs_test, y_probas_combi_test)
        #count, bins, _ = plt.hist(y_hs_train, bins=50)
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


    def train_featurizer1(self, includes_val=True, includes_test=False):
        '''
        TODO: train_featurizer11 and train_featurizer1 will be merged!
        '''
        train_dataset, val_dataset, test_dataset =\
        self._preprocess_data(includes_val=includes_val, includes_test=includes_test)
        optimizer = t.optim.Adam(self.f.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.verbose:
            print('loaded train data size')
            print(len(train_dataset))
            if includes_val:
                print('loaded validation data size')
                print(len(val_dataset))
            if includes_test:
                print('loaded test data size')
                print(len(test_dataset))
            print('')

        for epoch_i in range(self.num_epochs):
            if epoch_i % self.plot_freq == 0:
                print('running for {}th epoch'.format(epoch_i+1))
            # shuffle
            d = self.data['train']
            x_hand_train, x_board_train, y_hs_train, y_probas_combi_train \
          = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])

            # check if weights are NaN
            for p in self.f.parameters():
                if np.isnan(p.data.cpu().numpy()).sum()>0:
                    raise ValueError('nan weights !')


            for i in range(0, len(train_dataset), self.batch_size):
                batch_i = i // self.batch_size
                self.global_step += 1

                # init grad
                optimizer.zero_grad()

                # sample batch
                hand = variable(x_hand_train[i:i+self.batch_size], cuda=self.cuda)
                board = variable(x_board_train[i:i+self.batch_size], cuda=self.cuda)
                target = variable(y_hs_train[i:i+self.batch_size], cuda=self.cuda).squeeze()
                if len(hand) != self.batch_size:
                    break
                # pred
                HS_pred = self.f.forward(hand, board)[0].squeeze().float()

                HS = FeaturizerManager.clip(HS_pred)
                pred = t.log(HS/(1-HS))
                # QUESTION: why not average? now SSE not MSE
                loss = (target - pred)**2  # target already has this format (scaled logit)
                loss = t.sum(loss)
                self.train_losses.append(loss.data.cpu().numpy()[0]/self.batch_size)

                loss.backward()
                optimizer.step()
                if self.global_step % self.plot_freq == 0:
                    #display.clear_output(wait=True)

                    # test loss on a random test sample
                    # QUESTION: we may be validating for the same samples?
                    # we could have BatchIter class like cs281 homework
                    d = self.data['val']
                    x_hand_val, x_board_val, y_hs_val, y_probas_combi_val \
                    = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])
                    hand = variable(x_hand_val[:1000], cuda=self.cuda)
                    board = variable(x_board_val[:1000], cuda=self.cuda)
                    target = variable(y_hs_val[:1000], cuda=self.cuda).squeeze()
                    self.f.eval()

                    # QUESTION: why add 1e-5?
                    #HS_pred = self.f.forward(hand, board)[0].squeeze().float() + 1e-5
                    HS_pred = self.f.forward(hand, board)[0].squeeze().float()
                    HS = FeaturizerManager.clip(HS_pred)
                    # QUESTION: why train on validate data? do we even have train member function?
                    self.f.train()
                    pred = t.log(HS/(1-HS))
                    loss = (target - pred)**2
                    loss = t.sum(loss)
                    self.val_losses.append(loss.data.cpu().numpy()[0]/1000)

                    train_losses_ = moving_avg(self.train_losses)
                    if self.verbose:
                        print('validation loss')
                        print(np.mean(self.val_losses))
                        print('train loss')
                        print(np.mean(train_losses_[-10:]))
                        print('')

                    # plot

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

                if self.global_step % self.checkpoint_freq == 0:
                    # saving checkpoints
                    self.save_model(self.model_path, epoch_i, batch_i)
                    print('saved model', 'epoch: ', epoch_i, 'batch: ', batch_i)
        # after training is over, we save the model
        self.save_model(self.model_path, epoch_i, batch_i, is_best=True)
        print('saved final model', 'epoch: ', epoch_i, 'batch: ', batch_i)

    def plot(self):
        pass

    def test(self):
        pass

    def log_results(self):
        pass
