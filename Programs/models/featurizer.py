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


from models.q_network import CardFeaturizer11
from game.utils import variable, moving_avg
from game.game_utils import Card, cards_to_array


# 60% 20% 20% picked arbitrarily 
HS_DATA_PATH = 'data/hand_eval/'
HS_DATA_TRAIN_PATH = HS_DATA_PATH + 'train/'
HS_DATA_VAL_PATH = HS_DATA_PATH + 'val/'
# should never touch this unless we feel good about the model
HS_DATA_TEST_PATH = HS_DATA_PATH + 'test/'
PLOT_PATH = HS_DATA_PATH + 'img/'
MODEL_PATH = HS_DATA_PATH + 'saved_models/'

class FeaturizerManager():
    '''
    TODO: support checkpoint saving
    '''
    def __init__(self, hdim, n_filters, model_name=None, featurizer_type='11'):
        self.f = CardFeaturizer11(hdim, n_filters)
        self.data = {'train': {}, 'val': {}, 'test': {}}
        self.train_paths = g.glob(HS_DATA_TRAIN_PATH +'*.p')[:3]
        self.val_paths = g.glob(HS_DATA_VAL_PATH + '*')[:3]
        self.test_paths = g.glob(HS_DATA_TEST_PATH + '*t')[:3]

        self.lr = 1e-4
        self.batch_size = 64
        self.num_epochs = 50
        self.weight_decay = 1e-3
        self.train_losses = []
        self.val_losses = []
        self.plot_freq = 500
        if model_name is None:
            self.model_name = '{}_{}x{}_model'.format(featurizer_type, hdim, n_filters)

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


    def load_data(self, includes_val=True, includes_test=False):
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


    @staticmethod
    def clip(x):
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


    def _preprocess_data(self, includes_val=True, includes_test=False):
        train_dataset, val_dataset, test_dataset = self.load_data(True, True)

        print(len(train_dataset), len(test_dataset), len(val_dataset))

        x_hand_train, x_board_train, y_hs_train, y_probas_combi_train \
                = self.parse_dataset(train_dataset)
        x_hand_val, x_board_val, y_hs_val, y_probas_combi_val \
                = self.parse_dataset(val_dataset)
        x_hand_test, x_board_test, y_hs_test, y_probas_combi_test \
                = self.parse_dataset(test_dataset)

        scaler = StandardScaler()
        self.data['train']['x_hand'] = x_hand_train
        self.data['train']['x_board'] = x_board_train
        self.data['train']['y_hs'] = scaler.fit_transform(y_hs_train)
        self.data['train']['y_probs_combi'] = y_probas_combi_train
        
        if includes_val:
            self.data['val']['x_hand'] = x_hand_val
            self.data['val']['x_board'] = x_board_val
            self.data['val']['y_hs'] = scaler.transform(y_hs_val)
            self.data['val']['y_probs_combi'] = y_probas_combi_val

        if includes_test:
            self.data['test']['x_hand'] = x_hand_test
            self.data['test']['x_board'] = x_board_test
            self.data['test']['y_hs'] = scaler.transform(y_hs_test)
            self.data['test']['y_probs_combi'] = y_probas_combi_test

        print(x_hand_train, x_board_train, y_hs_train, y_probas_combi_train)
        print(x_hand_val, x_board_val, y_hs_val, y_probas_combi_val)
        print(x_hand_test, x_board_test, y_hs_test, y_probas_combi_test)

        #count, bins, _ = plt.hist(y_hs_train, bins=50)
        return train_dataset, val_dataset, test_dataset

    def save_model(self, path):
        if os.path.isfile(path):
            t.save(self.f.state_dict(), path + '1')
        else:
            t.save(self.f.state_dict(), path)


    def load_model(self, path):
        if os.path.isfile(path):
            self.f.load_state_dict(t.load(path))
        else:
            raise LoadModelError('The path does not exist')

    
    def train(self):
        print('hold your horses! processing data')

        train_dataset, val_dataset, test_dataset = self._preprocess_data()
        # get variables
        print('#params: ', np.sum([np.prod(p.data.numpy().shape) for p in self.f.parameters()]))
        optimizer = t.optim.Adam(self.f.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        for _ in range(self.num_epochs):
            print('epoch', _)
            # shuffle
            d = self.data['train']
            x_hand_train, x_board_train, y_hs_train, y_probas_combi_train \
          = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])
            # check if weights are NaN

            for p in self.f.parameters():
                if np.isnan(p.data.numpy()).sum()>0:
                    raise ValueError('nan weights !')

            for i in range(0, len(train_dataset), self.batch_size):
                # check if BN var is NaN
        #         check_bn_var_nan(self.f)
                # sample batch
                hand = variable(x_hand_train[i:i+self.batch_size])
                board = variable(x_board_train[i:i+self.batch_size])
                target_HS = variable(y_hs_train[i:i+self.batch_size])
                target_probas = variable(y_probas_combi_train[i:i+self.batch_size])
                if len(hand) != self.batch_size:
                    break
                # init grad
                optimizer.zero_grad()
                # pred
                HS, probas = self.f.forward(hand, board)
                HS = FeaturizerManager.clip(HS.squeeze()).float()
                pred_HS = t.log(HS/(1-HS)).squeeze()
                mse = (target_HS - pred_HS)**2  # target aself.lready has this format (scaled logit)
                kl_div = t.sum(target_probas[target_probas>0]*t.log(target_probas[target_probas>0]/probas[target_probas>0]), -1)
                # KL divergence between target distribution and predicted distribution
                loss = mse + kl_div
        #         weights = inv_freq(target, count, bins)  # give more importance to rare samples
        #         loss = loss * weights
                loss = t.sum(loss)
                self.train_losses.append(loss.data.numpy()[0]/self.batch_size)
                loss.backward()
        #       clip_gradients(f, 5)

                optimizer.step()
                if (i//self.batch_size) % self.plot_freq == 0:
                    display.clear_output(wait=True)
                    print('epoch', _)
                    # test loss on a random test sample
                    d = self.data['val']
                    x_hand_val, x_board_val, y_hs_val, y_probas_combi_val \
                    = shuffle(d['x_hand'], d['x_board'], d['y_hs'], d['y_probs_combi'])
                    hand = variable(x_hand_val[:1000])
                    board = variable(x_board_val[:1000])
                    target_hs = variable(y_hs_val[:1000]).squeeze()
                    target_probas = variable(y_probas_combi_val[:1000]).squeeze()
                    self.f.eval()
                    HS, probas = self.f.forward(hand, board)
                    HS = clip(HS)
                    # why train again here with test data??? cheating?
                    self.f.train()
                    pred_hs = t.log(HS/(1-HS))
                    mse = (target_hs - pred_hs)**2
                    kl_div = t.sum(target_probas*((target_probas>0).float())*t.log(target_probas/probas), -1)
                    loss = t.sum(mse+kl_div)
                    self.val_losses.append(loss.data.numpy()[0]/1000)
                    # plot
                    fig = plt.figure(figsize=(10,10))
                    print(self.val_losses[-10:])
                    plt.plot(self.val_losses)
                    fig.savefig('{}val_loss_i{}'.format(PLOT_PATH, i), ppi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    fig = plt.figure(figsize=(10,10))
                    self.train_losses_ = moving_avg(self.train_losses)
                    print(self.train_losses_[-10:])
                    plt.plot(self.train_losses_)
                    fig.savefig('{}train_loss_i{}'.format(PLOT_PATH, i), ppi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
        # after training is over, we save the model
        self.save_model(MODEL_PATH + self.model_name)
    
    def plot(self):
        pass

    def test(self):
        pass

    def log_results(self):
        pass
