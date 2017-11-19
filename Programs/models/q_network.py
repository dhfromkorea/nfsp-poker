import torch as t
from torch.nn import Conv1d as conv, SELU, Linear as fc, Softmax, Sigmoid
import torch.nn as nn
import torch.optim as optim
import numpy as np


selu = SELU()
softmax = Softmax()
sigmoid = Sigmoid()


def get_shape(x):
    try:
        return x.data.numpy().shape
    except:
        return x.numpy().shape


def flatten(x):
    shape = get_shape(x)
    return x.resize(shape[0], int(np.prod(shape[1:])))


class CardFeaturizer(t.nn.Module):
    def __init__(self, hdim):
        super(CardFeaturizer, self).__init__()
        self.hdim = hdim
        self.conv1 = conv(2, 5, 1)
        self.conv2 = conv(5, 1, 5, padding=2)
        self.conv3 = conv(5, 1, 3, padding=1)
        self.conv4 = conv(5, 3, 3, dilation=2, padding=2)
        self.conv5 = conv(2, 1, 1)

        self.fc1 = fc(13 * 5, hdim)
        self.fc2 = fc(13 * 2, hdim)
        self.fc3 = fc(hdim, hdim)
        self.fc4 = fc(4, hdim)
        self.fc5 = fc(52, hdim)
        self.fc6 = fc(hdim, hdim)
        self.fc7 = fc(52, hdim)
        self.fc8 = fc(hdim, hdim)
        self.fc9 = fc(52, hdim)
        self.fc10 = fc(hdim, hdim)
        self.fc11 = fc(3 * hdim, hdim)
        self.fc12 = fc(52, hdim)
        self.fc13 = fc(hdim, hdim)
        self.fc14 = fc(5 * hdim, hdim)
        self.fc15 = fc(hdim, hdim)
        self.fc16 = fc(hdim, 9)  # 9 is the number of combinations (high card, pair, two pairs, trips, straight, flush, full house, quads, straight flush
        self.fc17 = fc(hdim, 9)
        self.fc18 = fc(hdim, 1)

    def forward(self, hand, board):
        # DETECTING PATTERNS IN THE BOARD AND HAND
        # Aggregate by suit and kind
        color_hand = t.sum(hand, 1)
        color_board = t.sum(t.sum(board, 2), 1)
        kinds_hand = t.sum(hand, -1)
        kinds_board = t.sum(t.sum(board, -1), 1)
        colors = t.cat([color_hand.resize(len(color_hand), 1, 4), color_board.resize(len(color_board), 1, 4)], 1)
        kinds = t.cat([kinds_hand.resize(len(kinds_hand), 1, 13), kinds_board.resize(len(kinds_board), 1, 13)], 1)

        # Process board and hand to detect straights using convolutions with kernel size 5, 3, and 3 with dilation
        kinds_straight = selu(self.conv1((kinds > 0).float()))
        kinds_straight = t.cat([
            selu(self.conv2(kinds_straight)),
            selu(self.conv3(kinds_straight)),
            selu(self.conv4(kinds_straight))
        ], 1)
        kinds_straight = flatten(kinds_straight)
        kinds_straight = selu(self.fc1(kinds_straight))

        # Process board and hand to detect pairs, trips, quads, full houses
        kinds_ptqf = selu(self.fc2(flatten(kinds)))
        kinds_ptqf = selu(self.fc3(kinds_ptqf))

        # Process board and hand to detect flushes
        colors = flatten(selu(self.conv5(colors)))
        colors = selu(self.fc4(colors))

        # Process the board with FC layers
        flop_alone = selu(self.fc5(flatten(board[:, 0, :, :])))
        flop_alone = selu(self.fc6(flop_alone))
        turn_alone = selu(self.fc7(flatten(board[:, 1, :, :])))
        turn_alone = selu(self.fc8(turn_alone))
        river_alone = selu(self.fc9(flatten(board[:, 2, :, :])))
        river_alone = selu(self.fc10(river_alone))
        board_alone = selu(self.fc11(t.cat([flop_alone, turn_alone, river_alone], -1)))

        # Process board and hand together with FC layers
        h = selu(self.fc12(flatten(hand)))
        h = selu(self.fc13(h))
        bh = selu(self.fc14(t.cat([h, board_alone, colors, kinds_ptqf, kinds_straight], -1)))
        bh = selu(self.fc15(bh))

        # Predict probabilities of having a given hand + hand strength
        probabilities_of_each_combination_board_only = softmax(self.fc16(board_alone))
        probabilities_of_each_combination_board_hand = softmax(self.fc17(bh))
        hand_strength = sigmoid(self.fc18(bh))
        return hand_strength, probabilities_of_each_combination_board_hand, probabilities_of_each_combination_board_only, flop_alone, turn_alone, river_alone, bh


class SharedNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim):
        super(SharedNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        hdim = hidden_dim

        self.fc19 = fc(5 * 6 * 2, hdim)
        self.fc20 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc21 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc22 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc23 = fc(hdim, hdim)
        self.fc24 = fc(5, hdim)
        self.fc25 = fc(3 * hdim, hdim)
        self.fc26 = fc(hdim, hdim)

    def forward(self, cards_features, flop_features, turn_features, river_features, board_hand_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        # PROCESS THE ACTIONS THAT WERE TAKEN IN THE CURRENT EPISODE
        processed_preflop = selu(self.fc19(flatten(preflop_plays)))
        processed_flop = selu(self.fc20(t.cat([flatten(flop_plays), flop_features], -1)))
        processed_turn = selu(self.fc21(t.cat([flatten(turn_plays), turn_features], -1)))
        processed_river = selu(self.fc22(t.cat([flatten(river_plays), river_features], -1)))
        plays = selu(self.fc23(processed_preflop + processed_flop + processed_turn + processed_river))

        # add pot, dealer, blinds, dealer, stacks
        pbds = selu(self.fc24(t.cat([pot, stack, opponent_stack, big_blind, dealer], -1)))

        # USE ALL INFORMATION (CARDS/ACTIONS/MISC) TO PREDICT THE Q VALUES
        situation_with_opponent = selu(self.fc25(t.cat([plays, pbds, board_hand_features], -1)))
        situation_with_opponent = selu(self.fc26(situation_with_opponent))

        return situation_with_opponent


class QNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, shared_network=None, pi_network=None,
                 learning_rate=0.01):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        hdim = self.hidden_dim

        assert not (shared_network is not None and pi_network is not None), "you should provide either pi_network or shared_network"
        if pi_network is not None:
            self.shared_network = pi_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetwork(n_actions, hidden_dim)
        for i in range(19, 27):
            setattr(self, 'fc' + str(i), getattr(self.shared_network, 'fc' + str(i)))
        self.fc27 = fc(hdim, hdim)
        self.fc28 = fc(hdim, n_actions)

        # loss as MSE
        self.criterion = nn.MSELoss()
        # TODO: need model params
        # need to build model using model.sequential ...
        # https://github.com/harvard-ml-courses/cs281-demos/blob/master/SoftmaxTorch.ipynb
        self.optim = optim.SGD([self.fc27.weight], lr=learning_rate)

    def forward(self, cards_features, flop_features, turn_features, river_features, board_hand_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        situation_with_opponent = self.shared_network.forward(cards_features, flop_features, turn_features, river_features, board_hand_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                                                                                                                                                                         flop_plays, turn_plays, river_plays)
        q_values = selu(self.fc27(situation_with_opponent))
        q_values = self.fc28(q_values)
        return q_values

    def train(self, states, actions, targets, imp_weights):
        self.optim.zero_grad()
        # TODO: support batch forward?
        # not sure if it's supported as it's written now
        Q_preds = self.forward(*states)[:, 0].squeeze()
        loss, td_deltas = self.compute_loss(Q_preds, Q_targets, imp_weights)
        loss.backward()
        # update weights
        self.optim.step()
        return td_deltas

    def compute_loss(self, x, y, imp_weights):
        '''
        compute weighted mse loss
        loss for each sample is scaled by imp_weight
        we need this to account for bias in replay sampling
        '''
        td_deltas = x - y
        mse = imp_weights.dot(td_deltas.pow(2)).mean()
        return mse, td_deltas


class PiNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, shared_network=None, q_network=None):
        super(PiNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        hdim = self.hidden_dim

        assert not (shared_network is not None and q_network is not None), "you should provide either q_network or shared_network"
        if q_network is not None:
            self.shared_network = q_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetwork(n_actions, hidden_dim)
        for i in range(19, 27):
            setattr(self, 'fc' + str(i), getattr(self.shared_network, 'fc' + str(i)))
        self.fc27 = fc(hdim, hdim)
        self.fc28 = fc(hdim, n_actions)
        # TODO: add softmax loss

    def forward(self, cards_features, flop_features, turn_features, river_features, board_hand_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        situation_with_opponent = self.shared_network.forward(cards_features, flop_features, turn_features, river_features, board_hand_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                                                                                                                                                                         flop_plays, turn_plays, river_plays)
        pi_values = selu(self.fc27(situation_with_opponent))
        pi_values = softmax(self.fc28(pi_values))
        return pi_values
