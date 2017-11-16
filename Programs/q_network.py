import torch as t
from torch.nn import Conv1d as conv, SELU, Linear as fc, Softmax, Sigmoid
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


class SharedNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim):
        super(SharedNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        hdim = self.hidden_dim
        n_actions = self.n_actions

        # DETECTING PATTERNS IN THE BOARD AND HAND
        # Aggregate by suit and kind
        color_hand = t.sum(hand, 1)
        color_board = t.sum(t.sum(board, 2), 1)
        kinds_hand = t.sum(hand, -1)
        kinds_board = t.sum(t.sum(board, -1), 1)

        colors = t.cat([color_hand.resize(len(color_hand), 1, 4), color_board.resize(len(color_board), 1, 4)], 1)
        kinds = t.cat([kinds_hand.resize(len(kinds_hand), 1, 13), kinds_board.resize(len(kinds_board), 1, 13)], 1)

        # Process board and hand to detect straights using convolutions with kernel size 5, 3, and 3 with dilation
        kinds_straight = selu(conv(2, 5, 1)((kinds > 0).float()))
        kinds_straight = t.cat([
            selu(conv(5, 1, 5, padding=2)(kinds_straight)),
            selu(conv(5, 1, 3, padding=1)(kinds_straight)),
            selu(conv(5, 3, 3, dilation=2, padding=2)(kinds_straight))
        ], 1)
        kinds_straight = flatten(kinds_straight)
        kinds_straight = selu(fc(int(np.prod(get_shape(kinds_straight)[1:])), hdim)(kinds_straight))

        # Process board and hand to detect pairs, trips, quads, full houses
        kinds_ptqf = selu(fc(int(np.prod(get_shape(kinds)[1:])), hdim)(flatten(kinds)))
        kinds_ptqf = selu(fc(hdim, hdim)(kinds_ptqf))

        # Process board and hand to detect flushes
        colors = flatten(selu(conv(2, 1, 1)(colors)))
        colors = selu(fc(4, hdim)(colors))

        # Process the board with FC layers
        flop_alone = selu(fc(52, hdim)(flatten(board[:, 0, :, :])))
        flop_alone = selu(fc(hdim, hdim)(flop_alone))
        turn_alone = selu(fc(52, hdim)(flatten(board[:, 1, :, :])))
        turn_alone = selu(fc(hdim, hdim)(turn_alone))
        river_alone = selu(fc(52, hdim)(flatten(board[:, 2, :, :])))
        river_alone = selu(fc(hdim, hdim)(river_alone))
        board_alone = selu(fc(3 * hdim, hdim)(t.cat([flop_alone, turn_alone, river_alone], -1)))

        # Process board and hand together with FC layers
        h = selu(fc(52, hdim)(flatten(hand)))
        h = selu(fc(hdim, hdim)(h))
        bh = selu(fc(5 * hdim, hdim)(t.cat([h, board_alone, colors, kinds_ptqf, kinds_straight], -1)))
        bh = selu(fc(hdim, hdim)(bh))

        # Predict probabilities of having a given hand + hand strength
        probabilities_of_each_combination_board_only = softmax(fc(hdim, n_actions)(board_alone))
        probabilities_of_each_combination_board_hand = softmax(fc(hdim, n_actions)(bh))
        hand_strength = sigmoid(fc(hdim, 1)(bh))

        # PROCESS THE ACTIONS THAT WERE TAKEN IN THE CURRENT EPISODE
        processed_preflop = selu(fc(5 * 6 * 2, hdim)(flatten(preflop_plays)))
        processed_flop = selu(fc(5 * 6 * 2 + hdim, hdim)(t.cat([flatten(flop_plays), flop_alone], -1)))
        processed_turn = selu(fc(5 * 6 * 2 + hdim, hdim)(t.cat([flatten(turn_plays), turn_alone], -1)))
        processed_river = selu(fc(5 * 6 * 2 + hdim, hdim)(t.cat([flatten(river_plays), river_alone], -1)))
        plays = selu(fc(hdim, hdim)(processed_preflop + processed_flop + processed_turn + processed_river))

        # add pot, dealer, blinds, dealer, stacks
        pbds = selu(fc(5, hdim)(t.cat([pot, stack, opponent_stack, big_blind, dealer], -1)))

        # USE ALL INFORMATION (CARDS/ACTIONS/MISC) TO PREDICT THE Q VALUES
        situation_with_opponent = selu(fc(3 * hdim, hdim)(t.cat([plays, pbds, bh], -1)))
        situation_with_opponent = selu(fc(hdim, hdim)(situation_with_opponent))

        return situation_with_opponent, hand_strength, probabilities_of_each_combination_board_hand, probabilities_of_each_combination_board_only


class QNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, shared_network=None, pi_network=None):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        assert not (shared_network is not None and pi_network is not None), "you should provide either pi_network or shared_network"
        if pi_network is not None:
            self.shared_network = pi_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetwork(n_actions, hidden_dim)

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        hdim = self.hidden_dim
        n_actions = self.n_actions

        situation_with_opponent, hand_strength, probabilities_of_each_combination_board_hand, probabilities_of_each_combination_board_only = self.shared_network.forward(hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                                                                                                                                                                         flop_plays, turn_plays, river_plays)
        q_values = selu(fc(hdim, hdim)(situation_with_opponent))
        q_values = selu(fc(hdim, hdim)(q_values))
        q_values = fc(hdim, n_actions)(q_values)
        return q_values


class PiNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, shared_network=None, q_network=None):
        super(PiNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        assert not (shared_network is not None and q_network is not None), "you should provide either q_network or shared_network"
        if q_network is not None:
            self.shared_network = q_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetwork(n_actions, hidden_dim)

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        hdim = self.hidden_dim
        n_actions = self.n_actions

        situation_with_opponent, hand_strength, probabilities_of_each_combination_board_hand, probabilities_of_each_combination_board_only = self.shared_network.forward(hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                                                                                                                                                                         flop_plays, turn_plays, river_plays)
        q_values = selu(fc(hdim, hdim)(situation_with_opponent))
        q_values = selu(fc(hdim, hdim)(q_values))
        q_values = fc(hdim, n_actions)(q_values)
        return q_values
