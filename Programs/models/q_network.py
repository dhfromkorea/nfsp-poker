import torch as t
from torch.nn import Conv1d as conv, SELU, Linear as fc, Softmax, Sigmoid, AlphaDropout, BatchNorm1d as BN, PReLU, LeakyReLU
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from game.game_utils import bucket_encode_actions, array_to_cards
from game.utils import variable

selu = SELU()
softmax = Softmax()
sigmoid = Sigmoid()
leakyrelu = LeakyReLU()


def get_shape(x):
    try:
        return x.data.cpu().numpy().shape
    except:
        return x.numpy().shape


def flatten(x):
    shape = get_shape(x)
    return x.resize(shape[0], int(np.prod(shape[1:])))


class CardFeaturizer1(t.nn.Module):
    """
    The one i got results with
    SELU + AlphaDropout + smart initialization
    """

    def __init__(self, hdim, n_filters, cuda=False):
        super(CardFeaturizer1, self).__init__()
        self.hdim = hdim
        self.conv1 = conv(2, n_filters, 1)
        self.conv2 = conv(n_filters, n_filters, 5, padding=2)
        self.conv3 = conv(n_filters, n_filters, 3, padding=1)
        self.conv4 = conv(n_filters, n_filters, 3, dilation=2, padding=2)
        self.conv5 = conv(2, n_filters, 1)

        self.fc1 = fc(13 * n_filters * 3, hdim)
        self.fc2 = fc(13 * 2, hdim)
        self.fc3 = fc(hdim, hdim)
        self.fc4 = fc(4 * n_filters, hdim)
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
        # self.fc17 = fc(hdim, 9)
        self.fc18 = fc(hdim, 1)

        for i in range(1, 19):
            if i == 16 or i == 17:
                continue
            fcc = getattr(self, 'fc' + str(i))
            shape = fcc.weight.data.cpu().numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

        for i in range(1, 6):
            convv = getattr(self, 'conv' + str(i))
            shape = convv.weight.data.cpu().numpy().shape
            convv.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[-1] * shape[-2]), shape)).float()

        if cuda:
            # configure the model params on gpu
            self.cuda()

    def forward(self, hand, board):
        dropout = AlphaDropout(.1)
        dropout.training = self.training

        # DETECTING PATTERNS IN THE BOARD AND HAND
        # Aggregate by suit and kind
        color_hand = t.sum(hand, 1)
        color_board = t.sum(t.sum(board, 2), 1)
        kinds_hand = t.sum(hand, -1)
        kinds_board = t.sum(t.sum(board, -1), 1)
        colors = t.cat([color_hand.resize(len(color_hand), 1, 4), color_board.resize(len(color_board), 1, 4)], 1)
        kinds = t.cat([kinds_hand.resize(len(kinds_hand), 1, 13), kinds_board.resize(len(kinds_board), 1, 13)], 1)

        # Process board and hand to detect straights using convolutions with kernel size 5, 3, and 3 with dilation
        kinds_straight = selu(dropout(self.conv1((kinds > 0).float())))
        kinds_straight = t.cat([
            selu(dropout(self.conv2(kinds_straight))),
            selu(dropout(self.conv3(kinds_straight))),
            selu(dropout(self.conv4(kinds_straight)))
        ], 1)
        kinds_straight = flatten(kinds_straight)
        kinds_straight = selu(dropout(self.fc1(kinds_straight)))

        # Process board and hand to detect pairs, trips, quads, full houses
        kinds_ptqf = selu(dropout(self.fc2(flatten(kinds))))
        kinds_ptqf = selu(dropout(self.fc3(kinds_ptqf)))

        # Process board and hand to detect flushes
        colors = flatten(selu(dropout(self.conv5(colors))))
        colors = selu(dropout(self.fc4(colors)))

        # Process the board with FC layers
        flop_alone = selu(dropout(self.fc5(flatten(board[:, 0, :, :]))))
        flop_alone = selu(dropout(self.fc6(flop_alone)))
        turn_alone = selu(dropout(self.fc7(flatten(t.sum(board[:, :2, :, :], 1)))))
        turn_alone = selu(dropout(self.fc8(turn_alone)))
        river_alone = selu(dropout(self.fc9(flatten(t.sum(board[:, :3, :, :], 1)))))
        river_alone = selu(dropout(self.fc10(river_alone)))
        board_alone = selu(dropout(self.fc11(t.cat([flop_alone, turn_alone, river_alone], -1))))

        # Process board and hand together with FC layers
        h = selu(dropout(self.fc12(flatten(hand))))
        h = selu(dropout(self.fc13(h)))
        cards_features = selu(dropout(self.fc14(t.cat([h, board_alone, colors, kinds_ptqf, kinds_straight], -1))))
        cards_features = selu(dropout(self.fc15(cards_features)))

        # Predict probabilities of having a given hand + hand strength
        #         probabilities_of_each_combination = softmax(self.fc17(bh))
        hand_strength = sigmoid(self.fc18(cards_features))
        return hand_strength, cards_features, flop_alone, turn_alone, river_alone


def clip_gradients(nn, bound=10):
    for p in nn.parameters():
        if p.grad is not None:
            p.grad = p.grad * ((bound <= p.grad).float()) * ((bound >= p.grad).float()) + bound * ((p.grad > bound).float()) - bound * ((p.grad < -bound).float())


class SharedNetwork(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, cuda=False):
        super(SharedNetwork, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        hdim = hidden_dim

        self.fc19 = fc(5 * 6 * 2, hdim)
        self.fc20 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc21 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc22 = fc(5 * 6 * 2 + hdim, hdim)
        self.fc23 = fc(hdim, hdim)
        self.fc24 = fc(6, hdim)
        self.fc25 = fc(3 * hdim, hdim)
        self.fc26 = fc(hdim, hdim)

        for i in range(19, 26):
            fcc = getattr(self, 'fc' + str(i))
            shape = fcc.weight.data.cpu().numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

        if cuda:
            self.cuda()

    def forward(self, HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        dropout = AlphaDropout(.1)
        dropout.training = self.training

        # PROCESS THE ACTIONS THAT WERE TAKEN IN THE CURRENT EPISODE
        processed_preflop = selu(dropout(self.fc19(flatten(preflop_plays))))
        processed_flop = selu(dropout(self.fc20(t.cat([flatten(flop_plays), flop_features], -1))))
        processed_turn = selu(dropout(self.fc21(t.cat([flatten(turn_plays), turn_features], -1))))
        processed_river = selu(dropout(self.fc22(t.cat([flatten(river_plays), river_features], -1))))
        plays = selu(dropout(self.fc23(processed_preflop + processed_flop + processed_turn + processed_river)))

        # add pot, dealer, blinds, dealer, stacks
        pbds = selu(dropout(self.fc24(t.cat([pot, stack, opponent_stack, big_blind, dealer, HS], -1))))

        # USE ALL INFORMATION (CARDS/ACTIONS/MISC) TO PREDICT THE Q VALUES
        situation_with_opponent = selu(dropout(self.fc25(t.cat([plays, pbds, cards_features], -1))))
        situation_with_opponent = selu(dropout(self.fc26(situation_with_opponent)))

        return situation_with_opponent


class QNetwork(t.nn.Module):
    def __init__(self,
                 n_actions,
                 hidden_dim,
                 featurizer,
                 game_info,
                 player_id,
                 #neural_network_history,
                 #neural_network_loss,
                 learning_rate,
                 optimizer,
                 grad_clip=None,
                 tensorboard=None,
                 is_target_Q=False,
                 shared_network=None,
                 pi_network=None,
                 cuda=False):
        super(QNetwork, self).__init__()

        self.is_cuda=cuda
        self.n_actions = n_actions
        self.featurizer = featurizer
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

        # SHARE WEIGHTS
        for i in range(19, 27):
            setattr(self, 'fc' + str(i), getattr(self.shared_network, 'fc' + str(i)))
        # LAST PERSONAL LAYERS
        self.fc27 = fc(hdim, hdim)
        self.fc28 = fc(hdim, n_actions)

        # INIT WEIGHTS SELU
        for i in range(27, 29):
            fcc = getattr(self, 'fc' + str(i))
            shape = fcc.weight.data.cpu().numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()
        # optimizer
        self.grad_clip = grad_clip

        # exclude frozen weights (requires_grad=False) for featurizer
        params = filter(lambda p: p.requires_grad, self.parameters())
        if optimizer == 'adam':
            self.optim = optim.Adam(params, lr=learning_rate)
        elif optimizer == 'sgd':
            # 2016 paper
            self.optim = optim.SGD(params, lr=learning_rate)
        else:
            raise Exception('unsupported optimizer: use adam or sgd (lower cased)')

        # to initialize network on gpu
        if cuda:
            self.cuda()

        # for saving neural network history data
        self.game_info = game_info
        self.player_id = player_id  # know the owner of the network
        #self.neural_network_history = neural_network_history
        #self.neural_network_loss = neural_network_loss
        self.tensorboard = tensorboard

        # hyperparams for loss
        self.beta = 0.1

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                flop_plays, turn_plays, river_plays, for_play=False):
        dropout = AlphaDropout(.1)
        dropout.training = self.training

        HS, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)

        if for_play:
            # if forward was used during play (not training)
            if self.tensorboard is not None:
                hand_strength = float(HS.data.cpu().numpy().flatten()[0])
                self.tensorboard.add_scalar_value('p{}_hand_strength_q(play)'.format(self.player_id + 1),
                                                  hand_strength, time.time())

        # HS, proba_combinations, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)
        situation_with_opponent = self.shared_network.forward(HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays)
        q_values = selu(dropout(self.fc27(situation_with_opponent)))
        q_values = self.fc28(dropout(q_values))

        # for saving neural network history data
        #episode_id = self.game_info['#episodes']
        #if not episode_id in self.neural_network_history:
        #    self.neural_network_history[episode_id] = {}
        #self.neural_network_history[episode_id][self.player_id] = {}
        #self.neural_network_history[episode_id][self.player_id]['q'] = q_values.data.cpu().numpy()

        return q_values

    def learn(self, states, actions, Q_targets, imp_weights):
        self.optim.zero_grad()
        # TODO: support batch forward?
        # not sure if it's supported as it's written now
        all_Q_preds = self.forward(*states)

        actions_ = (bucket_encode_actions(actions, cuda=self.is_cuda) + 1).long()
        Q_preds = t.cat([all_Q_preds[i, aa] for i, aa in enumerate(actions_.data)]).squeeze()  # Q(s,a)

        #loss, mse, td_deltas = self.compute_loss(Q_preds, Q_targets, imp_weights)
        loss, mse, entropy, td_deltas = self.compute_loss(Q_preds, Q_targets, imp_weights)

        # log loss history data
        #if not 'q' in self.neural_network_loss[self.player_id]:
        #    self.neural_network_loss[self.player_id]['q'] = []
        mse = mse.data.cpu().numpy()[0]
        # todo: refactor the hard coded name
        if self.tensorboard is not None:
            self.tensorboard.add_scalar_value('p{}_q_mse_loss'.format(self.player_id + 1), float(mse), time.time())
            self.tensorboard.add_scalar_value('p{}_q_entropy_loss'.format(self.player_id + 1), float(-entropy), time.time())
        #self.neural_network_loss[self.player_id]['q'].append(mse)

        loss.backward()
        # @debug @todo
        if self.grad_clip is not None:
            t.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)

        # update weights
        self.optim.step()
        return td_deltas

    def compute_loss(self, pred, target, imp_weights):
        '''
        compute weighted mse loss
        loss for each sample is scaled by imp_weight
        we need this to account for bias in replay sampling

        added entropy to term to increase diversity and exploration
        beta is a hyperparameter
        '''
        td_deltas = pred - target
        mse = t.mean(imp_weights * td_deltas.pow(2))
        #mse = t.mean(td_deltas.pow(2))

        # @experimental: entropy term
        episode_id = self.game_info['#episodes']
        self.beta = np.max([self.beta/np.power(np.max([1, episode_id]), 1/4), 0.01])
        beta_var = variable(self.beta, cuda=self.is_cuda)
        sm = Softmax(dim=0)
        probs = sm(pred.unsqueeze(dim=1))
        m = variable(np.array([1e-6]), cuda=self.is_cuda)
        entropy = -t.sum(probs * t.log(t.max(probs, m)))

        loss = mse - beta_var * entropy
        return loss, mse, entropy, td_deltas
        #loss = mse
        #return loss, mse, td_deltas

class PiNetwork(t.nn.Module):
    def __init__(self,
                 n_actions,
                 hidden_dim,
                 featurizer,
                 game_info,
                 player_id,
                 #neural_network_history,
                 #neural_network_loss,
                 learning_rate,
                 optimizer,
                 grad_clip=None,
                 tensorboard=None,
                 shared_network=None,
                 q_network=None,
                 cuda=False):
        super(PiNetwork, self).__init__()
        # cuda is a reserved property for t.nn.Module
        self.is_cuda = cuda
        self.n_actions = n_actions
        self.featurizer = featurizer
        self.hidden_dim = hidden_dim
        hdim = self.hidden_dim

        # SHARE WEIGHTS
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

        # LAST PERSONAL LAYERS
        self.fc27 = fc(hdim, hdim)
        self.fc28 = fc(hdim, n_actions)

        # INIT WEIGHTS SELU
        for i in range(27, 29):
            fcc = getattr(self, 'fc' + str(i))
            shape = fcc.weight.data.cpu().numpy().shape
            fcc.weight.data = t.from_numpy(np.random.normal(0, 1 / np.sqrt(shape[0]), shape)).float()

        self.grad_clip = grad_clip

        params = filter(lambda p: p.requires_grad, self.parameters())
        if optimizer == 'adam':
            self.optim = optim.Adam(params, lr=learning_rate)
        elif optimizer == 'sgd':
            # 2016 paper
            self.optim = optim.SGD(params, lr=learning_rate)
        else:
            raise Exception('unsupported optimizer: use adam or sgd (lower cased)')

        if cuda:
            self.cuda()

        # for saving neural network history data
        self.game_info = game_info
        self.player_id = player_id  # know the owner of the network
        #self.neural_network_history = neural_network_history
        #self.neural_network_loss = neural_network_loss
        self.tensorboard = tensorboard

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays,
                flop_plays, turn_plays, river_plays, for_play=False):
        dropout = AlphaDropout(.1)
        dropout.training = self.training

        HS, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)

        if for_play:
            # if forward was used during play (not training)
            if self.tensorboard is not None:
                hand_strength = float(HS.data.cpu().numpy().flatten()[0])
                self.tensorboard.add_scalar_value('p{}_hand_strength_pi(play)'.format(self.player_id + 1),
                                                  hand_strength, time.time())
        situation_with_opponent = self.shared_network.forward(HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays)

        pi_values = selu(dropout(self.fc27(situation_with_opponent)))
        softmax = Softmax(dim=0)
        pi_values = softmax(dropout(self.fc28(pi_values)))

        # for saving neural network history data
        #episode_id = self.game_info['#episodes']
        #if not episode_id in self.neural_network_history:
        #    self.neural_network_history[episode_id] = {}
        #self.neural_network_history[episode_id][self.player_id] = {}
        #self.neural_network_history[episode_id][self.player_id]['pi'] = pi_values.data.cpu().numpy()

        return pi_values

    def learn(self, states, actions):
        """
        From Torch site
         loss = nn.CrossEntropyLoss()
         input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
         target = autograd.Variable(torch.LongTensor(3).random_(5))
         output = loss(input, target)
         output.backward()
        """
        self.optim.zero_grad()
        pi_preds = self.forward(*states).squeeze()
        criterion = nn.CrossEntropyLoss()
        one_hot_actions = bucket_encode_actions(actions, cuda=self.is_cuda)
        loss = criterion(pi_preds, (1+one_hot_actions).long())

        # log loss history data
        #if not 'pi' in self.neural_network_loss[self.player_id]:
        #    self.neural_network_loss[self.player_id]['pi'] = []
        raw_loss = loss.data.cpu().numpy()[0]
        #self.neural_network_loss[self.player_id]['pi'].append(raw_loss)
        if self.tensorboard is not None:
            self.tensorboard.add_scalar_value('p{}_pi_ce_loss'.format(self.player_id + 1), float(raw_loss), time.time())

        loss.backward()
        # @debug @hack
        if self.grad_clip is not None:
            t.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)
        self.optim.step()

        return loss


class SharedNetworkBN(t.nn.Module):
    def __init__(self, n_actions, hidden_dim, cuda=False):
        super(SharedNetworkBN, self).__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        hdim = hidden_dim

        self.fc19 = fc(5 * 6 * 2, hdim)
        self.bn19 = BN(hdim, momentum=.99)
        self.fc20 = fc(5 * 6 * 2 + hdim, hdim)
        self.bn20 = BN(hdim, momentum=.99)
        self.fc21 = fc(5 * 6 * 2 + hdim, hdim)
        self.bn21 = BN(hdim, momentum=.99)
        self.fc22 = fc(5 * 6 * 2 + hdim, hdim)
        self.bn22 = BN(hdim, momentum=.99)
        self.fc23 = fc(hdim, hdim)
        self.bn23 = BN(hdim, momentum=.99)
        self.fc24 = fc(6, hdim)
        self.bn24 = BN(hdim, momentum=.99)
        self.fc25 = fc(3 * hdim, hdim)
        self.bn25 = BN(hdim, momentum=.99)
        self.fc26 = fc(hdim, hdim)
        self.bn26 = BN(hdim, momentum=.99)

        if cuda:
            self.cuda()

    def forward(self, HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        # PROCESS THE ACTIONS THAT WERE TAKEN IN THE CURRENT EPISODE
        processed_preflop = leakyrelu(self.bn19(self.fc19(flatten(preflop_plays))))
        processed_flop = leakyrelu(self.bn20(self.fc20(t.cat([flatten(flop_plays), flop_features], -1))))
        processed_turn = leakyrelu(self.bn21(self.fc21(t.cat([flatten(turn_plays), turn_features], -1))))
        processed_river = leakyrelu(self.bn22(self.fc22(t.cat([flatten(river_plays), river_features], -1))))
        plays = leakyrelu(self.bn23(self.fc23(processed_preflop + processed_flop + processed_turn + processed_river)))

        # add pot, dealer, blinds, dealer, stacks
        pbds = leakyrelu(self.bn24(self.fc24(t.cat([pot, stack, opponent_stack, big_blind, dealer, HS], -1))))

        # USE ALL INFORMATION (CARDS/ACTIONS/MISC) TO PREDICT THE Q VALUES
        situation_with_opponent = leakyrelu(self.bn25(self.fc25(t.cat([plays, pbds, cards_features], -1))))
        situation_with_opponent = leakyrelu(self.bn26(self.fc26(situation_with_opponent)))

        return situation_with_opponent


class QNetworkBN(t.nn.Module):
    def __init__(self,
                 n_actions,
                 hidden_dim,
                 featurizer,
                 game_info,
                 player_id,
                 #neural_network_history,
                 #neural_network_loss,
                 learning_rate,
                 optimizer,
                 grad_clip=None,
                 tensorboard=None,
                 is_target_Q=False,
                 shared_network=None,
                 pi_network=None,
                 cuda=False):

        super(QNetworkBN, self).__init__()

        self.is_cuda = cuda
        self.n_actions = n_actions
        self.featurizer = featurizer
        self.hidden_dim = hidden_dim
        hdim = self.hidden_dim

        assert not (shared_network is not None and pi_network is not None), "you should provide either pi_network or shared_network"
        if pi_network is not None:
            self.shared_network = pi_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetworkBN(n_actions, hidden_dim)

        # SHARE WEIGHTS
        for i in range(19, 27):
            setattr(self, 'bn' + str(i), getattr(self.shared_network, 'bn' + str(i)))
            setattr(self, 'fc' + str(i), getattr(self.shared_network, 'fc' + str(i)))
        # LAST PERSONAL LAYERS
        self.fc27 = fc(hdim, hdim)
        self.bn27 = BN(hdim, momentum=.99)
        self.fc28 = fc(hdim, n_actions)
        self.bn28 = BN(n_actions, momentum=.99)

        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.parameters(), lr=learning_rate)

        # to initialize network on gpu
        if cuda:
            self.cuda()

        # for saving neural network history data
        self.game_info = game_info
        self.player_id = player_id  # know the owner of the network
        #self.neural_network_history = neural_network_history
        #self.neural_network_loss = neural_network_loss
        self.tensorboard = tensorboard

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        HS, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)
        # HS, proba_combinations, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)

        if self.tensorboard is not None:
            self.tensorboard.add_scalar_value('p{}_hand_strength'.format(self.player_id + 1), float(HS.data.cpu().numpy().flatten()[0]), time.time())
        situation_with_opponent = self.shared_network.forward(HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays)
        q_values = leakyrelu(self.bn27(self.fc27(situation_with_opponent)))
        q_values = self.bn28(self.fc28(q_values))

        # for saving neural network history data
        #episode_id = self.game_info['#episodes']
        #if not episode_id in self.neural_network_history:
        #    self.neural_network_history[episode_id] = {}
        #self.neural_network_history[episode_id][self.player_id] = {}
        #self.neural_network_history[episode_id][self.player_id]['q'] = q_values.data.cpu().numpy()

        return q_values

    def learn(self, states, actions, Q_targets, imp_weights):
        self.optim.zero_grad()
        all_Q_preds = self.forward(*states)
        actions_ = (bucket_encode_actions(actions, cuda=self.is_cuda) + 1).long()
        Q_preds = t.cat([all_Q_preds[i, aa] for i, aa in enumerate(actions_.data)]).squeeze()  # Q(s,a)
        loss, td_deltas = self.compute_loss(Q_preds, Q_targets, imp_weights)

        # log loss history data
        #if not 'q' in self.neural_network_loss[self.player_id]:
        #    self.neural_network_loss[self.player_id]['q'] = []
        #raw_loss = loss.data.cpu().numpy()[0]
        #self.neural_network_loss[self.player_id]['q'].append(raw_loss)
        # todo: refactor the hard coded name
        if self.tensorboard is not None:
            self.tensorboard.add_scalar_value('p{}_q_mse_loss'.format(self.player_id + 1), float(raw_loss), time.time())

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
        mse = t.mean(imp_weights * td_deltas.pow(2))
        return mse, td_deltas


class PiNetworkBN(t.nn.Module):
    def __init__(self,
                 n_actions,
                 hidden_dim,
                 featurizer,
                 game_info,
                 player_id,
                 #neural_network_history,
                 #neural_network_loss,
                 learning_rate,
                 optimizer,
                 tensorboard=None,
                 grad_clip=None,
                 shared_network=None,
                 q_network=None,
                 cuda=False):
        super(PiNetworkBN, self).__init__()
        self.is_cuda = cuda
        self.n_actions = n_actions
        self.featurizer = featurizer
        self.hidden_dim = hidden_dim
        hdim = self.hidden_dim

        # SHARE WEIGHTS
        assert not (shared_network is not None and q_network is not None), "you should provide either q_network or shared_network"
        if q_network is not None:
            self.shared_network = q_network.shared_network
        else:
            if shared_network is not None:
                self.shared_network = shared_network
            else:
                self.shared_network = SharedNetworkBN(n_actions, hidden_dim)
        for i in range(19, 27):
            setattr(self, 'fc' + str(i), getattr(self.shared_network, 'fc' + str(i)))
            setattr(self, 'bn' + str(i), getattr(self.shared_network, 'bn' + str(i)))

        # LAST PERSONAL LAYERS
        self.fc27 = fc(hdim, hdim)
        self.bn27 = BN(hdim, momentum=.99)
        self.fc28 = fc(hdim, n_actions)
        self.bn28 = BN(n_actions, momentum=.99)

        self.optim = optim.Adam(self.parameters(), lr=learning_rate)
        if cuda:
            self.cuda()

        # for saving neural network history data
        self.game_info = game_info
        self.player_id = player_id  # know the owner of the network
        #self.neural_network_history = neural_network_history
        #self.neural_network_loss = neural_network_loss
        self.tensorboard = tensorboard

    def forward(self, hand, board, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays):
        HS, flop_features, turn_features, river_features, cards_features = self.featurizer.forward(hand, board)
        situation_with_opponent = self.shared_network.forward(HS, cards_features, flop_features, turn_features, river_features, pot, stack, opponent_stack, big_blind, dealer, preflop_plays, flop_plays, turn_plays, river_plays)

        pi_values = leakyrelu(self.bn27(self.fc27(situation_with_opponent)))
        pi_values = softmax(self.bn28(self.fc28(pi_values)))

        # for saving neural network history data
        #episode_id = self.game_info['#episodes']
        #if not episode_id in self.neural_network_history:
        #    self.neural_network_history[episode_id] = {}
        #self.neural_network_history[episode_id][self.player_id] = {}
        #self.neural_network_history[episode_id][self.player_id]['pi'] = pi_values.data.cpu().numpy()

        return pi_values

    def learn(self, states, actions):
        """
        From Torch site
         loss = nn.CrossEntropyLoss()
         input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
         target = autograd.Variable(torch.LongTensor(3).random_(5))
         output = loss(input, target)
         output.backward()
        """
        self.optim.zero_grad()
        pi_preds = self.forward(*states).squeeze()
        criterion = nn.CrossEntropyLoss()
        one_hot_actions = bucket_encode_actions(actions, cuda=self.is_cuda)
        loss = criterion(pi_preds, (1 + one_hot_actions).long())

        # log loss history data
        #if not 'pi' in self.neural_network_loss[self.player_id]:
        #    self.neural_network_loss[self.player_id]['pi'] = []
        raw_loss = loss.data.cpu().numpy()[0]
        #self.neural_network_loss[self.player_id]['pi'].append(raw_loss)
        if self.tensorboard is not None:
            self.tensorboard.add_scalar_value('p{}_pi_ce_loss'.format(self.player_id + 1), float(raw_loss), time.time())

        loss.backward()
        self.optim.step()
        return loss


