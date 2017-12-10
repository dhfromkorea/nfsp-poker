from odds.evaluation import evaluate_hand
from models.q_network import QNetwork, QNetworkBN, PiNetwork, PiNetworkBN

from players.strategies import strategy_RL, strategy_random, strategy_mirror, StrategyNFSP
from players.player import Player, NeuralFictitiousPlayer

from game.utils import get_last_round
from game.game_utils import Deck, set_dealer, blinds, deal, agreement, actions_to_array, array_to_cards, action_to_array, cards_to_array
from game.config import BLINDS
from game.state import build_state, create_state_variable_batch

from models.featurizer import FeaturizerManager
import time
import numpy as np
import torch as t
import pickle
import models.q_network
import os.path
from torch.autograd import Variable

# paths
SAVED_FEATURIZER_PATH = 'data/hand_eval/best_models/card_featurizer1.50-10.model.pytorch'
GAME_SCORE_HISTORY_PATH = 'data/game_score_history/'
PLAY_HISTORY_PATH = 'data/play_history/'
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/'
NEURAL_NETWORK_LOSS_PATH = 'data/neural_network_history/loss/'
EXPERIMENT_PATH = 'data/tensorboard/'
MODEL_SAVEPATH = 'data/neural_network_history/models/'

# define game constants here
INITIAL_MONEY = 100 * BLINDS[0]
NUM_ROUNDS = 4  # pre, flop, turn, river
NUM_HIDDEN_LAYERS = 50
NUM_ACTIONS = 16

strategy_function_map = {'random': strategy_random, 'mirror': strategy_mirror,
                         'RL': strategy_RL}

baseline_strategies = ['mirror', 'random']
qnetwork_strategies = ['RL']
NFSP_strategies = ['NFSP']
allowed_strategies = baseline_strategies + qnetwork_strategies + NFSP_strategies
p_names = ['SB', 'DH']


class Simulator:
    """
    right now, RL algo, Simulator, ER are all tightly coupled
    this is bad.
    TODO: decouple them

    right now, players are not using networks to choose actions
    TODO: couple players with networks
    """

    def __init__(self,
                 # these should be explicitly passed
                 log_freq,
                 learn_start,
                 eta_p1,
                 eta_p2,
                 eps,
                 gamma,
                 learning_rate_rl,
                 learning_rate_sl,
                 learning_freq,
                 target_Q_update_freq,
                 use_batch_norm,
                 optimizer,
                 experiment_id,
                 p1_strategy,
                 p2_strategy,
                 # use default values
                 featurizer_path=SAVED_FEATURIZER_PATH,
                 game_score_history_path=GAME_SCORE_HISTORY_PATH,
                 play_history_path=PLAY_HISTORY_PATH,
                 neural_network_history_path=NEURAL_NETWORK_HISTORY_PATH,
                 neural_network_loss_path=NEURAL_NETWORK_LOSS_PATH,
                 memory_rl_config={},
                 memory_sl_config={},
                 grad_clip=None,
                 verbose=False,
                 cuda=False,
                 tensorboard=None):
        # define msc.
        self.verbose = verbose
        self.cuda = cuda
        self.log_freq = log_freq
        self.eps = eps
        self.gamma = gamma
        self.learning_rate_rl = learning_rate_rl
        self.learning_rate_sl = learning_rate_sl
        self.learning_freq = learning_freq
        self.target_Q_update_freq = target_Q_update_freq
        self.learn_start = learn_start
        self.use_batch_norm = use_batch_norm
        self.strategy_p1 = p1_strategy
        self.strategy_p2 = p2_strategy

        # NFSP-speicfic network hyperparams
        self.etas = {0: eta_p1, 1: eta_p2}
        self.memory_rl_config = memory_rl_config
        self.memory_sl_config = memory_sl_config

        # historical data
        # 1. game score
        self.games = {'n': 0, '#episodes': 0, 'winnings': {}}
        self.game_score_history_path = game_score_history_path
        # 2. play
        self.play_history = {}
        self.play_history_path = play_history_path
        # 3. neural network
        self.neural_network_history_path = neural_network_history_path
        self.neural_network_history = {}
        self.neural_network_loss_path = neural_network_loss_path
        self.neural_network_loss = {0: {'q': [], 'pi': []},
                                    1: {'q': []}, 'pi': []}
        # 4. tensorboard
        self.tensorboard = tensorboard
        self.experiment_id = experiment_id

        # debugging utility variables

        self.last_game_start_time = None
        self.last_episode_start_time = None
        self.game_play_speeds = []
        self.episode_play_speeds = []

        # define game-level game states here
        self.new_game = True
        self.global_step = 0

        # define players
        featurizer = FeaturizerManager.load_model(featurizer_path, cuda=cuda)

        if self.use_batch_norm:
            # ugly switch that does not scale
            # for simplicy let's keep this way
            # as we may still want to keep testing Q without BN
            Q = QNetworkBN
            Pi = PiNetworkBN
        else:
            Q = QNetwork
            Pi = PiNetwork

        Q0 = Q(n_actions=NUM_ACTIONS,
               hidden_dim=NUM_HIDDEN_LAYERS,
               featurizer=featurizer,
               game_info=self.games,  # bad, but simple (@hack)
               learning_rate=learning_rate_rl,
               player_id=0,
               optimizer=optimizer,
               grad_clip=grad_clip,
               neural_network_history=self.neural_network_history,
               neural_network_loss=self.neural_network_loss,
               tensorboard=self.tensorboard,
               cuda=cuda)
        Q1 = Q(n_actions=NUM_ACTIONS,
               hidden_dim=NUM_HIDDEN_LAYERS,
               featurizer=featurizer,
               game_info=self.games,  # bad, but simple (@hack)
               player_id=1,
               optimizer=optimizer,
               grad_clip=grad_clip,
               learning_rate=learning_rate_rl,
               neural_network_history=self.neural_network_history,
               neural_network_loss=self.neural_network_loss,
               tensorboard=self.tensorboard,
               cuda=cuda)
        Pi0 = Pi(n_actions=NUM_ACTIONS,
                 hidden_dim=NUM_HIDDEN_LAYERS,
                 featurizer=featurizer,
                 q_network=Q0,  # to share weights
                 game_info=self.games,  # bad, but simple (@hack)
                 player_id=0,
                 optimizer=optimizer,
                 grad_clip=grad_clip,
                 learning_rate=learning_rate_sl,
                 neural_network_history=self.neural_network_history,
                 neural_network_loss=self.neural_network_loss,
                 tensorboard=self.tensorboard,
                 cuda=cuda)
        Pi1 = Pi(n_actions=NUM_ACTIONS,
                 hidden_dim=NUM_HIDDEN_LAYERS,
                 featurizer=featurizer,
                 q_network=Q1,  # to share weights with Q1
                 game_info=self.games,  # bad, but simple (@hack)
                 player_id=1,
                 optimizer=optimizer,
                 grad_clip=grad_clip,
                 learning_rate=learning_rate_sl,
                 neural_network_history=self.neural_network_history,
                 neural_network_loss=self.neural_network_loss,
                 tensorboard=self.tensorboard,
                 cuda=cuda)
        Q_networks = {0: Q0, 1: Q1}
        Pi_networks = {0: Pi0, 1: Pi1}
        self.players = self._generate_player_instances(p1_strategy, p2_strategy,
                                                       Q_networks, Pi_networks,
                                                       learn_start, verbose)

        # define episode-level game states here
        self.deck = Deck()
        self.dealer = set_dealer(self.players)
        self.board = []
        self.experiences = [None] * len(self.players)

    def start(self, term_game_count=-1, return_results=False):
        while True:
            if term_game_count > 0 and self.games['n'] > term_game_count:
                break
            if self.new_game:
                self._prepare_new_game()
            safe_to_start = self._prepare_new_episode()
            if not safe_to_start:
                raise Exception('corrupt game')
            self._start_episode()

        if return_results:
            # return self.games.winnings
            return self.games['winnings']

    def _generate_player_instances(self, p1_strategy, p2_strategy,
                                   Q_networks, Pi_networks, learn_start, verbose):
        players = []
        p_id = 0
        # Strategies that do not require Q
        for strategy in [p1_strategy, p2_strategy]:
            if strategy not in allowed_strategies:
                raise ValueError("Not a valid strategy")
            elif strategy in baseline_strategies:
                players.append(Player(p_id, strategy_function_map[strategy], INITIAL_MONEY, p_names[p_id], verbose=verbose))
                p_id += 1
            elif strategy in qnetwork_strategies:
                players.append(Player(p_id, strategy_function_map[strategy](Q_networks[p_id], True), INITIAL_MONEY, p_names[p_id], verbose=verbose))
                p_id += 1
            elif strategy in NFSP_strategies:
                strategy = StrategyNFSP(Q=Q_networks[p_id],
                                        pi=Pi_networks[p_id],
                                        eta=self.etas[p_id],
                                        eps=self.eps,
                                        cuda=self.cuda)

                nfp = NeuralFictitiousPlayer(pid=p_id,
                                             strategy=strategy,
                                             stack=INITIAL_MONEY,
                                             name=p_names[p_id],
                                             gamma=self.gamma,
                                             learning_freq=self.learning_freq,
                                             target_Q_update_freq=self.target_Q_update_freq,
                                             memory_rl_config=self.memory_rl_config,
                                             memory_sl_config=self.memory_sl_config,
                                             learn_start=self.learn_start,
                                             verbose=self.verbose,
                                             tensorboard=self.tensorboard,
                                             cuda=self.cuda)
                players.append(nfp)
                p_id += 1
        return players

    def _prepare_new_game(self):
        '''
        if new game -> initialize
        '''
        # at the beginning of a whole new game (one of the player lost or it is the first), all start with the same amounts of money again
        self.games['n'] += 1
        if self.verbose:
            self._log_game_play_speed()
        # buffer_length = buffer_rl.size
        buffer_length = str(tuple([p.memory_rl._buffer.record_size for p in self.players if p.player_type == 'nfsp'] + [p.memory_sl._buffer.record_size for p in self.players if p.player_type == 'nfsp']))

        if self.verbose:
            t0 = time.time()
            print('####################'
                  'New game (%s) starts.\n'
                  'Players get cash\n'
                  'Last game lasted %.1f\n'
                  'Memory contains %s experiences\n'
                  '####################' % (str(self.games['n']), time.time() - t0, buffer_length))
        self.players[0].cash(INITIAL_MONEY)
        self.players[1].cash(INITIAL_MONEY)

    def _prepare_new_episode(self):
        self.games['#episodes'] += 1
        # PAY BLINDS
        self.pot = blinds(self.players, self.verbose)
        if self.verbose:
            print('pot: ' + str(self.pot))

        # SHUFFLE DECK AND CLEAR BOARD
        self.deck.populate()
        self.deck.shuffle()
        self.board = []
        self.players[0].is_all_in = False
        self.players[1].is_all_in = False

        # MONITOR ACTIONS
        # -1 is for the blinds
        self.actions = {b_round: {player: [] for player in range(2)} for b_round in range(-1, 4)}
        self.actions[-1][self.players[0].id] = self.players[0].side_pot
        self.actions[-1][self.players[1].id] = self.players[1].side_pot

        # dramatic events monitoring (fold, all-in)
        self.fold_occured = False
        self.null = 0  # once one or two all-ins occurred, actions are null. Count them to stop the loop
        self.all_in = 0  # 0, 1 or 2. If 2, the one of the player is all-in and the other is either all-in or called. In that case, things should be treated differently
        if self.players[0].stack == 0:  # in this case the blind puts the player all-in
            self.all_in = 2
            self.players[0].is_all_in = True
        if self.players[1].stack == 0:
            self.all_in = 2
            self.players[1].is_all_in = True

        # return True if safe to start
        return True

    def _start_episode(self):
        '''
        an episode is a sequence of rounds
        '''
        self._play_rounds()
        # store terminal experience from the previous step
        # we want to modify the reward of the previous step
        # based on the calculation below

        # initialize experiences for player 1 and 2
        self.experiences[0] = self._make_new_exp()
        self.experiences[1] = self._make_new_exp()
        # WINNER GETS THE MONEY.
        # WATCH OUT! TIES CAN OCCUR. IN THAT CASE, SPLIT
        self.split = False
        showdown_occured = 0
        if not self.fold_occured:
            self._showdown()
            showdown_occured = 1

        # show down happened, we send data
        if self.tensorboard is not None:
            self._send_showdown_data_to_tensorboard(showdown_occured)

        # these two clauses update the `s`, `is_terminal` and `final_reward` of self.experiences[0/1]
        if self.split:
            self._handle_split()
        else:
            self._handle_no_split()
        # store final experience
        # KEEP TRACK OF TRANSITIONS
        last_round = get_last_round(self.actions, 0)
        if last_round > -1:  # in that case you didnt play and was allin because of the blinds
            if self.players[0].player_type == 'nfsp':
                self.players[0].remember(self.experiences[0])

        last_round = get_last_round(self.actions, 1)
        if last_round > -1:  # in that case you didnt play and was allin because of the blinds
            if len(self.actions[last_round][1]) > 0:
                if self.players[1].player_type == 'nfsp':
                    self.players[1].remember(self.experiences[1])

        try:
            self.update_play_history_with_final_rewards()
            if self.tensorboard is not None:
                self._send_data_to_tensorboard()
        except KeyError:
            raise KeyError

        for p in self.players:
            if p.player_type == 'nfsp':
                p.learn(self.global_step, self.games['#episodes'])

        if self.verbose:
            self._log_episode_play_speed()
        self._reset_variables()
        # TODO: remove this! temp variable
        self.is_new_game = True
        # IS IT THE END OF THE GAME ? (bankruptcy)
        self._set_new_game()

    def _play_rounds(self):
        # EPISODE ACTUALLY STARTS HERE
        for r in range(NUM_ROUNDS):
            if r > 0:
                self.new_game = False
            self.b_round = r
            # DIFFERENTIATE THE CASES WHERE PLAYERS ARE ALL-IN FROM THE ONES WHERE NONE OF THEM IS
            if self.all_in < 2:
                # DEAL CARDS
                deal(self.deck, self.players, self.board, self.b_round, verbose=self.verbose)
                self.agreed = False  # True when the max bet has been called by everybody

                # PLAY
                if self.b_round != 0:
                    self.to_play = 1 - self.dealer
                else:
                    self.to_play = self.dealer

                while not self.agreed:
                    self._play_round()

                self._update_side_pot()

                # POTENTIALLY STOP THE EPISODE IF FOLD OCCURRED
                if self.fold_occured:
                    # end episode
                    break
            else:
                # DEAL REMAINING CARS
                for r in range(self.b_round, 4):
                    deal(self.deck, self.players, self.board, r, verbose=self.verbose)

                # END THE EPISODE
                self._update_side_pot()
                break

    def _play_round(self):
        """
        Note that here `round` does not designate a round of betting (preflop, flop, turn or river), but rather a
        unique transition/action/decision point (as you prefer to call it)
        """
        self.global_step += 1

        # CHOOSE AN ACTION
        self.player = self.players[self.to_play]
        assert self.player.stack >= 0, self.player.stack
        assert not ((self.player.stack == 0) and not self.player.is_all_in), (self.player, self.player.is_all_in, self.actions)

        self.action = self.player.play(self.board, self.pot,
                                       self.actions, self.b_round,
                                       self.players[1 - self.to_play].stack,
                                       self.players[1 - self.to_play].side_pot,
                                       BLINDS, self.games['#episodes'])

        if self.action.type == 'null':
            # this happens when a player is all-in. In this case it can no longer play
            # if the blinds put the player all-in
            self.global_step -= 1  # no action is taken, so no transition should be stored, so global step should be the as before the player turn
            self.to_play = 1 - self.to_play
            self.null += 1
            if self.null >= 2:
                self.agreed = True
                # end the round with agreement
                return
            # go to the next agreement step
            return
        self.player.has_played = True

        # we save play history data here
        # we're computing exp tuple twice here (ugly..)
        self.play_history[self.global_step] = {}
        ph = self.play_history[self.global_step]
        exp = self.experiences[self.player.id] = self.make_experience(self.player, self.action, self.new_game, self.board, self.pot, self.dealer, self.actions, BLINDS[1], self.global_step, self.b_round)
        ph[self.player.id] = {'s': (array_to_cards(exp['s'][0]), array_to_cards(exp['s'][1])), 'a': exp['a'], 'r': 0}
        ph['game'] = {
            'episode_index': self.games['#episodes'],
            'to_play': self.to_play,
            'b_round': self.b_round
        }

        # RL : STORE EXPERIENCES IN MEMORY.
        # Just for the NSFP agents. Note that it is saved BEFORE that the chosen action updates the state
        if self.player.player_type == 'nfsp':
            self.experiences[self.player.id] = self.make_experience(self.player, self.action, self.new_game, self.board,
                                                                    self.pot, self.dealer, self.actions, BLINDS[1],
                                                                    self.global_step, self.b_round)
            self.player.remember(self.experiences[self.player.id])

        # UPDATE STATE DEPENDING ON THE ACTION YOU TOOK
        # Sanity check: it should be impossible to bet/call/all in with value 0
        if self.action.type in {'all in', 'bet', 'call'}:
            assert self.action.value > 0

        if self.action.type == 'call':
            # if you called, the value of your call should make the side pots match
            self._is_call_legit()

        # update pot, side pots, and stacks
        value = self.action.total
        self._update_pot_and_stacks(value)

        # DRAMATIC ACTION MONITORING
        self._handle_dramatic_action()

        # DECIDE WHETHER IT IS THE END OF THE BETTING ROUND OR NOT, AND GIVE LET THE NEXT PLAYER PLAY
        # if fold, it is the end
        if self.action.type == 'fold':
            self._handle_fold()
            self.agreed = True
            return

        # otherwise, check if players came to an agreement
        self.agreed = agreement(self.actions, self.b_round) or (self.all_in == 2)
        self.to_play = 1 - self.to_play

    def update_winnings(self):
        self.games['winnings'][self.games['n']] = {self.players[0].stack, self.players[1].stack}

    def save_history_results(self):
        if self.games['n'] % self.log_freq == 0:
            # we save all history data here. Clear the dicts after saving them.
            cur_t = time.strftime('%y%m%d_%H%M%S', time.gmtime())
            # fix typo em
            exp_id = self.experiment_id
            gh_path = '{}{}_{}.p'.format(self.game_score_history_path, cur_t, exp_id)
            ph_path = '{}{}_{}.p'.format(self.play_history_path, cur_t, exp_id)
            nnh_path = '{}{}_{}.p'.format(self.neural_network_history_path, cur_t, exp_id)
            nnl_path = '{}{}_{}.p'.format(self.neural_network_loss_path, cur_t, exp_id)
            self._save_results(self.games['winnings'], gh_path)
            self._save_results(self.play_history, ph_path)
            self._save_results(self.neural_network_history, nnh_path)
            self._save_results(self.neural_network_loss, nnl_path)

            # save the trained model
            for p in self.players:
                if p.player_type == 'nfsp':
                    q_model = p.strategy._Q.state_dict()
                    pi_model = p.strategy._pi.state_dict()
                    t.save(q_model, '{}{}_{}_q_{}.pt'.format(MODEL_SAVEPATH, cur_t, exp_id, p.id + 1))
                    t.save(pi_model, '{}{}_{}_pi_{}.pt'.format(MODEL_SAVEPATH, cur_t, exp_id, p.id + 1))

            if self.tensorboard is not None:
                self.tensorboard.to_zip('{}{}_{}'.format(EXPERIMENT_PATH, cur_t, exp_id))
                self._send_winnings_data_to_tensorboard()

            print(self.games['n'], " games played")

            # flush data from the memory for gc
            self.games['winnings'] = {}
            self.neural_network_history = {}
            self.play_history = {}
            self.neural_network_loss = {
                0: {'q': [], 'pi': []},
                1: {'q': [], 'pi': []}
            }
            print('game results logged')

    def _send_data_to_tensorboard(self):
        '''
        send only the corrected final rewards after every episode
        '''
        cur_t = time.time()

        # define episode length as the # of rounds where any action is taken by any player
        # currently there seems to be a bug where
        # self.actions[1] for p1 is [] and p2 [some action] which should not happen
        for p in self.players:
            episode_length = 0
            num_folds = 0
            num_calls = 0
            num_checks = 0
            num_all_ins = 0
            num_bets = 0
            num_nulls = 0
            num_raises = 0
            all_in_amounts = 0
            bet_amounts = 0
            raise_amounts = 0
            for b_round in range(4):
                action = self.actions[b_round][p.id]
                if action == []:
                    # no action recoded for this round
                    break
                episode_length += 1
                for a in action:
                    if a.type == 'fold':
                        num_folds += 1
                    elif a.type == 'call':
                        num_calls += 1
                    elif a.type == 'check':
                        num_checks += 1
                    elif a.type == 'all in':
                        num_all_ins += 1
                        all_in_amounts += a.value
                    elif a.type == 'null':
                        num_nulls += 1
                    elif a.type == 'bet':
                        num_bets += 1
                        bet_amounts += a.value
                    elif a.type == 'raise':
                        num_raises += 1
                        raise_amounts += a.value
                    else:
                        raise Exception('unrecognized action type {}'.format(a.type))
            num_actions = np.sum([num_folds, num_calls, num_checks, num_all_ins, num_bets, num_nulls,
                           num_raises])
            assert num_actions >= episode_length, "num_actions {} should be greater than or equal to episode \
            length {}".format(num_actions, episode_length)
            self.tensorboard.add_scalar_value('p{}_episode_length'.format(p.id+1), episode_length, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_folds_per_episode'.format(p.id+1), num_folds, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_calls_per_episode'.format(p.id+1), num_calls, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_checks_per_episode'.format(p.id+1), num_checks, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_all_ins_per_episode'.format(p.id+1), num_all_ins, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_nulls_per_episode'.format(p.id+1), num_nulls, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_bets_per_episode'.format(p.id+1), num_bets, cur_t)
            self.tensorboard.add_scalar_value('p{}_num_raises_per_episode'.format(p.id+1), num_raises, cur_t)
            self.tensorboard.add_scalar_value('p{}_all_in_amounts'.format(p.id+1), all_in_amounts, cur_t)
            self.tensorboard.add_scalar_value('p{}_bet_amounts'.format(p.id+1), bet_amounts, cur_t)
            self.tensorboard.add_scalar_value('p{}_raise_amounts'.format(p.id+1), raise_amounts, cur_t)
            self.tensorboard.add_scalar_value('p{}_reward'.format(p.id+1), self.total_reward_in_episode[p.id], cur_t)

    def _send_winnings_data_to_tensorboard(self):
        # logging the last winning results every log frequency
        for res in self.games['winnings'].values():
            for p in self.players:
                did_win = int(list(res)[p.id] == INITIAL_MONEY * len(self.players))
                self.tensorboard.add_scalar_value('p{}_winnings'.format(p.id+1), did_win, time.time())

    def _send_showdown_data_to_tensorboard(self, showdown_occurred):
        '''
        checks if showdown happened in this episode

        warning: may not include all showdown cases
        '''
        self.tensorboard.add_scalar_value('is_showdown_per_episode', showdown_occurred, time.time())

    def update_play_history_with_final_rewards(self):
        try:
            if self.players[0].has_played:
                self.play_history[self.global_step][0]['r'] += self.experiences[0]['final_reward']
            if self.players[1].has_played:
                self.play_history[self.global_step - 1][1]['r'] += self.experiences[1]['final_reward']
        except KeyError:
            if self.players[0].has_played:
                self.play_history[self.global_step - 1][0]['r'] += self.experiences[0]['final_reward']
            if self.players[1].has_played:
                self.play_history[self.global_step][1]['r'] += self.experiences[1]['final_reward']

    def _save_results(self, new_data, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                # extend the existing data with the new data
                data = pickle.load(f)
                data.update(new_data)

            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(new_data, f)

    def _set_new_game(self):
        if self.players[0].stack == 0 or self.players[1].stack == 0:
            self.update_winnings()
            self.save_history_results()
            self.new_game = True
        else:
            self.new_game = False

    def _handle_split(self):
        # SPLIT: everybody takes back its money
        # don't do self.pot_0. do pot_o
        pot_0 = self.players[0].contribution_in_this_pot
        pot_1 = self.players[1].contribution_in_this_pot
        pot_stack = (pot_0, pot_1, self.players[0].stack, self.players[1].stack)
        msg = 'split was not handled correctly:{}!={}'.format(pot_stack, 2 * INITIAL_MONEY)
        assert np.sum(pot_stack) == 2 * INITIAL_MONEY, msg
        self.players[0].stack += pot_0
        self.players[1].stack += pot_1
        self.players[0].contribution_in_this_pot = 0
        self.players[1].contribution_in_this_pot = 0
        self.total_reward_in_episode = {0: 0, 1: 0}

        # RL : update the memory with the amount you won
        self.experiences[0]['final_reward'] = 0  # the profit is 0 because you get back all what you put in the pot
        self.experiences[1]['final_reward'] = 0
        self.split = False

        self.experiences[0]['is_terminal'] = True
        self.experiences[1]['is_terminal'] = True

        opponent_stack = self.players[1].stack
        state_ = build_state(self.players[0], self.board, self.pot, self.actions, opponent_stack, BLINDS[1], as_variable=False)
        self.experiences[0]['s'] = state_
        self.experiences[0]['a'] = None

        opponent_stack = self.players[0].stack
        state_ = build_state(self.players[1], self.board, self.pot, self.actions, opponent_stack, BLINDS[1], as_variable=False)
        self.experiences[1]['s'] = state_
        self.experiences[1]['a'] = None

    def _handle_no_split(self):
        """Note that this function actually updates self.experiences with the final rewards and next state"""
        self.total_reward_in_episode = {0: 0, 1: 0}
        s_pot = self.players[0].contribution_in_this_pot, self.players[1].contribution_in_this_pot
        # there are 3 cases:
        # - the winner put exactly the same amount in the pot as the loser.
        #       In that case, the reward for winner is (=profit) is the pot/2, or pot - contribution
        #       Reward for loser is -contribution
        # - the winner put strictly more
        #       The reward is pot - contribution (same as 1st case, so that they can be merged in a single case)
        #       Reward for loser is - contribution
        # - the winner put strictly less
        #       The reward for the winner is 2*contribution
        #       Reward for loser is pot - 2*opp_contribution - contribution
        if s_pot[self.winner] * 2 >= self.pot:  # if winner contributed at least to 50% of the pot, it takes all
            self.players[self.winner].stack += self.pot
            self.total_reward_in_episode[self.winner] += self.pot - self.players[self.winner].contribution_in_this_pot
            self.total_reward_in_episode[1 - self.winner] -= self.players[1 - self.winner].contribution_in_this_pot
        else:  # if winner contributed to less than 50% of the pot, it gets 2x its contribution
            self.players[self.winner].stack += 2 * s_pot[self.winner]
            self.players[1 - self.winner].stack += self.pot - 2 * s_pot[self.winner]
            self.total_reward_in_episode[self.winner] += 2 * s_pot[self.winner] - self.players[self.winner].contribution_in_this_pot
            self.total_reward_in_episode[1 - self.winner] += self.pot - 2 * s_pot[self.winner] - self.players[1 - self.winner].contribution_in_this_pot

        # RL
        if self.players[self.winner].player_type == 'nfsp':
            if not self.players[self.winner].memory_rl.is_last_step_buffer_empty:
                self.experiences[self.winner]['final_reward'] = self.total_reward_in_episode[self.winner]
        if self.players[1 - self.winner].player_type == 'nfsp':
            if not self.players[1 - self.winner].memory_rl.is_last_step_buffer_empty:
                self.experiences[1 - self.winner]['final_reward'] = self.total_reward_in_episode[1 - self.winner]

        self.experiences[0]['is_terminal'] = True
        self.experiences[1]['is_terminal'] = True

        opponent_stack = self.players[1].stack
        state_ = build_state(self.players[0], self.board, self.pot, self.actions, opponent_stack, BLINDS[1], as_variable=False)
        self.experiences[0]['s'] = state_
        self.experiences[0]['a'] = None

        opponent_stack = self.players[0].stack
        state_ = build_state(self.players[1], self.board, self.pot, self.actions, opponent_stack, BLINDS[1], as_variable=False)
        self.experiences[1]['s'] = state_
        self.experiences[1]['a'] = None

    def _showdown(self):
        # compute the value of hands
        self.hand_1 = evaluate_hand(self.players[1].cards + self.board)
        self.hand_0 = evaluate_hand(self.players[0].cards + self.board)

        # decide whether to split or not
        if self.hand_1[1] == self.hand_0[1]:
            if self.hand_1[2] == self.hand_0[2]:
                self.split = True
            else:
                for self.card_0, self.card_1 in zip(self.hand_0[2], self.hand_1[2]):
                    if self.card_0 < self.card_1:
                        self.winner = 1
                        break
                    elif self.card_0 == self.card_1:
                        continue
                    else:
                        self.winner = 0
                        break

        # if no split, somebody won
        else:
            self.winner = int(self.hand_1[1] > self.hand_0[1])

        if self.verbose:
            if not self.split:
                assert self.winner in [0, 1], (self.winner, self.split, self.hand_0, self.hand_1)
                print(self.players[0].name + ' cards : ' + str(self.players[0].cards)
                      + ' and score: ' + str(self.hand_0[0]))
                print(self.players[1].name + ' cards : ' + str(self.players[1].cards)
                      + ' and score: ' + str(self.hand_1[0]))
                print(self.players[self.winner].name + ' wins')
            else:
                print(self.players[0].name + ' cards : ' + str(self.players[0].cards) +
                      ' and score: ' + str(self.hand_0[0]))
                print(self.players[1].name + ' cards : ' + str(self.players[1].cards) +
                      ' and score: ' + str(self.hand_1[0]))
                print('Pot split')

    def _reset_variables(self):
        # RESET VARIABLES
        self.winner = None
        self.pot = 0
        self.players[0].has_played = False
        self.players[1].has_played = False
        self.dealer = 1 - self.dealer
        self.players[self.dealer].is_dealer = True
        self.players[1 - self.dealer].is_dealer = False
        self.players[0].cards = []
        self.players[1].cards = []
        self.players[0].contribution_in_this_pot = 0
        self.players[1].contribution_in_this_pot = 0
        assert self.players[1].side_pot == self.players[0].side_pot == 0

    def _is_call_legit(self):
        # if you call, it must be exactly the value of the previous bet or raise or all-in
        side_pot = (self.action.value, self.player.side_pot)
        msg = 'call was not handled correctly: {}'.format(side_pot)
        if self.b_round > 0:
            assert np.sum(side_pot) == self.players[1 - self.player.id].side_pot, msg
            pass
        else:
            if len(self.actions[self.b_round][1 - self.player.id]) == 0:
                assert self.action.value == 1
                pass
            else:
                side_pot = (self.action.value, self.player.side_pot)
                assert np.sum(side_pot) == self.players[1 - self.player.id].side_pot, msg
                pass

    def _handle_fold(self):
        self.fold_occured = True
        self.players[0].contribution_in_this_pot = self.players[0].side_pot
        self.players[1].contribution_in_this_pot = self.players[1].side_pot
        self.players[0].side_pot = 0
        self.players[1].side_pot = 0
        self.winner = 1 - self.to_play
        if self.verbose:
            print(self.players[self.winner].name + ' wins because its opponent folded')

    def _handle_dramatic_action(self):
        if self.action.type == 'all in':
            self.all_in += 1
            self.player.is_all_in = True
            if self.player.side_pot <= self.players[1 - self.to_play].side_pot:
                # if self.action.value <= self.players[1 - self.to_play].side_pot:
                # in this case, the all in is a call and it leads to showdown
                self.all_in += 1
        elif (self.action.type == 'call') and (self.all_in == 1):
            # in this case, you call a all-in and it goes to showdown
            self.all_in += 1

    def _update_side_pot(self):
        self.players[0].contribution_in_this_pot += self.players[0].side_pot * 1
        self.players[1].contribution_in_this_pot += self.players[1].side_pot * 1
        self.players[0].side_pot = 0
        self.players[1].side_pot = 0

    def _update_pot_and_stacks(self, value):
        self.player.side_pot += value
        self.player.stack -= value
        assert self.player.stack >= 0, (self.player.stack, self.actions, self.action, value)
        self.pot += value
        assert self.pot + self.players[0].stack + self.players[1].stack == 2 * INITIAL_MONEY, (self.players, self.actions, self.action)
        assert not ((self.player.stack == 0) and self.action.type != 'all in'), (self.actions, self.action, self.player)
        self.actions[self.b_round][self.player.id].append(self.action)

    def make_experience(self, player, action, new_game, board, pot, dealer, actions,
                        big_blind, global_step, b_round):
        opponent_stack = self.players[1 - player.id].stack
        state_ = build_state(player, board, pot, actions, opponent_stack, big_blind, as_variable=False)

        action_ = action_to_array(action)
        reward_ = 0  # terminal rewards only !!!!!!!!!
        # should_blinds_be_added_to_the_action_value = int((len(actions[0][player.id]) == 0) * (b_round == 0))
        # reward_ = -action.total - should_blinds_be_added_to_the_action_value * ((dealer == player.id) * big_blind / 2 + (dealer != player.id) * big_blind)
        step_ = global_step

        '''
        a clumsy way to detect showdown transitions
        it clearly misses out some scenarios
        '''

        experience = {'s': state_,
                      'a': action_,
                      'r': reward_,
                      'next_s': None,
                      't': step_,
                      'is_new_game': self.new_game,
                      'is_terminal': False,
                      'final_reward': 0
                      }
        return experience

    def _make_new_exp(self):
        exp = {'s': 'TERMINAL',
               'a': None,
               'r': 0,
               's': None,
               't': self.global_step,
               'is_new_game': self.new_game,
               'is_terminal': False,
               'final_reward': 0
               }
        return exp

    def _log_game_play_speed(self):
        if self.last_game_start_time is None:
            self.last_game_start_time = time.time()
        else:
            game_delta = time.time() - self.last_game_start_time
            self.game_play_speeds.append(game_delta)
            self.last_game_start_time = time.time()
            if self.games['n'] % 50 == 0:
                print('seconds per game play: {}'.format(np.mean(self.game_play_speeds[-50:])))

    def _log_episode_play_speed(self):
        if self.last_episode_start_time is None:
            self.last_episode_start_time = time.time()
        else:
            episode_delta = time.time() - self.last_episode_start_time
            self.episode_play_speeds.append(episode_delta)
            self.last_episode_start_time = time.time()
            if self.games['#episodes'] % 50 == 0:
                print('seconds per episode play: {}'.format(np.mean(self.episode_play_speeds[-50:])))
