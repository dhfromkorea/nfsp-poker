from odds.evaluation import evaluate_hand
from models.q_network import QNetworkBN as QNetwork, PiNetworkBN as PiNetwork

from players.strategies import strategy_RL, strategy_random, strategy_mirror, StrategyNFSP
from players.player import Player, NeuralFictitiousPlayer

from game.utils import get_last_round
from game.game_utils import Deck, set_dealer, blinds, deal, agreement, actions_to_array, array_to_cards, action_to_array, cards_to_array
from game.config import BLINDS
from game.state import build_state, create_state_variable_batch

from models.featurizer import FeaturizerManager
from time import time
import numpy as np
import torch as t
import pickle
import models.q_network
from time import time
import os.path
from torch.autograd import Variable

# paths
SAVED_FEATURIZER_PATH = 'data/hand_eval/best_models/card_featurizer1.50-10.model.pytorch'
GAME_SCORE_HISTORY_PATH = 'data/game_score_history/game_score_history_{}.p'.format(time())
PLAY_HISTORY_PATH = 'data/play_history/play_history_{}.p'.format(time())
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/neural_network_history_{}.p'.format(time())
NEURAL_NETWORK_LOSS_PATH = 'data/neural_network_history/loss/loss_{}.p'.format(time())
EXPERIMENT_PATH = 'data/tensorboard/'
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
                 learning_rate,
                 target_Q_update_freq,
                 # use default values
                 featurizer_path=SAVED_FEATURIZER_PATH,
                 game_score_history_path=GAME_SCORE_HISTORY_PATH,
                 play_history_path=PLAY_HISTORY_PATH,
                 neural_network_history_path=NEURAL_NETWORK_HISTORY_PATH,
                 neural_network_loss_path=NEURAL_NETWORK_LOSS_PATH,
                 memory_rl_config={},
                 memory_sl_config={},
                 verbose=False,
                 cuda=False,
                 tensorboard=None,
                 p1_strategy='RL',
                 p2_strategy='RL'):
        # define msc.
        self.verbose = verbose
        self.cuda = cuda
        self.log_freq = log_freq
        self.eps = eps
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_Q_update_freq = target_Q_update_freq
        self.learn_start = learn_start

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
                                    1: {'q':[]}, 'pi': []}
        # 4. tensorboard
        self.tensorboard = tensorboard

        # define game-level game states here
        self.new_game = True
        self.global_step = 0

        # define players
        featurizer = FeaturizerManager.load_model(featurizer_path, cuda=cuda)
        Q0 = QNetwork(n_actions=NUM_ACTIONS,
                      hidden_dim=NUM_HIDDEN_LAYERS,
                      featurizer=featurizer,
                      game_info=self.games,  # bad, but simple (@hack)
                      player_id=0,
                      neural_network_history=self.neural_network_history,
                      neural_network_loss=self.neural_network_loss,
                      tensorboard=tensorboard,
                      cuda=cuda)
        Q1 = QNetwork(n_actions=NUM_ACTIONS,
                      hidden_dim=NUM_HIDDEN_LAYERS,
                      featurizer=featurizer,
                      game_info=self.games,  # bad, but simple (@hack)
                      player_id=1,
                      neural_network_history=self.neural_network_history,
                      neural_network_loss=self.neural_network_loss,
                      tensorboard=tensorboard,
                      cuda=cuda)
        Pi0 = PiNetwork(n_actions=NUM_ACTIONS,
                        hidden_dim=NUM_HIDDEN_LAYERS,
                        featurizer=featurizer,
                        q_network=Q0,  # to share weights
                        game_info=self.games,  # bad, but simple (@hack)
                        player_id=0,
                        neural_network_history=self.neural_network_history,
                        neural_network_loss=self.neural_network_loss,
                        tensorboard=tensorboard,
                        cuda=cuda)
        Pi1 = PiNetwork(n_actions=NUM_ACTIONS,
                        hidden_dim=NUM_HIDDEN_LAYERS,
                        featurizer=featurizer,
                        q_network=Q1,  # to share weights with Q1
                        game_info=self.games,  # bad, but simple (@hack)
                        player_id=1,
                        neural_network_history=self.neural_network_history,
                        neural_network_loss=self.neural_network_loss,
                        tensorboard=tensorboard,
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
                                             learning_rate=self.learning_rate,
                                             target_Q_update_freq=self.target_Q_update_freq,
                                             memory_rl_config=self.memory_rl_config,
                                             memory_sl_config=self.memory_sl_config,
                                             learn_start=self.learn_start,
                                             verbose=self.verbose,
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
        # buffer_length = buffer_rl.size
        buffer_length = str(tuple([p.memory_rl._buffer.record_size for p in self.players if p.player_type == 'nfsp'] + [p.memory_sl._buffer.record_size for p in self.players if p.player_type == 'nfsp']))

        if self.verbose:
            t0 = time()
            print('####################'
                  'New game (%s) starts.\n'
                  'Players get cash\n'
                  'Last game lasted %.1f\n'
                  'Memory contains %s experiences\n'
                  '####################' % (str(self.games['n']), time() - t0, buffer_length))
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
        if not self.fold_occured:
            self._showdown()
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
                self._send_correct_final_reward_to_tensorboard()
        except KeyError:
            raise KeyError

        for p in self.players:
            if p.player_type == 'nfsp':
                p.learn(self.global_step, self.games['#episodes'])


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
                # DEAL REMAINING CARDS
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

        self.action = self.player.play(self.board, self.pot, self.actions, self.b_round,
                                       self.players[1 - self.to_play].stack, self.players[1 - self.to_play].side_pot, BLINDS)

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
        ph[self.player.id] = {'s': (array_to_cards(exp['s'][0]), array_to_cards(exp['s'][1])), 'a': exp['a'], 'r': exp['r']}
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
            # we save all history data here
            self._save_results(self.games['winnings'], self.game_score_history_path)
            self._save_results(self.play_history, self.play_history_path)
            self._save_results(self.neural_network_history, self.neural_network_history_path)
            self._save_results(self.neural_network_loss, self.neural_network_loss_path)
            self.tensorboard.to_zip('{}_'.format(EXPERIMENT_PATH, time()))
            print(self.games['n'], " games over")

    def _send_correct_final_reward_to_tensorboard(self):
        '''
        send only the corrected final rewards after every episode
        '''
        if 0 in self.play_history[self.global_step]:
            gs_p1 = self.global_step
            gs_p2 = self.global_step - 1
        else:
            gs_p1 = self.global_step - 1
            gs_p2 = self.global_step

        t = time()

        if self.players[0].has_played:
            # p1's final reward was updated, we overwrite the history
            final_reward_p1 = self.play_history[gs_p1][0]['r']
            self.tensorboard.add_scalar_value('reward_1', final_reward_p1, t)
        if self.players[1].has_played:
            # p2's final reward was updated, we overwrite the history
            final_reward_p2 = self.play_history[gs_p2][1]['r']
            self.tensorboard.add_scalar_value('reward_2', final_reward_p2, t)

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

        # RL : update the memory with the amount you won
        self.experiences[0]['final_reward'] = pot_0
        self.experiences[1]['final_reward'] = pot_1
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
        # if the winner isn't all in, it takes everything
        if self.players[self.winner].stack > 0:
            self.players[self.winner].stack += self.pot

            # RL
            if self.players[self.winner].player_type == 'nfsp':
                if not self.players[self.winner].memory_rl.is_last_step_buffer_empty:
                    self.experiences[self.winner]['final_reward'] = self.pot

        # if the winner is all in, it takes only min(what it put in the pot*2, pot)
        else:
            s_pot = self.players[0].contribution_in_this_pot, self.players[1].contribution_in_this_pot
            if s_pot[self.winner] * 2 > self.pot:
                self.players[self.winner].stack += self.pot

                # RL
                if self.players[self.winner].player_type == 'nfsp':
                    if not self.players[self.winner].memory_rl.is_last_step_buffer_empty:
                        self.experiences[self.winner]['final_reward'] = self.pot
            else:
                self.players[self.winner].stack += 2 * s_pot[self.winner]
                self.players[1 - self.winner].stack += self.pot - 2 * s_pot[self.winner]

                # RL
                if self.players[self.winner].player_type == 'nfsp':
                    if not self.players[self.winner].memory_rl.is_last_step_buffer_empty:
                        self.experiences[self.winner]['final_reward'] = 2 * s_pot[self.winner]
                if self.players[1 - self.winner].player_type == 'nfsp':
                    if not self.players[1 - self.winner].memory_rl.is_last_step_buffer_empty:
                        self.experiences[1 - self.winner]['final_reward'] = self.pot - 2 * s_pot[self.winner]

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
        should_blinds_be_added_to_the_action_value = int((len(actions[0][player.id]) == 0) * (b_round == 0))
        reward_ = -action.total - should_blinds_be_added_to_the_action_value * ((dealer == player.id) * big_blind / 2 + (dealer != player.id) * big_blind)
        step_ = global_step

        # we need to inform replay manager of some extra stuff
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
