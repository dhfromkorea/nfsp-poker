from odds.evaluation import evaluate_hand
from models.q_network import QNetwork, PiNetwork

from players.strategies import strategy_RL, strategy_random, strategy_mirror, StrategyNFSP
from players.player import Player, NeuralFictitiousPlayer

from game.utils import *
from game.game_utils import Deck, set_dealer, blinds, deal, agreement, actions_to_array, action_to_array, cards_to_array
from game.config import BLINDS
from game.state import build_state, create_state_variable_batch

from models.featurizer import FeaturizerManager
from time import time
import numpy as np
import torch as t
import models.q_network
from torch.autograd import Variable

# define game constants here
INITIAL_MONEY = 100 * BLINDS[0]
NUM_ROUNDS = 4  # pre, flop, turn, river
NUM_HIDDEN_LAYERS = 50
NUM_ACTIONS = 16
P1_ETA = 0.1
P2_ETA = 0.1

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

    def __init__(self, featurizer_path,
                 # this should be set to a power of two for buffers
                 learn_start=128,
                 verbose=False,
                 cuda=False,
                 p1_strategy='RL',
                 p2_strategy='RL'):
        # define msc.
        self.verbose = verbose

        # define other non-game mechanisms like players
        featurizer = FeaturizerManager.load_model(featurizer_path, cuda=cuda)
        Q0 = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, featurizer)
        Q1 = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, featurizer)
        Pi0 = PiNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, featurizer)
        Pi1 = PiNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, featurizer)
        Q_networks = {0: Q0, 1: Q1}
        Pi_networks = {0: Pi0, 1: Pi1}
        self.players = self._generate_player_instances(p1_strategy, p2_strategy,
                                                       Q_networks, Pi_networks,
                                                       learn_start, verbose)

        # define battle-level game states here
        self.new_game = True
        self.games = {'n': 0, '#episodes': 0, 'winnings': {}}  # some statistics on the games
        self.global_step = 0

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
                strategy = StrategyNFSP(Q_networks[p_id], Pi_networks[p_id], P1_ETA)
                nfp = NeuralFictitiousPlayer(pid=p_id,
                                             strategy=strategy,
                                             stack=INITIAL_MONEY,
                                             name=p_names[p_id],
                                             learn_start=learn_start,
                                             verbose=verbose)
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
        buffer_length = str(tuple([p.memory_rl._buffer.record_size for p in self.players if p.player_type == 'nfsp']+[p.memory_sl._buffer.record_size for p in self.players if p.player_type == 'nfsp']))

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
        '''
        '''
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
            self._handle_no_fold()
        if self.split:
            self._handle_split()
        else:
            self._handle_no_split()

        # store final experience
        # KEEP TRACK OF TRANSITIONS
        if len(self.actions[self.b_round][0]) > 0:
            if self.players[0].player_type == 'nfsp':
                last_action = self.actions[self.b_round][0][-1]
                self.experiences[0] = self.make_experience(self.players[0], last_action, self.new_game, self.board,
                                                           self.pot, self.dealer, self.actions, BLINDS[1],
                                                           self.global_step, self.b_round)
                self.players[0].remember(self.experiences[0])
        if len(self.actions[self.b_round][1]) > 0:
            if self.players[1].player_type == 'nfsp':
                last_action = self.actions[self.b_round][1][-1]
                self.experiences[1] = self.make_experience(self.players[1], last_action, self.new_game, self.board,
                                                           self.pot, self.dealer, self.actions, BLINDS[1],
                                                           self.global_step, self.b_round)
                self.players[1].remember(self.experiences[1])

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
                # end episode
                break

    def _play_round(self):
        self.global_step += 1

        # CHOOSE AN ACTION
        self.player = self.players[self.to_play]
        assert self.player.stack >= 0, self.player.stack
        assert not ((self.player.stack == 0) and not self.player.is_all_in), (self.player, self.player.is_all_in, self.actions)

        self.action = self.player.play(self.board, self.pot, self.actions, self.b_round,
                                       self.players[1 - self.to_play].stack, self.players[1 - self.to_play].side_pot, BLINDS)

        if self.action.type == 'null':
            self.to_play = 1 - self.to_play
            self.null += 1
            if self.null >= 2:
                self.agreed = True
                # end the round with agreement
                return
            # go to the next agreement step
            return

        # RL : Store experiences in memory. Just for the agent
        if self.player.player_type == 'nsfp':
            self.experiences[self.player.id] = self.make_experience(self.player, self.action, self.new_game, self.board,
                                                                    self.pot, self.dealer, self.actions, BLINDS[1],
                                                                    self.global_step, self.b_round)
            self.player.remember(self.experiences[self.player.id])

        # TRANSITION STATE DEPENDING ON THE ACTION YOU TOOK
        if self.action.type in {'all in', 'bet', 'call'}:  # impossible to bet/call/all in 0
            try:
                assert self.action.value > 0
                pass
            except AssertionError:
                raise AssertionError

        if self.action.type == 'call':
            self._handle_call()

        if self.action.type == 'raise':
            value = self.action.total
            # value = self._handle_raise()
        else:
            value = self.action.value
        # update pot
        self._update_pot(value)
        # DRAMATIC ACTION MONITORING
        self._handle_dramatic_action()

        # break if fold
        if self.action.type == 'fold':
            self._handle_fold()
            # TODO: break with agreement=True?
            self.agreed = True
            return

        # DECIDE WHETHER IT IS THE END OF THE BETTING ROUND OR NOT, AND GIVE LET THE NEXT PLAYER PLAY
        self.agreed = agreement(self.actions, self.b_round)
        self.to_play = 1 - self.to_play

    def update_winnings(self, log_freq=100):
        self.games['winnings'][self.games['n']] = {self.players[0].stack, self.players[1].stack}
        if self.games['n'] % log_freq == 0:
            print(self.games['n'], " games over")

    def _set_new_game(self):
        if self.players[0].stack == 0 or self.players[1].stack == 0:
            self.update_winnings()
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

    def _handle_no_split(self):
        # if the winner isn't all in, it takes everything
        if self.players[self.winner].stack > 0:
            self.players[self.winner].stack += self.pot

        # if the winner is all in, it takes only min(what it put in the pot*2, pot)
        else:
            s_pot = self.players[0].contribution_in_this_pot, self.players[1].contribution_in_this_pot
            if s_pot[self.winner] * 2 > self.pot:
                self.players[self.winner].stack += self.pot
            else:
                self.players[self.winner].stack += 2 * s_pot[self.winner]
                self.players[1 - self.winner].stack += self.pot - 2 * s_pot[self.winner]

        # RL
        # If the agent won, gives it the chips and reminds him that it won the chips
        if self.winner == 0:
            # if the opponent immediately folds, then the MEMORY is empty and there is no reward to add since you didn't have the chance to act
            if not self.players[0].memory_rl.is_last_step_buffer_empty:
                self.experiences[0]['final_reward'] = self.pot

            if not self.players[1].memory_rl.is_last_step_buffer_empty:
                self.experiences[1]['final_reward'] = self.pot

    def _handle_no_fold(self):
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

    def _handle_raise(self):
        value = self.action.value + self.players[1 - self.to_play].side_pot - self.player.side_pot - (len(self.actions[0][self.player.id]) == 0) * (self.b_round == 0) * BLINDS[1 - self.player.is_dealer]
        return value

    def _reset_variables(self):
        # RESET VARIABLES
        self.winner = None
        self.pot = 0
        self.dealer = 1 - self.dealer
        self.players[self.dealer].is_dealer = True
        self.players[1 - self.dealer].is_dealer = False
        self.players[0].cards = []
        self.players[1].cards = []
        self.players[0].contribution_in_this_pot = 0
        self.players[1].contribution_in_this_pot = 0
        assert self.players[1].side_pot == self.players[0].side_pot == 0

    def _handle_call(self):
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
            if self.action.value <= self.players[1 - self.to_play].side_pot:
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

    def _update_pot(self, value):
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
        reward_ = -action.value - (b_round == 0) * ((dealer == player.id) * big_blind / 2 + (dealer != player.id) * big_blind)
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
