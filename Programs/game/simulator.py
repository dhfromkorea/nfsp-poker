from time import time
from odds.evaluation import *
from models.q_network import QNetwork, PiNetwork

from players.strategies import strategy_RL, strategy_random
from players.player import Player

from game.utils import *
from game.game_utils import Deck, set_dealer, blinds, deal, agreement, actions_to_array, action_to_array, cards_to_array
from game.config import BLINDS


from game.state import build_state, create_state_variable_batch


import torch as t
from torch.autograd import Variable

# define game constants here
INITIAL_MONEY = 100*BLINDS[0]
NUM_ROUNDS = 4 # pre, flop, turn, river


class Simulator:
    '''
    right now, RL algo, Simulator, ER are all tightly coupled
    this is bad.
    TODO: decouple them

    right now, players are not using networks to choose actions
    TODO: couple players with networks
    '''
    def __init__(self, verbose):
        self.verbose = verbose
        # define other non-game mechanisms

        '''
        TODO: not ready fully yet
        self.players = [NeuralFictiousPlayer(pid=0, name='SB'),
                        NeuralFictiousPlayer(pid=1, name='DH')]
        '''

        # define battle-level game states here
        self.players = [Player(0, strategy_RL(Q, True), INITIAL_MONEY, verbose=self.verbose, name='SB'),
                   Player(1, strategy_RL(Q, True), INITIAL_MONEY, verbose=self.verbose, name='DH')]
        self.new_game = True
        self.games = {'n': 0, '#episodes': 0}  # some statistics on the games
        self.global_step = 0

        # define episode-level game states here
        self.deck = Deck()
        self.dealer = set_dealer(self.players)
        self.board = []


    def start(self):
        while True:
            self._prepare_new_game()
            safe_to_start = self._prepare_new_episode()
            if not safe_to_start:
                raise Exception('corrupt game')

            self._start_episode()
            self._train_with_experiences()

    def _prepare_new_game(self):
        '''
        if new game -> initialize
        '''
        # at the beginning of a whole new game (one of the player lost or it is the first), all start with the same amounts of money again
        if self.new_game:
            self.games['n'] += 1
            # buffer_length = buffer_rl.size
            buffer_length = 0

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
        self.pot = blinds(self.players, verbose=self.verbose)
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
        self.experience = {'s': 'TERMINAL',
              'r': 0,
              't': self.global_step,
              'is_new_game': self.new_game,
              'is_terminal': False,
              'final_reward': 0
             }
        # WINNERS GETS THE MONEY.
        # WATCH OUT! TIES CAN OCCUR. IN THAT CASE, SPLIT
        self.split = False
        self._handle_no_fold()
        self._handle_no_split()
        self._handle_split()
        
        # store final experience
        # KEEP TRACK OF TRANSITIONS
        self.experience = self.make_experience(self.players, self.action, self.new_game, self.board,
                                     self.pot, self.dealer, self.actions, BLINDS[1],
                                     self.global_step)

        players[0].remember(self.experience)
        # RESET VARIABLES
        self._reset_variables()
        # IS IT THE END OF THE GAME ? (bankruptcy)
        self._set_new_game()


    def _play_rounds(self):
        # EPISODE ACTUALLY STARTS HERE
        for r in range(NUM_ROUNDS):
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
                
                players[0].remember(self.experience)

                # END THE EPISODE
                self._update_side_pot()
                # end episode
                break


    def _play_round(self):
        self.global_step += 1
        
        # CHOOSE AN ACTION
        self.player = self.players[self.to_play]
        assert self.player.stack >= 0, self.player.stack
        assert not((self.player.stack == 0) and not self.player.is_all_in), (self.player, self.player.is_all_in, self.actions)
        self.action = self.player.play(self.board, self.pot, self.actions, self.b_round,
                             self.players[1 - self.to_play].stack, self.players[1 - self.to_play].side_pot, BLINDS)
        
        if self.action.type == 'null':
            self.to_play = 1 - self.to_play
            self.null += 1
            if null >= 2:
                self.agreed = True
                # end the round with agreement
                return
            # go to the next agreement step
            return

        # RL : Store experiences in memory. Just for the agent
        if self.player.id == 0:
            # KEEP TRACK OF TRANSITIONS
            self.experience = self.make_experience(self.players, self.action, self.new_game, self.board,
                                         self.pot, self.dealer, self.actions, BLINDS[1],
                                         self.global_step)
            players[0].remember(self.experience)

        # TRANSITION STATE DEPENDING ON THE ACTION YOU TOOK
        if self.action.type in {'all in', 'bet', 'call'}:  # impossible to bet/call/all in 0
            try:
                assert self.action.value > 0
            except AssertionError:
                raise AssertionError
        
        if self.action.type == 'call':
            self._handle_call()

        if self.action.type == 'raise':
            self._handle_raise()
        else:
            self.value = self.action.value

        # update pot
        self._update_pot()
        # DRAMATIC ACTION MONITORING
        self._handle_dramatic_action()

        # break if fold
        if self.action.type == 'fold':
           self._handle_fold()
           # TODO: break with agreement=True?
           return

        # DECIDE WHETHER IT IS THE END OF THE BETTING ROUND OR NOT, AND GIVE LET THE NEXT PLAYER PLAY
        self.agreed = agreement(self.actions, self.b_round)
        self.to_play = 1 - self.to_play


    def _set_new_game(self):
        if self.players[0].stack == 0 or self.players[1].stack == 0:
            self.new_game = True
        else:
            self.new_game = False


    def _handle_split(self):
        # SPLIT: everybody takes back its money
        self.pot_0, self.pot_1 = self.players[0].contribution_in_this_pot, self.players[1].contribution_in_this_pot
        res = 2*INITIAL_MONEY, (self.pot_0, self.pot_1, self.players[0].stack, self.players[1].stack)
        assert self.pot_0 + self.pot_1 + self.players[0].stack + self.players[1].stack == res
        self.players[0].stack += self.pot_0
        self.players[1].stack += self.pot_1
        self.players[0].contribution_in_this_pot = 0
        self.players[1].contribution_in_this_pot = 0

        # RL : update the memory with the amount you won
        self.experience['final_reward']= self.pot_0
        self.split = False


    def _handle_no_split(self):
        if not self.split:
            # if the winner isn't all in, it takes everything
            if self.players[winner].stack > 0:
                self.players[winner].stack += pot

            # if the winner is all in, it takes only min(what it put in the pot*2, pot)
            else:
                self.s_pot = self.players[0].contribution_in_this_pot, self.players[1].contribution_in_this_pot
                if self.s_pot[self.winner]*2 > self.pot:
                    self.players[self.winner].stack += self.pot
                else:
                    self.players[self.winner].stack += 2*self.s_pot[self.winner]
                    self.players[1 - self.winner].stack += self.pot - 2*self.s_pot[self.winner]

            # RL
            # If the agent won, gives it the chips and reminds him that it won the chips
            if self.winner == 0:
                # if the opponent immediately folds, then the MEMORY is empty and there is no reward to add since you didn't have the chance to act
                if not self.buffer_rl.is_last_step_buffer_empty:
                    self.experience['final_reward']= self.pot


    def _handle_no_fold(self):
        if not self.fold_occured:
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
                    print(self.players[0].name + ' cards : ' + str(self.players[0].cards) 
                          + ' and score: ' + str(self.hand_0[0]))
                    print(self.players[1].name + ' cards : ' + str(self.players[1].cards) 
                          + ' and score: ' + str(self.hand_1[0]))
                    print(self.players[winner].name + ' wins')
                else:
                    print(self.players[0].name + ' cards : ' + str(self.players[0].cards) +
                          ' and score: ' + str(self.hand_0[0]))
                    print(self.players[1].name + ' cards : ' + str(self.players[1].cards) +
                          ' and score: ' + str(self.hand_1[0]))
                    print('Pot split')


    def _handle_raise(self):
        self.value = self.action.value + self.players[1-self.to_play].side_pot - self.player.side_pot - (len(self.actions[0][self.player.id])==0)*(self.b_round==0)*BLINDS[1-self.player.is_dealer]

    def _reset_variables(self):
        # RESET VARIABLES
        self.pot = 0
        self.dealer = 1 - self.dealer
        self.players[dealer].is_dealer = True
        self.players[1 - self.dealer].is_dealer = False
        self.players[0].cards = []
        self.players[1].cards = []
        self.players[0].contribution_in_this_pot = 0
        self.players[1].contribution_in_this_pot = 0
        assert self.players[1].side_pot == self.players[0].side_pot == 0


    def _handle_call(self):
        # if you call, it must be exactly the value of the previous bet or raise or all-in
        if self.b_round > 0:
            res = self.players[1-player.id].side_pot, (self.action.value, self.actions[self.b_round][1-self.player.id][-1].value)
            assert self.action.value + self.player.side_pot == res
        else:
            if len(self.actions[self.b_round][1-self.player.id]) == 0:
                assert self.action.value == 1
            else:
                res =  self.players[1-self.player.id].side_pot, (self.actions, self.action.value, self.actions[self.b_round][1 - self.player.id][-1].value)
                assert self.action.value + self.player.side_pot == res


    def _handle_fold(self):
        self.fold_occured = True
        self.players[0].contribution_in_this_pot = self.players[0].side_pot*1
        self.players[1].contribution_in_this_pot = self.players[1].side_pot*1
        self.players[0].side_pot = 0
        self.players[1].side_pot = 0
        self.winner = 1 - self.to_play
        if self.verbose:
            print(self.players[self.winner].name + ' wins because its opponent folded')
        # break the episode


    def _handle_dramatic_action(self):
        if self.action.type == 'all in':
            self.all_in += 1
            self.player.is_all_in = True
            if self.action.value <= self.players[1-self.to_play].side_pot:
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


    def _update_pot(self):
        self.player.side_pot += self.value
        self.player.stack -= self.value
        assert self.player.stack >= 0, (self.player.stack, self.actions, self.action, self.value)
        self.pot += self.value
        assert self.pot + self.players[0].stack + self.players[1].stack == 2*INITIAL_MONEY, (self.players, self.actions, self.action)
        assert not ((self.player.stack == 0) and self.action.type != 'all in'), (self.actions, self.action, self.player)
        self.actions[self.b_round][self.player.id].append(self.action)


    def make_experience(self, players, action, new_game, board, pot, dealer, actions,
                        big_blind, global_step):
        # ugly...
        state = [players[0], board, np.array([players[0].stack]),
                 actions, None, np.array([players[1].stack]),
                 big_blind]
        state_ = build_state(*state, as_variable=False)

        action_ = action_to_array(action)
        reward_ = -action.value
        step_ = global_step

        # we need to inform replay manager of some extra stuff
        experience = {'s': state_,
                      'a': action_,
                      'r': reward_,
                      'next_s': None,
                      't': step_,
                      'is_new_game': new_game,
                      'is_terminal': False,
                      'final_reward': 0
                     }
        return experience
    

