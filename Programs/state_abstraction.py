import numpy as np

from game_utils import Card


def cards_to_array(cards):
    """
    Convert a list of cards (the board or the hand) into a numpy array to be passed as input of the DQN
    :param cards: a list of Card objects
    :return: an array representing these cards.
             Note that if there are more than 2 cards (i.e if this is the board),
             then the first 3 card are grouped together
    """
    if len(cards) == 2:
        array = np.zeros((13, 4))
        for card in cards:
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[value, suit] = 1
        return array
    elif len(cards) == 1 or len(cards) > 5:
        raise ValueError('there should be either 0, 2,3,4, or 5 cards')
    elif len(cards) == 0:
        return np.zeros((3, 13, 4))
    elif len(cards) == 3:
        array = np.zeros((3, 13, 4))
        for card in cards:
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[0, value, suit] = 1
        return array
    elif len(cards) == 4:
        array = np.zeros((3, 13, 4))
        for i, card in enumerate(cards):
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[int(i >= 3), value, suit] = 1
        return array
    elif len(cards) == 5:
        array = np.zeros((3, 13, 4))
        for i, card in enumerate(cards):
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            if i < 3:
                idx = 0
            elif i == 3:
                idx = 1
            elif i == 4:
                idx = 2
            array[idx, value, suit] = 1
        return array


def action_to_array(action):
    """
    Convert an action into a numpy array
    Actions will be `check`, `bet`, `call`, `raise`, `all in`
        `check` has to be included to be differentiated from not having played. If you just say that it is equivalent as
        betting 0, then you don't know whether it played or not
        `fold` should not be taken into account since it ends the game  @todo: sure of that ???
    :param action: an Action object
    :return: a numpy array
    """
    array = np.zeros((5,))
    if action.type == 'check':
        array[0] = 1
    elif action.type == 'bet':
        array[1] = action.value
    elif action.type == 'call':
        array[2] = action.value
    elif action.type == 'raise':
        array[3] = action.value
    elif action.type == 'all in':
        array[4] = action.value
    return array


def actions_to_array(actions):
    """
    Convert a sequence of actions into several numpy arrays
    :param actions: a dict {b_round: {player: [actions]}}
    :return: 4 arrays, one per betting round
    """
    all_plays = []
    for b_round, players in actions.items():
        if b_round == -1:
            continue
        b_round_plays = np.zeros((6, 5, 2))  # 6: max number of actions in one round. 5: total number of possible actions. 2: number of players. 0 is the agent and 1 its opponent
        for player, plays in players.items():
            for k, action in enumerate(plays):
                b_round_plays[k, :, player] = action_to_array(action)
        all_plays.append(b_round_plays)
    return all_plays


def build_state(player, board, pot, actions, b_round, opponent_stack, blinds):
    # @todo: add opponent modeling
    """
    Return state as numpy arrays (inputs of Q networks)
        - hand
        - board
        - pot
        - stack
        - opponent stack
        - blinds
        - dealer
        - opponent model
        - preflop plays
        - flop plays
        - turn plays
        - river plays
    :param player:
    :param board:
    :param pot:
    :param actions:
    :param b_round:
    :param opponent_stack:
    :param blinds:
    :return:
    """
    hand = cards_to_array(player.cards)
    board = cards_to_array(board)
    pot_ = np.array([pot])
    stack_ = np.array([player.stack])
    opponent_stack_ = np.array([opponent_stack])
    blinds_ = np.array(blinds)
    dealer = np.array([player.id if player.is_dealer else 1 - player.id])
    preflop_plays, flop_plays, turn_plays, river_plays = actions_to_array(actions)
    return hand, board, pot_, stack_, opponent_stack_, blinds_, dealer, preflop_plays, flop_plays, turn_plays, river_plays