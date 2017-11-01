from keras.layers import Input, Concatenate, Add, Reshape, Flatten, Lambda
from keras.models import Model
from models import Conv1D, Dense
import keras.backend as K


def Q(hidden_dim=10, activation='prelu', n_actions=20):
    hand = Input((13, 4))  # zero everywhere and 1 for your cards
    board = Input((3, 13, 4))  # 3 rounds of board: flop [0,:,:], turn [1,:,:] and river [2,:,:]
    pot = Input((1,))
    stack = Input((1,))  # @todo: include this in the neural network
    opponent_stack = Input((1,))  # @todo: include this in the neural network
    blinds = Input((2,))  # small, big blinds
    dealer = Input((1,))  # 1 if you are the dealer, 0 otherwise
    opponent_model = Input((2,))  # tendency to raise, number of hands played: 2 numbers between 0 and 1
    preflop_plays = Input((6, 4, 2))  # 6 plays max (check then 5 times raises), 4 possible actions (check,bet,call,raise), 2 players
    flop_plays = Input((6, 4, 2))
    turn_plays = Input((6, 4, 2))
    river_plays = Input((6, 4, 2))
    # value_of_hand = Input((2,))  # combination_type, rank_in_this_type

    # PROCESSING OF CARDS (HAND+BOARD)
    # @todo : check that it makes sense. Create a Keras.layer Sum(axis)
    color_hand = Lambda(lambda x: K.sum(x, 1))(hand)
    color_board = Lambda(lambda x: K.sum(x, (1, 2)))(board)
    kinds_hand = Lambda(lambda x: K.sum(x, 2))(hand)
    kinds_board = Lambda(lambda x: K.sum(x, (1, 3)))(board)

    colors = Concatenate(2)([Reshape((4, 1))(color_hand), Reshape((4, 1))(color_board)])
    kinds = Concatenate(2)([Reshape((13, 1))(kinds_hand), Reshape((13, 1))(kinds_board)])

    kinds = Conv1D(5, 1, activation=activation)(kinds)
    kinds = Conv1D(1, 1, activation=activation)(
        Concatenate(-1)([
            Conv1D(1, 3)(kinds),
            Conv1D(3, 3, dilation_rate=2)(kinds)
        ])
    )
    kinds = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(kinds))
    colors = Conv1D(1, 1, activation=activation)(colors)
    colors = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(colors))

    # process board only
    board_ = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(board))
    board_ = Dense(hidden_dim, activation=activation, BN=True)(board_)

    # process board and hand together
    bh = Dense(hidden_dim, activation=activation, BN=True)(Concatenate()([Flatten()(board), Flatten()(hand)]))
    bh = Dense(hidden_dim, activation=activation, BN=True)(bh)
    cards = Dense(hidden_dim, activation=activation, BN=True)(Concatenate()([kinds, colors, board_, bh]))
    cards = Dense(hidden_dim, activation=activation, BN=True)(cards)

    situation_without_opponent = Add()([cards,
                                        Dense(hidden_dim, activation=activation, BN=True)(pot),
                                        Dense(hidden_dim, activation=activation, BN=True)(blinds),
                                        Dense(hidden_dim, activation=activation, BN=True)(dealer)
                                        ])
    situation_without_opponent = Dense(hidden_dim, activation=activation, BN=True)(situation_without_opponent)
    situation_without_opponent = Dense(hidden_dim, activation=activation, BN=True)(situation_without_opponent)

    # Auxiliary inputs
    # @todo: maybe it should be only a function of the cards themselves.
    # @todo: outputs: probabilities of having each combination at the end of the game + probabilities that the flop gives rise to these combinations + rank of current hand
    value_of_hand = Dense(hidden_dim, activation='sigmoid', BN=True)(situation_without_opponent)

    processed_preflop = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(preflop_plays))
    processed_flop = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(flop_plays))
    processed_turn = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(turn_plays))
    processed_river = Dense(hidden_dim, activation=activation, BN=True)(Flatten()(river_plays))
    processed_opponent = Dense(hidden_dim, activation=activation, BN=True)(opponent_model)
    plays = Dense(hidden_dim, activation=activation, BN=True)(Add()([processed_preflop,
                                                                     processed_flop,
                                                                     processed_turn,
                                                                     processed_river,
                                                                     processed_opponent
                                                                     ]))
    plays = Dense(hidden_dim, activation=activation, BN=True)(plays)

    # combine information from board and information from opponent
    situation_with_opponent = Dense(hidden_dim, activation=activation, BN=True)(
        Concatenate()([plays, situation_without_opponent]))
    situation_with_opponent = Dense(hidden_dim, activation=activation, BN=True)(situation_with_opponent)

    # obtain the Q values
    Q_values = Dense(n_actions, activation=None, BN=True)(situation_with_opponent)
    return Model([hand, board, pot, blinds, dealer, opponent_model, preflop_plays, flop_plays, turn_plays, river_plays],
                 [value_of_hand, Q_values])
