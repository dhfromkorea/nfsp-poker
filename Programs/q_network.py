from keras.layers import Input, Concatenate, Add, Reshape, Flatten, Lambda
from keras.models import Model
from models import Conv1D, Dense, Sum, IsNonZero
from keras.optimizers import Adam


def Q_network(hidden_dim=10, n_actions=14):
    hand = Input((13,4))  # zero everywhere and 1 for your cards
    board = Input((3,13,4))  # 3 rounds of board: flop [0,:,:], turn [1,:,:] and river [2,:,:]
    pot = Input((1,))
    stack = Input((1,))  # @todo: include this in the neural network
    opponent_stack = Input((1,))  # @todo: include this in the neural network
    blinds = Input((2,))  # small, big blinds
    dealer = Input((1,))  # 1 if you are the dealer, 0 otherwise
    # opponent_model = Input((2,))  # tendency to raise, number of hands played: 2 numbers between 0 and 1
    preflop_plays = Input((6, 5, 2))  # 6 plays max (check then 5 times raises), 5 possible actions (check,bet,call,raise,all-in), 2 players
    flop_plays = Input((6, 5, 2))
    turn_plays = Input((6, 5, 2))
    river_plays = Input((6, 5, 2))
    # value_of_hand = Input((2,))  # combination_type, rank_in_this_type

    # Processing board and hand specifically to detect flush and straight
    color_hand = Sum(1)(hand)
    color_board = Sum((1,2))(board)
    kinds_hand_for_ptqf = Sum(2)(hand)
    kinds_hand_for_straight = IsNonZero()(kinds_hand_for_ptqf)
    kinds_board_for_ptqf = Sum((1,3))(board)
    kinds_board_for_straight = IsNonZero()(kinds_board_for_ptqf)

    colors = Concatenate(2)([Reshape((4,1))(color_hand), Reshape((4,1))(color_board)])
    kinds_for_straight = Concatenate(2)([Reshape((13,1))(kinds_hand_for_straight), Reshape((13,1))(kinds_board_for_straight)])
    kinds_for_ptqf = Flatten()(Concatenate(2)([Reshape((13,1))(kinds_hand_for_ptqf), Reshape((13,1))(kinds_board_for_ptqf)]))
    kinds_for_straight = Conv1D(5,1, activation="selu", BN=False)(kinds_for_straight)
    kinds_for_straight = Conv1D(1, 1, activation='selu', BN=False)(
        Concatenate(-1)([
            Conv1D(1, 3, activation='selu', BN=False)(kinds_for_straight),
            Conv1D(3, 3, dilation_rate=2, activation='selu', BN=False)(kinds_for_straight)
        ])
    )
    kinds_for_straight = Dense(hidden_dim, activation='selu', BN=False)(Flatten()(kinds_for_straight))

    kinds_for_ptqf = Dense(hidden_dim, activation='selu', BN=False)(kinds_for_ptqf)
    kinds_for_ptqf = Dense(hidden_dim, activation='selu', BN=False)(kinds_for_ptqf)
    colors = Conv1D(1, 1, activation='selu', BN=False)(colors)
    colors = Dense(hidden_dim, activation='selu', BN=False)(Flatten()(colors))

    # Process board only
    flop_alone = Dense(hidden_dim, activation='selu', BN=False)(Lambda(lambda x: x[:, 0, :, :])(board))
    flop_alone = Dense(hidden_dim, activation='selu', BN=False)(flop_alone)
    turn_alone = Dense(hidden_dim, activation='selu', BN=False)(Lambda(lambda x: x[:, 1, :, :])(board))
    turn_alone = Dense(hidden_dim, activation='selu', BN=False)(turn_alone)
    river_alone = Dense(hidden_dim, activation='selu', BN=False)(Lambda(lambda x: x[:, 2, :, :])(board))
    river_alone = Dense(hidden_dim, activation='selu', BN=False)(river_alone)

    board_alone = Dense(hidden_dim, activation='selu', BN=False)(Flatten()(Concatenate()([flop_alone, turn_alone, river_alone])))
    board_alone = Dense(hidden_dim, activation='selu', BN=False)(board_alone)

    # Process board and hand together
    bh = Dense(hidden_dim, activation='selu', BN=False)(Concatenate()([Flatten()(board), Flatten()(hand)]))
    bh = Dense(hidden_dim, activation='selu', BN=False)(bh)
    bh = Dense(hidden_dim, activation='selu', BN=False)(Concatenate()([kinds_for_ptqf, kinds_for_straight, colors, board_alone, bh]))
    bh = Dense(hidden_dim, activation='selu', BN=False)(bh)

    n_combination = 9
    probabilities_of_each_combination_board_only = Dense(n_combination, activation='softmax', BN=False)(bh)
    probabilities_of_each_combination_board_hand = Dense(n_combination, activation='softmax', BN=False)(bh)
    board_value = Dense(1, activation='sigmoid', BN=False)(bh)
    board_hand_value = Dense(1, activation='sigmoid', BN=False)(bh)

    # Add pot, blind, dealer, stacks
    pbds = Dense(hidden_dim, activation='selu')(Concatenate()([pot, blinds, dealer, stack, opponent_stack]))
    pbds = Dense(hidden_dim, activation='selu')(pbds)


    # Process plays
    processed_preflop = Dense(hidden_dim, activation='selu')(Flatten()(preflop_plays))
    processed_flop = Dense(hidden_dim, activation='selu')(Concatenate()([Flatten()(flop_plays), Flatten()(flop_alone)]))
    processed_turn = Dense(hidden_dim, activation='selu')(Concatenate()([Flatten()(turn_plays), Flatten()(flop_alone), Flatten()(turn_alone)]))
    processed_river = Dense(hidden_dim, activation='selu')(Concatenate()([Flatten()(river_plays), Flatten()(flop_alone), Flatten()(turn_alone), Flatten()(river_alone)]))
    # processed_opponent = Dense(hidden_dim, activation='prelu', BN=True)(opponent_model)
    plays = Dense(hidden_dim, activation='selu')(Add()([processed_preflop,
                                                        processed_flop,
                                                        processed_turn,
                                                        processed_river,
    #                                                     processed_opponent
                                                       ]))
    plays = Dense(hidden_dim, activation='selu')(plays)

    situation_with_opponent = Dense(hidden_dim, activation='selu')(Concatenate()([plays, pbds, bh]))
    situation_with_opponent = Dense(hidden_dim, activation='selu')(situation_with_opponent)
    Q_values = Dense(n_actions, activation=None)(situation_with_opponent)

    Q = Model([hand, board, pot, stack, opponent_stack, blinds, dealer, preflop_plays, flop_plays, turn_plays, river_plays], [Q_values, probabilities_of_each_combination_board_only, probabilities_of_each_combination_board_hand, board_value, board_hand_value])
    Q.compile(Adam(), ['mse', 'categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
    return Q
