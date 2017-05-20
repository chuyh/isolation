"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

# the improved score from the lectures, where we
# multiply the opponent's remaining moves by 2
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    import numpy as np
    return float(own_moves- 2*opp_moves)
    # return float(own_moves - (2*opp_moves))

# Here I am using the euclidean distance. I first
# transform the tuple into an array and calculate
# the distance between the players
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # own_moves = len(game.get_legal_moves(player))
    # opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    my_position = game.get_player_location(player)
    opponent_position = game.get_player_location(game.get_opponent(player))
    import numpy as np
    return float(np.linalg.norm(np.asarray(my_position)-np.asarray(opponent_position)))

# In this case I use the Manhattan distance which is
# simpler than the euclidean space distance, but I
# expect to behave a bit better given the squared nature
# of the board on this game
def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    import numpy as np
    my_position = game.get_player_location(player)
    opponent_position = game.get_player_location(game.get_opponent(player))
    import numpy as np
    my_position_np = np.asarray(my_position)
    opponent_position_np = np.asarray(opponent_position)
    return float(abs(my_position_np[0]-opponent_position_np[0]) + abs(my_position_np[1]-opponent_position_np[1]))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # if self.time_left() < self.TIMER_THRESHOLD:
        #     raise SearchTimeout()

        ##########################################
        # BEGINNING OF RECURSIVE MINIMAX         #
        ##########################################
        # get the current player in the board
        current_player = game.active_player

        # define the max function
        def max_value(game, depth):
            # make the check to throw the exception when time is up
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # first termination condition is when the node I am visiting
            # is a terminal node, which we identify when there are no legal
            # moves left
            if len(game.get_legal_moves(current_player)) <= 0:
                return game.utility(current_player)

            # second termination condition is when I have reached the max
            # depth or budget
            if depth == 0:
                # return the score for the current player
                return self.score(game, current_player)

            # do the recursion
            v = float('-inf')
            for move in game.get_legal_moves(current_player):
                forecasted_game = game.forecast_move(move)
                v = max(v, min_value(forecasted_game, depth - 1))
            return v

        # define the min function
        def min_value(game, depth):
            # make the check to throw the exception when time is up
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # first termination condition is when the node I am visiting
            # is a terminal node, which we identify when there are no legal
            # moves left. Get the utility from the perspective of the current
            # player: win, lose, otherwise
            if len(game.get_legal_moves(game.get_opponent(current_player))) <= 0:
                return game.utility(current_player)

            # second termination condition is when I have reached the max
            # depth or budget. Note that we return the score for the current
            # player!
            if depth == 0:
                return self.score(game, current_player)

            # do the recursion
            v = float('inf')
            for move in game.get_legal_moves(game.get_opponent(current_player)):
                forecasted_game = game.forecast_move(move)
                v = min(v, max_value(forecasted_game, depth - 1))
            return v

        # if there are no legal moves left, then return (-1, -1)
        if not game.get_legal_moves(current_player):
            return (-1, -1)
        # # otherwise call the recursive function.
        move_selection = {min_value(game.forecast_move(x), depth - 1): x for x in game.get_legal_moves(current_player)}
        return move_selection[max(move_selection.keys(), key=float)]

        ##########################################
        # BEGINNING OF RECURSIVE MINIMAX         #
        ##########################################

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        current_player = game.active_player
        #
        if not game.get_legal_moves(current_player):
            best_move = (-1, -1)
        else:
            best_move = game.get_legal_moves(current_player)[0]
        # try:
        #     depth_counter = 1
        #     while time_left() > 2:
        #         best_move = self.alphabeta(game, depth_counter)
        #         depth_counter += 1
        #     return best_move
        # except SearchTimeout:
        #     return best_move # Handle any actions required after timeout as needed

        try:
            for i in range(1, 30):
                # I CHANGED THE 0 BY self.TIMER_THRESHOLD IN THE CONDITION BELOW!
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, i)
            return best_move
        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        # return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # if self.time_left() < self.TIMER_THRESHOLD:
        #     raise SearchTimeout()

        ##########################################
        # BEGINNING OF RECURSIVE MINIMAX         #
        ##########################################
        # get the current player in the board
        current_player = game.active_player

        # define the max function
        def max_value(game, depth, alpha, beta):
            # make the check to throw the exception when time is up
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # first termination condition is when the node I am visiting
            # is a terminal node, which we identify when there are no legal
            # moves left
            if len(game.get_legal_moves(current_player)) <= 0:
                return game.utility(current_player)

            # second termination condition is when I have reached the max
            # depth or budget
            if depth <= 0:
                # return the score for the current player
                return self.score(game, current_player)

            # do the recursion
            v = float('-inf')
            for move in game.get_legal_moves(current_player):
                forecasted_game = game.forecast_move(move)
                v = max(v, min_value(forecasted_game, depth - 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            # print(v)
            return v

        # define the min function
        def min_value(game, depth, alpha, beta):
            # make the check to throw the exception when time is up
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # first termination condition is when the node I am visiting
            # is a terminal node, which we identify when there are no legal
            # moves left. Get the utility from the perspective of the current
            # player: win, lose, otherwise
            if len(game.get_legal_moves(game.get_opponent(current_player))) <= 0:
                return game.utility(current_player)

            # second termination condition is when I have reached the max
            # depth or budget. Note that we return the score for the current
            # player!
            if depth <= 0:
                return self.score(game, current_player)

            # do the recursion
            v = float('inf')
            for move in game.get_legal_moves(game.get_opponent(current_player)):
                forecasted_game = game.forecast_move(move)
                v = min(v, max_value(forecasted_game, depth - 1, alpha, beta))

                if v <= alpha:
                    return v
                beta = min(beta, v)
            # print(v)
            return v

        # if there are no legal moves left, then return (-1, -1)
        if not game.get_legal_moves(current_player):
            return game.get_player_location(current_player)

        # otherwise call the recursive function.

        # Initialize the variables
        # https://github.com/aimacode/aima-python/blob/master/games.py
        best_score = float('-inf')
        beta = float('inf')
        # need to initialize the best action in both the get_move()
        # and the alphabeta method. In this case I initialize it with
        # a (-1, -1) in case it is a terminal node, or a "random" action
        # from the ones that are available.
        if not game.get_legal_moves(current_player):
            best_action = (-1, -1)
        else:
            best_action = game.get_legal_moves(current_player)[0]

        for move in game.get_legal_moves(current_player):
            v = min_value(game.forecast_move(move), depth - 1, best_score, beta)
            if v > best_score:
                best_score = v
                best_action = move
        return best_action
        ##########################################
        # BEGINNING OF RECURSIVE MINIMAX         #
        ##########################################