from __future__ import annotations #for returning current Type
import typing
from typing import Dict, List, Tuple, Optional, TypeVar, Union
import collections
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import copy

MAXIMUM_FLOAT_VALUE = float("inf")

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])




###############################################################################
###############################################################################
###############################################################################
class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value




###############################################################################
###############################################################################
###############################################################################
class MuZeroConfig:
    """
    Parameters
    ----------
    training_steps : int
        The number of times update_weights is called per call to train_network
    """
    def __init__(
        self,
        action_space_size: int,
        max_moves: int, #after this, you loose the game
        discount: float, #for reward
        dirichlet_alpha: float, #for exploration, controls how much you explore vs how much you exploit
        num_simulations: int, #how many times you explore the tree
        batch_size: int,
        td_steps: int, # == max_moves ???
        num_actors: int, #parallel processes, 3k in google's paper
        lr_init: float,
        lr_decay_steps: float,
        visit_softmax_temperature_fn: float, #tuning how much we care about big values over small ones, raising result of MCTS to a certain power, do we keep small probabilities? do we give them importance? somehow a trust parameter on large probas...
        training_steps: int = 1e6,
        known_bounds: Optional[KnownBounds] = None,
    ):
        # general main config for MuZero, no matter the game

        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25 #adding randomness to explo

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(training_steps)
        self.checkpoint_interval = int(1e3) #after how many games you save
        self.window_size = int(1e2) # 1e2 'fresh' games how many past games we consider to learn on
        self.batch_size = batch_size
        self.num_unroll_steps = 5 #look-ahead in future
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9 #rms_prop

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_tictactoe_config(training_steps) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps): #interesting ???
        if num_moves < 5:
            return 1.0
        else:
            return 1.0  # Play according to the max. #??? change this?! random?!

    return MuZeroConfig(
        action_space_size=9,
        max_moves=9,
        discount=1,
        dirichlet_alpha=0.1,
        num_simulations=50,
        batch_size=1,
        td_steps=9,  # max_moves
        num_actors=2,
        lr_init=0.0001,
        lr_decay_steps=10e3, #tune according to game complexity, after how many steps you start decaying ???
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=None,
        training_steps=training_steps
    )




###############################################################################
###############################################################################
###############################################################################
class Action:
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __index__(self):
        return self.index




###############################################################################
###############################################################################
###############################################################################
class Player:  # TODO: this???
    pass




###############################################################################
###############################################################################
###############################################################################
class NetworkOutput(typing.NamedTuple):
    value: tf.Tensor
    reward: tf.Tensor
    policy_logits: tf.Tensor #output of network,not yet a proba dist action / how likely it is to be the best action to take
    hidden_state: tf.Tensor
    #value: float
    #reward: float
    #policy_logits: Dict[Action, float] #NOTE why a dict?
    #hidden_state: List[float]          #NOTE why a list?





###############################################################################
###############################################################################
###############################################################################
class Network(object):
    def __init__(self):
        self.pol1 = layers.Dense(64, activation="relu", name="pol1")
        self.pol2 = layers.Dense(32, activation="relu", name="pol2")
        self.pol3 = layers.Dense(9, activation="softmax", name="pol3")  # not logits???
        self.rew1 = layers.Dense(64, activation="relu", name="rew1")
        self.rew2 = layers.Dense(9, activation="relu", name="rew2")
        self.rew3 = layers.Dense(1, name="rew3")
        self.val1 = layers.Dense(64, activation="relu", name="val1")
        self.val2 = layers.Dense(9, activation="relu", name="val2")
        self.val3 = layers.Dense(1, name="val3")
        self.dyn1 = layers.Dense(64, activation="relu", name="dyn1")
        self.dyn2 = layers.Dense(64, activation="relu", name="dyn2")
        self.dyn3 = layers.Dense(64, name="dyn3") #updates inetrenal state representation
        self.repr1 = layers.Dense(9, activation="relu", name="repr1") #9 = tictactoe board
        self.repr2 = layers.Dense(32, activation="relu", name="repr2")
        self.repr3 = layers.Dense(64, name="repr3") #expanding repr from 9 to 64, 64 will be input for others
        self.steps = 0
        self.layers = (
            self.pol1,
            self.pol2,
            self.pol3,
            self.rew1,
            self.rew2,
            self.rew3,
            self.val1,
            self.val2,
            self.val3,
            self.dyn1,
            self.dyn2,
            self.dyn3,
            self.repr1,
            self.repr2,
            self.repr3,
        )

    def initial_inference(self, image) -> NetworkOutput:
        """
        Takes an observation from the env (board, unitary, screenshot, etc) and outputs a network output object
        """
        hidden_state = self.repr3(self.repr2(self.repr1(tf.convert_to_tensor(image, dtype=tf.float32)[None, :])))
        policy_logits = tf.math.log(self.pol3(self.pol2(self.pol1(hidden_state))) )  # logits after log
        value = self.val3(self.val2(self.val1(hidden_state)))
        return NetworkOutput(
            tf.squeeze(value),
            0, #rwd
            tf.squeeze(policy_logits),
            tf.squeeze(hidden_state),
        )

    #??? might have t change formulas for our unitary purpose!
    def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
        x = tf.concat([hidden_state, tf.one_hot(action.index, depth=9)], axis=0)[None,:] #input to dynamics, we pass state and action
        hidden_state = self.dyn3(self.dyn2(self.dyn1(x)))
        policy_logits = tf.math.log(self.pol3(self.pol2(self.pol1(hidden_state)))) # logits after log
        value = self.val3(self.val2(self.val1(hidden_state)))
        #what is the reward??? how do we compute it???
        return NetworkOutput(
            tf.squeeze(value),
            0, #wth, why fixed to 0???
            tf.squeeze(policy_logits),
            tf.squeeze(hidden_state),
        )
        # return NetworkOutput(
        #     0,
        #     tf.squeeze(reward),
        #     {Action(i): p for i, p in enumerate(tf.squeeze(policy_logits))},
        #     [h for h in hidden_state],
        # )

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.extend(layer.weights)
        return weights

    def update_steps(self) -> None:
        self.steps += 1

    def training_steps(self) -> int:
        """
        How many steps / batches the network has been trained for
        """
        return self.steps





###############################################################################
###############################################################################
###############################################################################
class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1 #paper has it same but what is it???
        self.prior = prior #probability to pick that node from parent?
        self.value_sum = 0
        self.children: Dict[Action, Node] = {}
        self.hidden_state = None
        self.reward = 0 #??? when is it updated? in self.play expand node

    def expanded(self) -> bool:
        return bool(self.children) #empty or not?

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count #??? I'm unclear abt this





###############################################################################
###############################################################################
###############################################################################
class ActionHistory:
    """
    Simple history container used inside the search.
    Only used to keep track of the actions executed.

    Attributes
    ----------
    history : list of lists of actions
        Contains a list in which each item is a sequential list of actions taken / a trajectory / a game ???
    action_space_size : int
        The number of actions available.

    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def __repr__(self) -> str:
      return str([action.index for action in self.history])

    def clone(self) -> None:
        return ActionHistory(self.history, self.action_space_size)
        #return ActionHistory(copy.deepcopy(self.history), self.action_space_size)

    def add_action(self, action: Action) -> None:
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:  # NOTE: what's this for???
        return Player()





###############################################################################
###############################################################################
###############################################################################
class Environment:
    #slicing state which is a list
    #rows
    x1 = slice(0, 3)
    x2 = slice(3, 6)
    x3 = slice(6, 9)
    #cols
    y1 = slice(0, 9, 3) #0 to 8 in steps of 3
    y2 = slice(1, 9, 3)
    y3 = slice(2, 9, 3)
    #diagonals
    d1 = slice(0, 9, 4)
    d2 = slice(2, 7, 2)
    wins = [x1, x2, x3, y1, y2, y3, d1, d2] #configs where you would have marked points

    def __init__(self):
      self.state_history = []
      self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
      self.turn = 1
      self.save_state()

    def draw(self):
      return self.reward() == 0 and all([b != 0 for b in self.state]) #no reward AND notExist b=0 in state, a.k.a all has been played

    def reward(self):
      if any(sum(self.state[w]) == 3 for w in self.wins) or any(sum(self.state[w]) == -3 for w in self.wins): #one player is 1 the other is -1
        return 1 #do we retunr the same for both??? change?
      else:
        return 0

    def legal_actions(self) -> List[Action]:
      return [Action(i) for i, b in enumerate(self.state) if b == 0] #b==0 value in the list

    def isLegal(self, action: Action) -> bool:
      return self.state[action] == 0

    def step(self, action: Action) -> int:
      if self.isLegal(action):
        self.state[action] = self.turn
        self.turn *= -1
        self.save_state()
        return self.reward()
      else:
          raise RuntimeError("Agent tried to make an illegal move.")


    def save_state(self):
        self.state_history.append(list(self.state))





###############################################################################
###############################################################################
###############################################################################
class Game(object):
    """
    A class to represent a single episode of interaction with the environment.
    Also referred to as "trajectory"

    Attributes
    ----------
    environment : Environment object
        The environment in which the game takes places.
    history : list of actions
        The sequential list of all actions taken during that particular episode of the game.
    rewards : list of int
        The sequential list of all actions taken during that particular episode of the game.
    child_visits : lost of ???
        ???
    root_values : list of ???
        ???
    action_space_size : int
        Number of possible actions?
    discount : float
        The discount factor (gamma) for TD learning
    """


    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount


    def terminal(self) -> bool:
        """
        Informs whether the state is terminal or not.

        Returns
        -------
        bool
            True if the reward is non-zero ???!!! or the game has ended in draw.
        """
        return self.environment.reward() != 0 or self.environment.draw()


    def legal_actions(self) -> List[Action]:
        """
        Getter. Provides the list of actions which are feasible in this
        environment.

        Returns
        -------
        list of actions
            The list of legal actions in this environment.
        """
        return self.environment.legal_actions()


    def apply(self, action: Action) -> None:
        """
        Performs an action with environment.step, which therefore grants a reward.
         Both reward and action are added to their respective histories.

        Parameters
        ----------
        action : Action
            The action to be applied
        """
        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)


    #wth???
    def store_search_statistics(self, root: Node) -> None:
        """
        Keeps track of the number of times children were visited from their parent node???

         Parameters
         ----------
         root : Node
            ???

         """
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size)) #list of action objects

        #list containing fraction of visits per children or 0  for a in action space if a in children then xxx else 0
        new_list = [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space]
        self.child_visits.append(new_list) #append a list to that list of lists
        self.root_values.append(root.value())


    def make_image(self, state_index: int): #obs of env
        """
        Returns the state_indexth board state from the state history.

         Parameters
         ----------
         state_index : int
            The index of the board's state we are interested in within state_history.

         Returns
         ----------
        ???
            The board state indexed by state_index.
         """
        return self.environment.state_history[state_index]


    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player): #??? critical but given
        """
        !!!Potential algorithmical error bomb!!!
        ???
        The value target is the discounted root value of the search tree N steps
        into the future, plus the discounted sum of all rewards until then.


         Parameters
         ----------
         state_index : int
            ???
         num_unroll_steps : int
            Number of steps which must be unrolled/played into the future ???
         td_steps : int
            Number of TD learning steps to perform, maximal number of moves for
             MC return. Note that for board games, td_steps=max_moves.
         to_play : Player


         Returns
         ----------
        ???
            ???
         """

        targets = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            if bootstrap_index < len(self.root_values): #you never come here in board games
                value = self.root_values[bootstrap_index] * self.discount**td_steps

            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards): #not first nor last move
                last_reward = self.rewards[current_index - 1]

            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))

            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, self.child_visits[-1]))#[])) #NOTE this was changed
                #targets.append((0, last_reward, [])) #originally

        return targets


    def to_play(self) -> Player: #this should be something else!
        """
        ???

        returns
        ----------
        Player
            ???
        """
        return Player()


    def action_history(self) -> ActionHistory:
        """
        ???

        Returns
        ----------
        ActionHistory
            ???
        """
        return ActionHistory(self.history, self.action_space_size)


    def __repr__(self) -> str:
        return str(self.action_history()) + f' rew = {self.environment.reward()}'





###############################################################################
###############################################################################
###############################################################################
class ReplayBuffer:
    """
    A class representing a replay buffer where the trajectory data is stored
    after the end of an episode. Trajectories are selected by sampling a state
     from any game in this replay buffer, then unrolling from that state.
    In board games the training job keeps an in-memory replay buffer of the
    most recent 1 million games received; in Atari, where the visual
    observations are larger, the most recent 125 thousand sequences of length
    200 are kept.

    Attributes
    ----------
    window_size : int
        Maximum number of games/trajectories to be kept in the replay buffer.
    batch_size : int
        ???
    buffer : list of games???trajectories???
        Contains all the sample games/trajectories ??? which can be sampled
        from.
    """


    def __init__(self, config: MuZeroConfig):
        """
        Parameters
        ----------
        config : MuZeroConfig
            The configuration of the MuZero agent.
        """
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []


    def save_game(self, game) -> None:
        """
        Saves the last game/trajectory played in the replay buffer's last index
        while making sure there are never more than window-size
        games/trajectories saved.

        Parameters
        ----------
        game : ???
            the game/trajectory to save.
        """
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)


    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        """
        Saves the last game/trajectory played in the replay buffer's last index
        while making sure there are never more than window-size
        games/trajectories saved.

        Parameters
        ----------
        game : ???
            the game/trajectory to save.

        Returns
        -------
        list of triplets
            Each triplet containing:
                - The state of the environment at that point in time
                - A slice of the game history from the randomly chosen point
                 in time to that point in time + the unrolled future.
                - A triplet made of:
                    - The TD value of the state
                    - The latest reward received ???
                    - A list of frequency visits per children of ???
        """

        #pick (at random) as many games from the buffer as dictated per batchsize
        games = [self.sample_game() for _ in range(self.batch_size)]

        #tuples of game and a move/action index in that game
        game_pos = [(g, self.sample_position(g)) for g in games]

        result = [  (g.make_image(i), g.history[i : i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play()), )
                   for (g, i) in game_pos ]
        return result


    def sample_game(self) -> Game:
        """
        Samples a game/trajectory ??? at random from the replay buffer.

        Returns
        -------
        int
            The random index.
        """
        rd_idx = randint(0, len(self.buffer) - 1) #why -1? ub is already excluded???
        return self.buffer[rd_idx]


    #wth does this do?!???
    def sample_position(self, game) -> int:
        """
        Pick at random the index of an action in the game history (list of actions). ???

        Returns
        -------
        int
            The random index from the game history.
        """
        return randint(0, len(game.history) - 2)  # NOTE: -2? or -1? ???


    def __len__(self) -> int:
        """
        Gives the length of the replay buffer.

        Returns
        -------
        int
            The length of the current buffer, a.k.a. the number of trajectories
            which have been strored up to now.
        """
        return len(self.buffer)





###############################################################################
###############################################################################
###############################################################################
class SharedStorage(object):
    """
    A class used to represent a shared storage containing networks.

     Attributes
    ----------
        _networks : dictionary of Network objects indexed by step count (int)
        _networks[some_step] = some_Network
    """

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        """
        Returns the network corresponding to the last one used,
        or a new network object if there is no previous network

        Returns
        -------
        Network
            The network with more training steps if there is one, or a
            newly created network.
        """
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0 why does it return a new network???
            return Network() #why don't we save that new network???


    def save_network(self, step: int, network: Network) -> None:
        """
        Saved the network in the SharedStorage object, indexing by the step as
        key

        Parameters
        ----------
        step : int
            The training step the main algorithm is at ???
        network : Network
            The network to be added to the self storage
        """
        self._networks[step] = network
