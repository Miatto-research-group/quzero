import typing
from typing import Dict, List, Tuple, Optional
import collections
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MAXIMUM_FLOAT_VALUE = float("inf")

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


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


class MuZeroConfig:
    def __init__(
        self,
        action_space_size: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        num_actors: int,
        lr_init: float,
        lr_decay_steps: float,
        visit_softmax_temperature_fn,
        training_steps: int = 1e6,
        known_bounds: Optional[KnownBounds] = None,
    ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

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
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e2) # 1e2 'fresh' games 
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_tictactoe_config(training_steps) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 5:
            return 1.0
        else:
            return 0.0  # Play according to the max.

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
        lr_decay_steps=10e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=None,
        training_steps=training_steps
    )


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


class Player:  # TODO: this?
    pass


class NetworkOutput(typing.NamedTuple):
    value: tf.Tensor
    reward: tf.Tensor
    policy_logits: tf.Tensor
    hidden_state: tf.Tensor
    #value: float
    #reward: float
    #policy_logits: Dict[Action, float] #NOTE why a dict?
    #hidden_state: List[float]          #NOTE why a list?


class Network(object):
    def __init__(self):
        self.pol1 = layers.Dense(64, activation="relu", name="pol1")
        self.pol2 = layers.Dense(32, activation="relu", name="pol2")
        self.pol3 = layers.Dense(9, activation="softmax", name="pol3")  # not logits?
        self.rew1 = layers.Dense(64, activation="relu", name="rew1")
        self.rew2 = layers.Dense(9, activation="relu", name="rew2")
        self.rew3 = layers.Dense(1, name="rew3")
        self.val1 = layers.Dense(64, activation="relu", name="val1")
        self.val2 = layers.Dense(9, activation="relu", name="val2")
        self.val3 = layers.Dense(1, name="val3")
        self.dyn1 = layers.Dense(64, activation="relu", name="dyn1")
        self.dyn2 = layers.Dense(64, activation="relu", name="dyn2")
        self.dyn3 = layers.Dense(64, name="dyn3")
        self.repr1 = layers.Dense(9, activation="relu", name="repr1")
        self.repr2 = layers.Dense(32, activation="relu", name="repr2")
        self.repr3 = layers.Dense(64, name="repr3")
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
        hidden_state = self.repr3(self.repr2(self.repr1(tf.convert_to_tensor(image, dtype=tf.float32)[None, :])))
        policy_logits = self.pol3(self.pol2(self.pol1(hidden_state)))
        value = self.val3(self.val2(self.val1(hidden_state)))
        return NetworkOutput(
            tf.squeeze(value),
            0,
            tf.squeeze(tf.math.log(policy_logits)), # logits after log
            tf.squeeze(hidden_state),
        )

    def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
        x = tf.concat([hidden_state, tf.one_hot(action.index, depth=9)], axis=0)[None,:]
        hidden_state = self.dyn3(self.dyn2(self.dyn1(x)))
        policy_logits = self.pol3(self.pol2(self.pol1(hidden_state)))
        reward = self.rew3(self.rew2(self.rew1(hidden_state)))
        return NetworkOutput(
            0,
            tf.squeeze(reward),
            tf.squeeze(tf.math.log(policy_logits)), # logits after log
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

    def training_steps(self) -> int:
        return self.steps


class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[Action, Node] = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory:
    """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def __repr__(self):
      return str([action.index for action in self.history])

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:  # NOTE: what's this for?
        return Player()


class Environment:
    x1 = slice(0, 3)
    x2 = slice(3, 6)
    x3 = slice(6, 9)
    y1 = slice(0, 9, 3)
    y2 = slice(1, 9, 3)
    y3 = slice(2, 9, 3)
    d1 = slice(0, 9, 4)
    d2 = slice(2, 7, 2)
    wins = [x1, x2, x3, y1, y2, y3, d1, d2]

    def __init__(self):
      self.state_history = []
      self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
      self.turn = 1
      self.save_state()

    def draw(self):
      return self.reward() == 0 and all([b != 0 for b in self.state])

    def reward(self):
      if any(sum(self.state[w]) == 3 for w in self.wins) or any(sum(self.state[w]) == -3 for w in self.wins):
        return 1
      else:
        return 0

    def legal_actions(self) -> List[Action]:
      return [Action(i) for i, b in enumerate(self.state) if b == 0]

    def isLegal(self, action: Action) -> bool:
      return self.state[action] == 0

    def step(self, action: Action):
      if self.isLegal(action):
        self.state[action] = self.turn
        self.turn *= -1
        self.save_state()
        return self.reward()
      return -1

    def save_state(self):
        self.state_history.append(list(self.state))


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        return self.environment.reward() != 0 or self.environment.draw()

    def legal_actions(self) -> List[Action]:
        return self.environment.legal_actions()

    def apply(self, action: Action):
        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space]
        )
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        return self.environment.state_history[state_index]

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, self.child_visits[-1]))#[])) #NOTE this was changed
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def __repr__(self):
        return str(self.action_history()) + f' rew = {self.environment.reward()}'


class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        # print(f"replay buffer len={len(self.buffer)}")

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [
            (
                g.make_image(i),
                g.history[i : i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]

    def sample_game(self) -> Game:
        return self.buffer[randint(0, len(self.buffer) - 1)]  # random

    def sample_position(self, game) -> int:
        return randint(0, len(game.history) - 2)  # NOTE: -2? or -1?

    def __len__(self):
      return len(self.buffer)


class SharedStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return Network()#make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def make_uniform_network():
    return Network()