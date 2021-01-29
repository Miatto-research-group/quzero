from tqdm import trange
import time
import tensorflow as tf
from .helpers import SharedStorage, ReplayBuffer, MuZeroConfig, Network
from .selfplay import run_selfplay


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.



def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer) -> None:
  """
  Performs the training of the network.

  Parameters
  ----------
  config : MuZeroConfig
    The configuration of the MuZero agent.
  storage : SharedStorage
    A shared object containing networks.
  replay_buffer : ReplayBuffer
   The replay buffer of ???
  """
  print('##### START TRAINING #####')
  network = Network()
  #TODO fix the learning rate issue, here it doesn't change!
  learning_rate = config.lr_init #* config.lr_decay_rate ** (tf.train.get_global_step() / config.lr_decay_steps)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate, momentum=config.momentum) #this is not deepmind' opti, why???

  for i in trange(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)



def scale_gradient(tensor, scale):
  """Scales the gradient for the backward pass."""
  return tf.convert_to_tensor(tensor) * scale + tf.stop_gradient(tensor) * (1 - scale)



def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
  loss = 0
  with tf.GradientTape() as tape:
    for image, actions, targets in batch:
      # Initial step, from the real observation.
      value, reward, policy_logits, hidden_state = network.initial_inference(image)
      predictions = [(1.0, value, reward, policy_logits)]

      # Recurrent steps, from action and previous hidden state.
      for action in actions:
        value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
        predictions.append((1.0 / len(actions), value, reward, policy_logits))

        hidden_state = scale_gradient(hidden_state, 0.5)

      #potential bomb!!! comparing what tree says, what we say and we want them
      for prediction, target in zip(predictions, targets):
        gradient_scale, value, reward, policy_logits = prediction
        target_value, target_reward, target_policy = target
        l = (
            scalar_loss(value, target_value)
            + scalar_loss(reward, target_reward)
            + tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=target_policy)
        )

        loss += scale_gradient(l, gradient_scale)

    all_weights = network.get_weights()
    for weights in all_weights:
        loss += weight_decay * tf.nn.l2_loss(weights)
  
  grad = tape.gradient(loss, all_weights)
  optimizer.apply_gradients(zip(grad, all_weights))
  network.steps += 1
  # if network.steps % 100 == 0:
  #   print([action.index for action in actions])



def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return 0.5 * (prediction - target) ** 2