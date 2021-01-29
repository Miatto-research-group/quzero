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
  Creates a new network object which is trained on available data for n training steps
  Parameters
  ----------
  config : MuZeroConfig
    The configuration of the MuZero agent.
  storage : SharedStorage
    A shared object containing networks.
  replay_buffer : ReplayBuffer
   The replay buffer of ???
  """
  print('##### START TRAINING #####', flush=True)
  network = Network() #why on earth do we create a new network everytime???
  #TODO fix the learning rate issue, here it doesn't change!
  learning_rate = config.lr_init * config.lr_decay_rate #** (tf.train.get_global_step() / config.lr_decay_steps)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate, momentum=config.momentum) #DIFF the optimizer was changed due to version difference

  for i in trange(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)
  print("NB NW IN STORAGE", len(storage._networks), flush=True)
  #network.update_steps()





def scale_gradient(tensor:tf.python.framework.ops.EagerTensor, scale:float) -> tf.python.framework.ops.EagerTensor:
  """
  Scales the gradient for the backward pass.

  Parameters
  ----------
  tensor : tensorflow.python.framework.ops.EagerTensor
    ???
  scale : float

  Returns
  -------
  tensorflow.python.framework.ops.EagerTensor
    The scaled-down version of the input tensor ???
  """
  res = tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
  return res



def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch, weight_decay: float):
  """
  Updates the weights of the network based on gradient optimisation.

  Parameters
  ----------
  optimiser : tf.keras.optimizers.Optimizer
    The optimiser to use for the weight updates task.
  network : Network
    The network on which to perform the weight updates.
  batch :

  weight_decay : float


  Returns
  -------
  tensorflow.python.framework.ops.EagerTensor
    The scaled-down version of the input tensor ???
  """
  #print(type(network))
  #print(type(batch))
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
  network.update_steps()
  print(network.training_steps())



def scalar_loss(prediction, target) -> float:
  """
  Definition of loss to be refined according to application, for instance
   Deepmind advises MSE in board games, cross entropy between categorical
   values in Atari.

  Parameters
  ----------
  prediction : int
    The value of ??? predicted by ???

  target : int
    The true value of ??? as given by ???

  Returns
  -------
  float
    A
  """
  #print("PRED",type(prediction), flush=True)
  #print("TARt",type(target), flush=True)
  res = 0.5 * (prediction - target) ** 2
  #print("RES",type(res), flush=True)
  return  res