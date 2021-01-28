from quzero import play_game, run_selfplay, make_tictactoe_config, Network, SharedStorage, ReplayBuffer
import tensorflow as tf
from tqdm import trange
from quzero.training import update_weights


if __name__ == "__main__":

    #in helpers, returns a MuZero congiguration object
  config = make_tictactoe_config(training_steps=1000)

  network = Network() #a NN object def in helpers

  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(100):
    run_selfplay(config, storage, replay_buffer, 20) #plays 20 games
    learning_rate = config.lr_init #* config.lr_decay_rate #** (tf.train.get_global_step() / config.lr_decay_steps)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, momentum=config.momentum)

    for i in trange(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network) 
