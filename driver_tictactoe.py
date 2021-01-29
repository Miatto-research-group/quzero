from quzero import play_game, run_selfplay, make_tictactoe_config, Network, SharedStorage, ReplayBuffer, train_network
import tensorflow as tf
from tqdm import trange
from quzero.training import update_weights


if __name__ == "__main__":

    #in helpers, returns a MuZero congiguration object
    config = make_tictactoe_config(training_steps=100)
    #####################################
    #network = Network() #a NN object def in helpers
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)


    for _ in range(10): #for 10 epochs
        run_selfplay(config, storage, replay_buffer, 10)  # plays 20 games
        tr = train_network(config, storage, replay_buffer)