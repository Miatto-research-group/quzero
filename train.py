from traceback_with_variables import activate_by_import
from quzero import train, make_tictactoe_config
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = make_tictactoe_config(training_steps = args.epochs)
    latest_network = train(config)
    print(latest_network)
