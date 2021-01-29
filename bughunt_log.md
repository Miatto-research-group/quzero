
INTERROGATIONS: IS THIS NORMAL/EXPECTED?
- training/scalar_loss prediction/result are sometimes int (float for res), sometimes tensorflow.python.framework.ops.EagerTensor
- we only ever have one NW saved... with training/train_network, since they each erase the previous one because they always have the same number of training steps, so we re-create a new network each time, never improve on the previous one.


MAJOR BUGS/ISSUES IDENTIFIED
- training_steps update?
- confusion between config.ts & network.ts : config.ts should be fixed (total nb we want) while network.ts should increase (per step we train)
- helpers/recurrent_inference : why are we returning 0 for the nw output, shouldn't be : how can we get the reward there? it's necessary.
- for the nw.training_steps, should we update them in update_weights or train_network? either way there's always only going to be one nw in the storage...

MINOR BUGS/LIMITATIONS IDENTIFIED
- LR in training/train_network is fixed, we should have it decay.


HYPOTHESIS / ANSWER(S)
Is train/training called? -> OK, in driver program.
Does training save the NW? -> OK, 
Exploding weights? -> OK. they seem normal (printed for 10 epochs).
Is config.training_steps correctly updated or are networks erasing each other? -> PB!
Is there a confusion between network.training_steps and config.training_steps? -> PB!
Does the agent get some reward other than 0 at some point? -> PB!
