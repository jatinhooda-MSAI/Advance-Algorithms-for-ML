# HW2: Deep Q-Network

Code and results for Assignment 2: a Deep Q-Network on CartPole-v1 and on MsPacman-v5. Everything was run on deepdish with a single GPU card.

## Setup

Install the packages once:

```
pip install "gymnasium[atari]>=1.1" "ale-py>=0.10" torch matplotlib
```

Before running a notebook, check `nvidia-smi` for a free card and set `GPU_INDEX` in its first code cell.

## How to run

1. **CartPole.** Open `dqn_cartpole-v2.ipynb` and run all cells. It trains three seeds on the CPU (about 10 minutes each). It then picks the best training snapshot of each seed on validation episodes, evaluates that network on 500 test episodes, and runs the tests and the decision-boundary plot. Results are written to `rl_hw2_cartpole_v3/`.
2. **CartPole baseline (optional).** `dqn_cartpole.ipynb` is the first, standard-DQN version, kept for comparison. Run all cells; results go to `rl_hw2_cartpole/`.
3. **MsPacman, training.** In `dqn_mspacman.ipynb`, set `RUN_TRAINING = True` in the settings cell and run all cells. Training saves a checkpoint every 25 episodes and continues from it if the cell is run again. To stop early, create an empty file named `STOP` in the run folder. For a long run you can also convert the notebook with `jupyter nbconvert --to script` and run the script inside `tmux`.
4. **MsPacman, results.** Set `RUN_TRAINING = False` and run all cells again. The notebook loads the saved run and produces the plots, the 500-game evaluation, the tests and the non-standard plots.
5. **Second MsPacman run.** `dqn_mspacman_run2seed1.ipynb` is the same notebook with `SEED = 1`, `RUN_NAME = "dqn_seed1"`, 6,000 episodes and a 300,000-frame buffer. Run it the same way as steps 3 and 4.

The replay buffers (`replay_buffer.npz`, 1.4 to 2.1 GB) are too large for GitHub and are not included. Step 4 does not need them; a resumed training run simply refills its buffer before learning continues.

## Files

| File or folder | What it is |
|---|---|
| `dqn_cartpole-v2.ipynb` | CartPole-v1, final run: Double DQN, 3 seeds of 2,500 episodes, snapshot selection, required plots, evaluation, tests A to G and the decision-boundary plot |
| `dqn_cartpole.ipynb` | CartPole-v1, baseline run with standard DQN, kept for comparison |
| `dqn_mspacman.ipynb` | MsPacman-v5, seed 0 (run `dqn_v1`, 5,000 episodes): training, required plots, evaluation, tests A to D and the non-standard plots |
| `dqn_mspacman_run2seed1.ipynb` | MsPacman-v5, seed 1 (run `dqn_seed1`, 6,000 episodes), same pipeline |
| `rl_hw2_cartpole_v3/` | Outputs of the final CartPole run: figures (`.png`), `decision_boundary.gif`, training histories (`history_*.json`), evaluation scores (`eval_scores.json`, `eval_selected.json`) and network weights (`qnet_*.pt`) |
| `rl_hw2_cartpole/` | The same outputs for the baseline CartPole run |
| `rl_hw2_mspacman/dqn_v1/`, `rl_hw2_mspacman/dqn_seed1/` | Outputs of each MsPacman run, described below |

Each MsPacman run folder contains:

- `checkpoint.pt`: the final network, the target network and the optimizer state.
- `best.pt`: the network with the best 100-episode average.
- `history.json`: per-episode scores and max-Q values.
- `heldout_states.npy`: the 256 fixed frames used for the max-Q curve.
- `eval_final.json`: the 500 evaluation scores.
- `log.txt`: training progress.
- the figures.
