# RL Assignment 1

| Notebook | Assignment part | How to run |
|---|---|---|
| `shuttle_mdp.ipynb` | 2: value and policy iteration (shuttle dispatching) | Jupyter or Colab, Run all. Needs numpy and matplotlib; under a minute. Figures go to `figures/`. |
| `cartpole_policy_gradient.ipynb` | 3: policy gradient on CartPole-v1, Parts 1 and 2 | Colab (CPU runtime), Runtime → Run all; about 10–15 minutes. Results go to Google Drive `MyDrive/rl_hw1_cartpole/`. |
| `pong_policy_gradient.ipynb` `pong_policy_gradient_MEAN.ipynb`| 3: policy gradient on Pong-v5, Parts 1 and 2 | Colab with many CPU cores. Set `BASELINE` in step 4 (`"none"` = Part 1, `"mean"` = Part 2), then Run all. Training saves to `MyDrive/rl_hw1_pong/` and resumes after disconnects; stop the training cell when done, then run steps 13–15. |

Validation tests are separate notebook cells that run with Run all; each prints its prediction next to its result. The report explains each test.
