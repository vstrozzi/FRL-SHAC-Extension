# SHAC-EXTENSION

This repository contains extensions to the implementation for the paper [Accelerated Policy Learning with Parallel Differentiable Simulation](https://short-horizon-actor-critic.github.io/) (ICLR 2022). \
Since our extensions are loosely related (fork) to the original repository, refer to their README.md for further information.

# Abstact

This paper presents modifications and extensions to the SHAC (Short-Horizon Actor-Critic) policy learning algorithm [SHAC](#), with the aim of improving its performance in reinforcement learning tasks. By incorporating alpha-order gradient estimators [AlphaGrad](#) and a TD($\lambda$) formulation [Sutton](#) in the policy loss, our objective is to improve sample efficiency, reduce the variance of stochastic gradients, and address the vanishing and exploding gradient problem. We implement our modifications and comparisons using the differentiable simulator code from the original SHAC paper [DiffRL](#). Lastly, we compare our implementation in four different complexity robotic control problems: *Ant*, *Hopper*, *Humanoid*, from the high-performance implementations RL games [RLGames](#), and *Muscle Tendon Humanoid*, from [Humanoid MTU](#), with the vanilla SHAC algorithm.

## Test Examples

To test any of the examples, please run the code in _google_colab_text.ipynb_ on Google Colab. This, together with original README.md should suffice to provide all the necessary information on the code's execution.
