# Extending SHAC with Multiple Variance Reduction Techniques

This repository contains extensions to the SHAC implementation of the paper [Accelerated Policy Learning with Parallel Differentiable Simulation](https://short-horizon-actor-critic.github.io/) (ICLR 2022). \
Since our extensions are loosely related (fork) to the original repository, refer to their README.md for further information. 

# Abstact

This paper presents modifications and extensions to the SHAC (Short-Horizon Actor-Critic) policy learning algorithm, with the aim of improving its performance in reinforcement learning tasks. By incorporating alpha-order gradient estimators and a TD($\lambda$) formulation in the policy loss, our objective is to improve sample efficiency, reduce the variance of stochastic gradients, and address the vanishing and exploding gradient problem. We implement our modifications and comparisons using the differentiable simulator code from the original SHAC paper. Lastly, we compare our implementation in four different complexity robotic control problems: *Ant*, *Hopper*, *Humanoid*, from the high-performance implementations RL games, and *Muscle Tendon Humanoid*, with the vanilla SHAC algorithm.

## Test Examples

To test any of the examples, please run the code in _google_colab_text.ipynb_ on Google Colab. This, together with original README.md should suffice to provide all the necessary information on the code's execution.
