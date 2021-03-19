## Setup and Preliminaries
You will need to install MuJoCo and some Python packages; see [installation.md]. 
Experiments can take a while to run (a few hours) so start early.

## Imitation Learning (20 points)
The Imitation Learning notebook will guide you through implementing components of basic behavior cloning and the Dagger algorithm. You will be able to train policies that can imitate the behavior of expert policies we have provided you.

## Policy Gradients (30 points)
The Policy Gradient notebook will have you implement components of basic policy gradient algorithms. You will then train policies from scratch in order to maximize rewards in various tasks.

## DQN (30 points)
The DQN notebook will have you implement components of Deep Q Networks, which will a different approach centered around learning Q-values, rather than explicitly optimizing a policy on the returns directly. 

## Actor Critic (20 points)
In the Actor Critic notebook, we will explore a hybrid approach that learns Q-values for the current policy and uses the learned Q-values in the policy gradient updates. 

## Submitting the Assignment
Run the collectSubmission.sh script to create a zip file for your submission and submit to Gradescope. If, for some reason, the script does not work for you, you can manually zip the deeprl folder, the logs folder, and your Jupyter notebooks and submit them.


