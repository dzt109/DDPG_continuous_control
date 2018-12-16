[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: results_plot.png
[image4]: DDPG2.png


# Solution for the `Reacher` Environment using Deep Deterministic Policy Gradients (DDPG)

![Trained Agent][image1]

## Introduction

The standard Deep Q Network is limited to domains with finite dimensional (discrete) action spaces. In case the action space is continuous, it has to be discretized. However, discretizing the action space too coursely will not yield good training outcomes, while discretizing the space too finely will lead to the curse of dimensionality. Google Deepmind has developed a number of policy gradient based approaches to tackle this class of problems. The policy gradient methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to ***θ***, ***π***_***θ*** ( ***a*** | ***s***). The value of the reward (objective) function depends on this policy and then various algorithms, such as gradient ascent, can be applied to optimize ***θ*** for the best reward [[2]](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#what-is-policy-gradient).

Computing the gradient of the reward function is tricky because it depends on both the action selection and the stationary distribution of states following the target selection behavior. Given that the environment is generally unknown, it is difficult to estimate the effect on the state distribution by a policy update. The *policy gradient theorem* allows us to rewrite the derivative of the objective function to not involve the derivative of the state distribution, which allows the gradient to be computed.

Several policy gradient algorithms can be used, such as REINFORCE (Monte-Carlo policy gradient) and vanilla Actor-Critic. Both these are *on-policy* algorithms -- training samples are collected according to the target policy — the very same policy that we try to optimize for. This could lead to instabilities. In comparison, off-policy methods allow more robust exploration of the space, and can use incomplete trajectories, and even past episodes ("*experience replay.*") 

#### Actor Critic Methods

Actor Critic methods are a class of methods that allow the learning of the value function in addition to the policy. Knowing the value function can assist the policy update, such as by reducing gradient variance in vanilla policy gradients [[3]](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#what-is-policy-gradient).

The actor-critic methods consist of two models, which may optionally share parameters. **Critic** updates the value function parameters. **Actor** updates the policy parameters in the direction suggested by the critic. The two models can have different learning rates. 

Following are some of the Actor-Critic methods considered in this project [[3]](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#what-is-policy-gradient): 

1. In the **Asynchronous Advantage Actor-Critic** (A3C) method, the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time, which is ideal for parallel training.

2. **A2C** is a synchronous, deterministic version of A3C. Unlike in A3C, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster.

3. Unlike the above two methods in which the policy function is modeled as a probability distribution over actions given the current state, **Deterministic policy gradient** (**DPG**) instead models the policy as a deterministic decision. 

4. **Deep Deterministic Policy Gradient** (DDPG), is a model-free off-policy actor-critic algorithm, combining DPG with DQN. Like DQN, it uses experience replay and the frozen target network to stabilize learning. Unlike DQN, DDPG works in continuous space with the actor-critic framework while learning a deterministic policy.
5. The **Distributed DDPG** (D4PG) applies a set of improvements on DDPG to make it run in the distributional fashion.

6. **Multi-agent DDPG** (MADDPG) extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. Eacg agent views the environment as non-stationary, since policies of other agents are quickly upgraded and remain unknown.
7. **Trust region policy optimization** (TRPO) improves training stability by enforcing a KL divergence constraint on the size of policy update at each iteration. 

8. **Proximal policy optimization** (PPO) achieves the same ends as TRPO using a simpler clipped surrogate objective while retaining similar performance.
9. **Actor-critic with experience replay** (ACER) is an off-policy actor-critic model with experience replay. Off-policy experience replay increases stability, sample efficiency and decreases data correlation.  

DDPG was chosen to solve the reacher environment, and is explained in more detail in the following section. 



### Deep Deterministic Policy Gradients (DDPG)

Deep Deterministic Policy Gradients is an Actor-Critic based Reinforcement learning algorithm for continuous action [[4]](https://arxiv.org/abs/1509.02971). This is an adaptation of the ideas underlying the success of Deep Q-Learning to the continuous action domain.  


The following figure ([[5]](https://www.renom.jp/notebooks/tutorial/reinforcement_learning/DDPG/notebook.html)) illustrates the DDPG algorithm.
 
![DDPG algorithm][image4]  


The steps of the DDPG algorithm are listed below. ([[5]](https://www.renom.jp/notebooks/tutorial/reinforcement_learning/DDPG/notebook.html)): 

1. Two Neural Networks , Actor ***μ*** and Critic ***Q*** are initialized.
2. Two other neural networks target actor ***μ′*** and target critic ***Q′***
3. Get initial state ***s***
4. Get action ***a*** = ***μ***(***s***)
5. Take action ***a*** and get reward ***r*** with next state ***s′***
6. Get the value of present state, **value** ( ***s*** , ***a*** )= ***Q*** ( ***s*** , ***a*** )
7. Get target action ***a′*** = ***μ′*** ( ***s′*** ) and target critic value of next state ***Q′*** ( ***s′*** , ***a′*** )
8. Get value of current state from Bellman equation, **value** _ **target** (***s*** , ***a*** ) = ***r*** + ***Q′*** ( ***s′*** , ***a′*** )
9. Get Loss 1/m ∑( **value** _ **target** ( ***s′*** , ***a′*** ) − **value** ( ***s*** , ***a*** ))^2
10. Update Critic Network according to the loss
11. Get Gradient of ***Q*** ( ***s*** , ***a***) with respect to actor network ***μ*** ( ***s*** ), ∂ ***Q*** /∂ ***μ*** ( ***s*** )
12. Get Gradient of ***μ*** ( ***s***) with respect to weights of actor network ***θ*** ***μ***, ∂*** μ*** (***s*** )∂ ***θ*** _ ***μ***
13. Update Critic Network by maximizing the value of ∂ ***Q*** /∂ ***μ*** ( ***s*** ) ∂ ***μ*** ( ***s*** )/∂ ***θ_μ***
14. Update target critic using the equation ***Q′*** ← ***τQ*** + (1− ***τ*** ) ***Q′***
15. Update target actor using the equation ***μ′*** ← ***τμ*** + (1− ***τ*** ) ***μ′***

### Hyperparameters

A brief description of the hyperparameters is provided below. The results section lists out the various combinations of hyperparameters that were tried for building the model. 

- `max_score`: The target average score. This was fixed at 30.   
- `BUFFER_SIZE`: Memory size for experience replay, a random sample of prior actions instead of the most recent action to proceed. 
- `BATCH_SIZE`: Batch size for optimization. Smaller batch size affected convergence stability. Larger batch size made the process very slow.   
- `TAU` : parameter for soft update of target parameters        
- `LR_ACTOR`, `LR_CRITIC`: Learning rate for the above algorithm, for the actor and the critic respectively. Large values are detrimental to getting a good final accuracy.          
- `WEIGHT_DECAY`: Decay rate per iteration of the weights used for learning. 
- `fc1_units`: Number of neurons in the first hidden layer of the neural network. 
- `fc2_units`: Number of neurons in the seocnd hidden layer of the neural network. 


### `Reacher` environment

![Trained Agent][image1]

In the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, two separate versions of the Unity environment are provided:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that the project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Instructions

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 
3. Install missing packages as needed.
4. Run the file `Continuous_Control.ipynb` cell by cell in order to train the agent as well as seeing a trained agent perform. 
5. A list of hyperparameters is provided and can be changed in order to explore training performance for a different set.
6. Models for all hyperparameters as well as a summary of number of steps to convergence is stored in the `results` variable, as well as in the file `results.pickle.`  


## Results

In this project, I solved the 20 agent model. The completed and debugged code, extended to 20 agents was run for a number of hyperparameters to train the model. The table below shows the various hyperparameters that were tested in different models.

For alll the cases, the model ultimately converged to an average score (over the 20 agents) between 30 and 35 within about 10 to 47 episodes, so, the iterations terminated after 110 to 147 episodes. Fastest convergence was achieved in Case 7 (within 7 episodes.) The second best case, Case 3, included a larger `BUFFER_SIZE`, a non-zero `WEIGHT_DECAY`, and the largest CNN model for deep learning, among all the cases tried. This case was computationally less efficient than Case 7, and also comparatively underperformed.   


 
 |Case| BUFFER_SIZE | BATCH_SIZE | TAU  | LR_ACTOR | LR_CRITIC | WEIGHT_DECAY | fc1_units | fc2_units           | Convergence |
 |---|--------------|-------------|-------|-----------|------------|---------------|------------|---------------------|------|
 | 1 | 100000      | 128         | 0.001 | 0.001     | 0.0001     | 0             | 200        | 400                 | 113 |
 | 2 | 100000      | 128         | 0.001 | 0.001     | 0.0001     | 0             | 200        | 400                 | 111 |
 | 3 | 200000      | 64          | 0.005 | 0.0005    | 5e-05      | 0.0001        | 300        | 600                 | 110 |
 | 4 | 100000      | 128         | 0.001 | 0.0003    | 0.0001     | 0             | 150        | 300                 | 114 |
 | 5 | 100000      | 128         | 0.001 | 0.0002    | 5e-05      | 0             | 150        | 300                 | 136 |
 | 6 | 100000      | 128         | 0.001 | 0.0001    | 3e-05      | 0             | 150        | 300                 | 130 |
 | 7 | 100000      | 64          | 0.002 | 0.001     | 0.0001     | 0             | 150        | 300                 | 107 |
 | 8 | 100000      | 64          | 0.002 | 0.001     | 0.0001     | 0             | 200        | 300                 | 112 |
 | 9 | 100000      | 32          | 0.002 | 0.001     | 0.0001     | 0             | 200        | 300                 | 147 |
 | 10 | 100000      | 32          | 0.002 | 0.001     | 0.0001     | 0             | 200        | 300                 | 147 |


The figure below shows the time history of average score over episodes for the 20 agents. 

![Results][image3]


All models are available for download in the models folder. A video of one of the models is provided for illustration [here](https://youtu.be/FnJfkLinb9w). 


## Future Work

Future work includes the following: 

1. Solving other environments: Some of the other environments available, such as the Spider environment was natural extensions of the work presented here and solutions would be attempted as future work using DDPG. 
2. Other Actor-Critic based methods listed in the introduction will be implemented as future work with the Reacher environment. This will allow benchmarking of different algorithms. 
3. A more rigorous hyperparameter study will offer insights into the relative effect of different hyperparameters in learning for DDPG and other methods. 
