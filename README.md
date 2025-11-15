# Multi-agent RL for robot scheduling at intersections 


# Framework1- Deep reinforcement learning aided distributed MPC for warehouse intersection of multi-agent system:
The work involves the use RL to aid the planning performed by the Model Predictive
Control(MPC) for an un-signalised intersection problem. The DDPG agent is used to coordinate the different robots to
cross the intersection safely and optimally overcoming the use of unscalable mixed integer problem. Based on the
sequence from the RL the planning with formal safety guarantees is performed. Further, to reduce the online
computational cost of MPC two different methods are used 1) The planning horizon is made shorter and RL is used to
estimate the terminal value function that posses the global information 2) The optimisation is warm-started with good
initial guess from the RL agent .

# Framework2- Attention based Safe reinforcement learning for intersection management of CAVs using action projection: 
Instead of using RL to solve multi-agent coordination, it was directly used to decide control
variables for each agent in a shared policy setting. The RL outputs were then mapped to a safe control set estimated at
each time step. This approach resulted in RL addressing a longer horizon problem. However, as the number of agents
increased, the optimality gap in RL worsened(120 agents). To overcome this sub-optimal policy, the latent state and
observation representations were fed to the RL using a bilinear LSTM with an attention mechanism. Reduced the
computation by huge margin in comparison with framework1.


https://docs.google.com/file/d/1LJUTGGWwl2RVPKT4519vmkfW_i12NGok/preview
