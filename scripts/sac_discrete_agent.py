import rospy
import torch
import numpy as np
from networks_discrete import update_params, Actor, Critic, ReplayBuffer, Dual_ReplayBuffer
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

class DiscreteSACAgent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 gamma=0.99, n_actions=3, buffer_max_size=1000000, tau=0.005,
                 update_interval=1, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 chkpt_dir=None, target_entropy_ratio=0.4):
        self.batch_size = rospy.get_param('rl_control/SAC/batch_size', 1000)
        self.layer1_size = rospy.get_param('rl_control/SAC/layer1_size', 100)
        self.layer2_size = rospy.get_param('rl_control/SAC/layer2_size', 100)
        self.gamma = rospy.get_param('rl_control/SAC/gamma', 100)
        self.tau = rospy.get_param('rl_control/SAC/tau', 100)            
        self.beta = rospy.get_param('rl_control/SAC/beta', 100)
        #self.target_entropy = rospy.get_param('rl_control/SAC/target_entropy_ratio', 100)
        self.target_entropy_ratio= 0.98 ##########################################################this is what we change for the entropy
        self.update_interval = update_interval
        self.buffer_max_size = buffer_max_size
        self.scale = reward_scale
        self.lr = 0.002
        self.input_dims = input_dims[0]
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir
        #self.target_entropy = target_entropy_ratio  # -np.prod(action_space.shape)\######
        # Lists to store entropy and temperature values
        self.entropy_history = []
        self.entropy_loss_history = []
        self.temperature_history = []

        # Initialize history lists for q1, q2, policy loss, and actor loss
        self.q1_history = []
        self.q2_history = []
        self.policy_loss_history = []
        self.actor_loss_history = []  # If you plan to track a separate actor loss
        # Initialize history lists for Q losses
        self.q1_loss_history = []
        self.q2_loss_history = []
        self.policy_history = []
        self.targetqhistory=[]
        self.block_history = []
        self.step_history = []
      
        self.actor = Actor(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(device)
        self.critic = Critic(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(device)
        # The above 3 lines are for the LfD participants gameplay to initialize the dual buffer
        self.demo_data = rospy.get_param("rl_control/Game/load_demonstrations_data_dir","opt/ros/catkin_ws/src/hrc_study_tsitosetal/buffers/demo_buffer.npy")
        self.lfd_participant_gameplay = rospy.get_param('rl_control/Game/lfd_participant_gameplay', False)
        self.percentages =  [ 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # Because we give the first 
        self.target_critic = Critic(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(
            device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.soft_update_target()

        # disable gradient for target critic
        # for param in self.target_critic.parameters():
        #     param.requires_grad = False


        # target -> maximum entropy (same prob for each action)
        # - log ( 1 / A) = log A
        # self.target_entropy = -np.log(1.0 / action_dim) * self.target_entropy_ratio
        #self.target_entropy = np.log(3) * self.target_entropy_ratio
        self.target_entropy = 0.4
        #self.target_entropy = 0.36 * np.log(np.prod(n_actions))

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        #self.alpha = self.log_alpha.exp().detach()
        #the previous line of code is supposed to set alpha=1. But it is not working properly so we just set alpha=1
        #!Attention! alpha is used to calculate entropy but in the first working edition is used as learning rate to actor and as a way to calculate entropy. 
        self.alpha=1.0
        #change: the actor and the critic have the same learning rate beta 
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.beta, eps=1e-8)
        self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=self.beta, eps=1e-8)
        self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=self.beta, eps=1e-8)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=1e-8)


        if self.lfd_participant_gameplay:
            self.memory= Dual_ReplayBuffer(self.buffer_max_size,self.demo_data,self.percentages)
        else:
            self.memory = ReplayBuffer(self.buffer_max_size)
            
    def learn(self, timestep, i_block, interaction=None):

        if interaction is None:
            if self.lfd_participant_gameplay:
                states, actions, rewards, states_, dones= self.memory.sample(self.batch_size,episode_number) # TO_DO add a print line here to make sure % are correct
            else:
                states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, states_, dones = interaction
            states, actions, rewards, states_, dones = [np.asarray([states]), np.asarray([actions]),
                                                        np.asarray([rewards]), np.asarray([states_]),
                                                        np.asarray([dones])]
        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        batch_transitions = states, actions, rewards, states_, dones
        #print('!!!!!!!!!!!!!!!!!!!!!')
        #print('This is bach transitions')
        #print(batch_transitions)
        weights = 1.  # default
        q1_loss, q2_loss, errors, mean_q1, mean_q2, targetq = self.calc_critic_loss(batch_transitions, weights)
        policy_loss,entropies,action_probs = self.calc_policy_loss(batch_transitions, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)


        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        update_params(self.critic_q1_optim, q1_loss)
        update_params(self.critic_q2_optim, q2_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        #for param in self.actor.parameters():
        # print(param)
        update_params(self.actor_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)
        # Save entropy and temperature values
        self.entropy_history.append(entropies.mean().item())
        #print("entropies", entropies.mean().item())

        self.entropy_loss_history.append(entropy_loss.item())
        #print("entropy loss", entropy_loss.item())

        self.temperature_history.append(self.log_alpha.exp().item())
        #print("temp", self.log_alpha.exp())
        #print(mean_q1)
        # Update history lists
        # print("Emean",entropies.mean().item())
        # print("q1mean",mean_q1)
        # print("q2mean",mean_q2)  
        # print("ploss",policy_loss.item())
        # print("q1loss",q1_loss.item())
        # print("q2loss",q2_loss.item())
        # print("entropyloss",entropy_loss.item())
        # print("loga",self.log_alpha.exp().item())
        # #print("policy",action_probs)
        # print("targetq",targetq.mean().item())


        self.q1_history.append(mean_q1)  # Assuming mean_q1 is a tensor
        self.q2_history.append(mean_q2)
        self.policy_loss_history.append(policy_loss.item())
        self.q1_loss_history.append(q1_loss.item())
        self.q2_loss_history.append(q2_loss.item())
        #self.policy_history.append(action_probs.mean().item())
        self.targetqhistory.append(targetq.mean().item() )
        self.block_history.append(i_block)
        self.step_history.append(timestep)
        """
        print("Length of actions prob history:", len(self.policy_history))
        print("Length of temperature history:", len(self.temperature_history))
        print("Length of entropy history:", len(self.entropy_history))
        print("Length of entropy loss history:", len(self.entropy_loss_history))
        print("Length of q1 history:", len(self.q1_history))
        print("Length of q2 history:", len(self.q2_history))
        print("Length of policy loss history:", len(self.policy_loss_history))
        print("Length of q1 loss history:", len(self.q1_loss_history))
        print("Length of q2 loss history:", len(self.q2_loss_history))
        """
        return mean_q1, mean_q2, entropies

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)
            # next_q = (action_probs * (
            #     torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            # )).mean(dim=1).view(self.memory_batch_size, 1) # E = probs T . values

            alpha = self.log_alpha.exp()
            next_q = action_probs * (torch.min(next_q1, next_q2) - alpha * log_action_probs)
            next_q = next_q.sum(dim=1)

            target_q = rewards + (1 - dones) * self.gamma * (next_q)
            return target_q.unsqueeze(1)

        # assert rewards.shape == next_q.shape
        # return rewards + (1.0 - dones) * self.gamma * next_q

    def calc_critic_loss(self, batch, weights):
        target_q = self.calc_target_q(*batch)
        #print(target_q)
        # TD errors for updating priority weights
        #errors = torch.abs(curr_q1.detach() - target_q)
        errors = None
        mean_q1, mean_q2 = None, None

        # We log means of Q to monitor training.
        #mean_q1 = curr_q1.detach().mean().item()
        #mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        # q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        curr_q1, curr_q2 = self.calc_current_q(*batch)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
        #print(curr_q1)
        #print(target_q)
        q1_loss = F.mse_loss(curr_q1, target_q)
        #print(q1_loss)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2,target_q

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # with torch.no_grad():
        # Q for every actions to calculate expectations of Q.
        # q1, q2 = self.critic(states)
        # q = torch.min(q1, q2)

        q1, q2 = self.critic(states)

        alpha = self.log_alpha.exp()
        # minq = torch.min(q1, q2)
        # inside_term = alpha * log_action_probs - minq
        # policy_loss = (action_probs * inside_term).mean()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies, action_probs

    def calc_entropy_loss2(self, pi_s, log_pi_s):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        alpha = self.log_alpha.exp()
        inside_term = - alpha * (log_pi_s + self.target_entropy).detach()
        entropy_loss = (pi_s * inside_term).mean()
        return entropy_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss

    def save_models(self, block):
        if self.chkpt_dir is not None:
            print('.... Saving models ....')
            self.actor.save_checkpoint(block)
            self.critic.save_checkpoint(block)
            self.target_critic.save_checkpoint(block)

    def load_models(self):
        print('.... Loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def load_baseline_models(self, baseline_dir):
        print('Loading baseline policy weights from: ', baseline_dir)
        self.actor.load_state_dict(torch.load(baseline_dir + "/baseline_actor"))
        self.critic.load_state_dict(torch.load(baseline_dir + "/baseline_critic"))
        self.target_critic.load_state_dict(torch.load(baseline_dir + "/baseline_critic"))

    def load_expert_policy(self, expert_number):
        print('.... Loading expert policy ....')
        self.actor.load_expert_checkpoint(expert_number)

