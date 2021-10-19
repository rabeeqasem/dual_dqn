import numpy as np
import random
from torch._C import device
from tqdm.autonotebook import tqdm
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import random
from create_graph import create_graph
from environment import environment
from Qnetwork import QNetwork
import matplotlib.pyplot as plt
import seaborn as sns




class agent:
	def __init__(self,batch_size=32,buffer_size=500000,target_update_freq=1000,episodes=2000):
		super(agent, self).__init__()
		cp=create_graph()
		self.data=cp.data
		self.g=cp.get_g()

		self.batch_size=batch_size
		self.buffer_size=buffer_size
		self.min_replay_size=int(self.buffer_size*0.25)
		self.target_update_freq=target_update_freq
		self.episodes=episodes
		flag=0
		self.gamma=0.99
		self.epsilon=0.5
		self.start=1
		self.end=0.1
		self.rew_buffer=[0]
		penalties=[]

		self.gamma_list=[]
		self.mean_reward=[]
		self.done_location=[]
		self.loss_list=[]
		self.number_of_episodes=[]

		self.env=environment()
		#device_class=get_device()
		#self.device=device_class.device
		self.device=T.device('cuda' if T.cuda.is_available() else 'cpu')
		self.num_actions=self.env.num_actions
		self.online=QNetwork(self.num_actions*2,self.num_actions)
		self.target=QNetwork(self.num_actions*2,self.num_actions)
		self.target.load_state_dict(self.online.state_dict())
		self.optimizer=T.optim.Adam(self.online.parameters(),lr=1e-4)


		
		action_list=np.arange(0,len(self.g.nodes)).tolist()
		self.replay_buffer=deque(maxlen=self.min_replay_size)
		self.env=environment()
		self.episode_reward=0

	def get_neighbors(self,obs):
	  current_node,end=self.env.state_dec(obs)
	  neighbors=[self.env.enc_node[i] for i in self.g.neighbors(self.env.dec_node[current_node])]
	  return neighbors

	def train(self):
		obs=self.env.reset()	
		for _ in tqdm(range(self.min_replay_size)):
		  #action=np.random.choice(action_list)
		  #
		  neighbors=self.get_neighbors(obs)
		  action=np.random.choice(neighbors)
		  
		  new_obs,rew,done=self.env.step(obs,action)
		  transition=(obs,action,rew,done,new_obs)
		  self.replay_buffer.append(transition)
		  obs=new_obs
		  if done:
		    obs=self.env.reset()

		#main training loop
		obs=self.env.reset()
		decay=self.episodes


		self.stat_dict={'episodes':[],'epsilon':[],'explore_exploit':[],'time':[]}


		#for i in tqdm(range(episodes)):
		for i in tqdm(range(self.episodes)):

		  itr=0
		  #epsilon=np.interp(i,[0,decay],[start,end])
		  #gamma=np.interp(i,[0,decay],[start,end])
		  epsilon=np.exp(-i/(self.episodes/2))
		  rnd_sample=random.random()

		  self.stat_dict['episodes'].append(i)
		  self.stat_dict['epsilon'].append(self.epsilon)

		  #choose an action
		  if rnd_sample <=epsilon:
		    #action=np.random.choice(action_list)
		    neighbors=self.get_neighbors(obs)
		    action=np.random.choice(neighbors)
		    self.stat_dict['explore_exploit'].append('explore')

		  else:
		    source,end=self.env.state_dec(obs)
		    v_obs=self.env.state_to_vector(source,end)
		    t_obs=T.tensor([v_obs]).to(self.device)
		    action=self.online.select_action(t_obs)
		    self.stat_dict['explore_exploit'].append('exploit')

		  #fill transition and append to replay buffer

		  
		  new_obs,rew,done=self.env.step(obs,action)

		  transition=(obs,action,rew,done,new_obs)
		  self.replay_buffer.append(transition)
		  obs=new_obs
		  self.episode_reward+=rew


		  if done:
		    obs=self.env.reset()
		    self.rew_buffer.append(self.episode_reward)
		    self.episode_reward=0.0
		    self.done_location.append(i)


		  #start gradient step
		  transitions=random.sample(self.replay_buffer,self.batch_size)
		  obses=np.asarray([t[0] for t in transitions])
		  actions=np.asarray([t[1] for t in transitions])
		  rews=np.asarray([t[2] for t in transitions])
		  dones=np.asarray([t[3] for t in transitions])
		  new_obses=np.asarray([t[4] for t in transitions])


		  obses_t=T.as_tensor(obses,dtype=T.float32).to(self.device)
		  actions_t=T.as_tensor(actions,dtype=T.int64).to(self.device).unsqueeze(-1)
		  rews_t=T.as_tensor(rews,dtype=T.float32).to(self.device)
		  dones_t=T.as_tensor(dones,dtype=T.float32).to(self.device)
		  new_obses_t=T.as_tensor(new_obses,dtype=T.float32).to(self.device)
		  list_new_obses_t=T.tensor(self.env.list_of_vecotrs(new_obses_t)).to(self.device)
		  target_q_values=self.target(list_new_obses_t)##


		  max_target_q_values=target_q_values.max(dim=1,keepdim=False)[0]
		  targets=rews_t+self.gamma*(1-dones_t)*max_target_q_values
		  targets=targets.unsqueeze(-1)
		  
		  list_obses_t=T.tensor(self.env.list_of_vecotrs(obses_t)).to(self.device)
		  q_values=self.online(list_obses_t)
		  action_q_values=T.gather(input=q_values,dim=1,index=actions_t)

		  
		  #warning UserWarning: Using a target size (torch.Size([24, 24])) that is different to the input size (torch.Size([24, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
		  
		  loss=nn.functional.mse_loss(action_q_values,targets)
		  #loss=dqn_clipped_loss(action_q_values,targets,max_target_q_values,gamma)
		  self.loss_list.append(loss.item())

		  self.optimizer.zero_grad()
		  loss.backward()
		  self.optimizer.step()

		  #plot
		  self.mean_reward.append(np.mean(self.rew_buffer))
		  self.number_of_episodes.append(i)
		  self.gamma_list.append(self.gamma)
		  #dec = {'number_of_episodes':number_of_episodes,'mean_reward':mean_reward,'gamma':gamma_list}

		  

		  if i % self.target_update_freq==0:
		    self.target.load_state_dict(self.online.state_dict())
		  if i % 1000 ==0:
		    print('step',i,'avg rew',round(np.mean(self.rew_buffer),2))

	def plot_result(self):
		dec = {'number_of_episodes':self.number_of_episodes,'mean_reward':self.mean_reward,'gamma':self.gamma_list,'loss':self.loss_list,'explore_exploit':self.stat_dict['explore_exploit']}
		fig, ax =plt.subplots(1,3,figsize=(15,5))
		sns.lineplot(data=dec, x="number_of_episodes", y="mean_reward",ax=ax[0])
		sns.lineplot(data=dec, x="number_of_episodes", y="loss",ax=ax[1])
		sns.countplot(data=dec,x='explore_exploit', ax=ax[2])

		plt.show()
	def test(self):
		obs=self.env.reset()
		done=False
		sp=[obs]
		while not done:
		  source,end=self.env.state_dec(obs)
		  v_obs=self.env.state_to_vector(source,end)
		  t_obs=T.tensor([v_obs]).to(self.device)
		  action=self.target.select_action(t_obs)
		  new_obs,rw,done=self.env.step(obs,action)
		  sp.append(new_obs)
		  obs=new_obs
		prnit(sp)
