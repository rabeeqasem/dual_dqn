import numpy as np
import torch as T
import numpy as np
from create_graph import create_graph




class environment:
  def __init__(self):
      super(environment, self).__init__()
      cp=create_graph()
      self.data=cp.data
      self.g=cp.get_g()
      self.enc_node={}
      self.dec_node={}
      self.n=len(self.g.nodes)
      self.num_actions = len(self.g.nodes)
      self.action_list=np.arange(0,len(self.g.nodes)).tolist()

      #you can define your wights here
      self.wights=np.array([[0.07051304, 0.2594585, 0.03459922, 0.06227937,0.15332614,0.14832941,0.18294566,0.08854866]])

      for index,nd in enumerate(self.g.nodes):
        self.enc_node[nd]=index
        self.dec_node[index]=nd
            
  def state_enc(self,dst, end):
    return dst+self.n*end

  def state_dec(self,state):
    dst = state%self.n
    end = (state-dst)/self.n
    return dst, int(end)

  def reset(self):
    self.state=self.state_enc(self.enc_node[153531392],self.enc_node[5239133571])
    return self.state
  

  def wayenc(self,node,action):
    #encoded
    if action in self.g[node]:
      rw=self.g[node][action]['weight']*-1
      return rw,True
    rw=-1000
    return rw,False

  def slop_reward(self,current_node,new_node):
    slop=-self.data[self.g[current_node][new_node]['parent']]['slop']
    result=np.power(slop,3)
    return result


  def road_type_reward(self,current_node,new_node,pref_road_type:list):
    road_type_between_two_nods=self.data[self.g[current_node][new_node]['parent']]['tags']['highway']
    if road_type_between_two_nods in pref_road_type:
      if "cycleway" in pref_road_type:
        return 100
      else:
        return 50
    else:
      return -50


  def lane_reward(self,current_node,new_node):
    parent_id=self.g[current_node][new_node]['parent']
    if self.data[parent_id].__contains__('tags'):
      if self.data[parent_id]['tags'].__contains__('lanes'):
        num_lanes=int(self.data[parent_id]['tags']['lanes'])
        reward=num_lanes*-10
        return reward
      else: return 0
    else:return 0

  def light_reward(self,current_node,new_node):
    parent_id=self.g[current_node][new_node]['parent']
    if self.data[parent_id].__contains__('tags'):
      if self.data[parent_id]['tags'].__contains__('lit'):
        light_type=self.data[parent_id]['tags']['lit']
        if light_type=="yes":
          reward=50
        elif light_type in ['24/7','sunset-sunrise','automatic','operating times']:
          reward=10
        elif light_type in ['no','disused']:
          reward=-20
        return reward
      else: return 0
    else:return 0
    
  def surface_type_reward(self,current_node,new_node,pref_surface_type:list):
    parent_id=self.g[current_node][new_node]['parent']
    if self.data[parent_id].__contains__('tags'):
      if self.data[parent_id]['tags'].__contains__('surface'):
        surface_type_between_two_nodes=self.data[self.g[current_node][new_node]['parent']]['tags']['surface']
        if surface_type_between_two_nodes in pref_surface_type:
          return 50
        else:
          return -50
      else:return 0 
    else:return 0
  
  def tracer_number_reward(self,current_node,new_node):
    parent_id=self.g[current_node][new_node]['parent']
    trace_number_edge =self.data[parent_id]['trace_number']
    reward=-10*trace_number_edge
    return reward

  def max_speed_reward(self,current_node,new_node):
    parent_id=self.g[current_node][new_node]['parent']
    if self.data[parent_id].__contains__('tags'):
      if self.data[parent_id]['tags'].__contains__('maxspeed'):
        max_speed_value=int(self.data[parent_id]['tags']['maxspeed'])
        reward=max_speed_value*-1
        return reward
      else:return 0
    else: return 0

  def rw_function(self,node,action):
    

    node=self.dec_node[node]
    action=self.dec_node[action]
    beta0=self.wights[0][0] #between 1 and 0
    beta1=self.wights[0][1]
    beta2=self.wights[0][2]
    beta3=self.wights[0][3]
    beta4=self.wights[0][4]
    beta5=self.wights[0][5]
    beta6=self.wights[0][6]
    beta7=self.wights[0][7]

    rw0,link=self.wayenc(node,action)

    if link:
      rw1=self.slop_reward(node,action)
      rw2=self.road_type_reward(node,action,['primary','residential'])
      rw3=self.lane_reward(node,action)
      rw4=self.light_reward(node,action)
      rw5=self.max_speed_reward(node,action)
      rw6=self.surface_type_reward(node,action,['asphalt'])
      rw7=self.tracer_number_reward(node,action)
      frw=beta0*rw0 + beta1*rw1+ beta2*rw2  + beta3*rw3+ beta4*rw4 +rw5*beta5 + rw6*beta6+rw7*beta7
    else:
      frw=rw0


    return frw,link



  def step(self,state,action):
    done=False
    #end=1731824802

    current_node , end = self.state_dec(state)

    new_state = self.state_enc(action,end)
    rw,link=self.rw_function(current_node,action)

    if not link:
        new_state = state
        return new_state,rw,False  

    elif action == end:
        rw = 10000 #500*12
        done=True
      
    return new_state,rw,done


  def state_to_vector(self,current_node,end_node):
    n=len(self.g.nodes)
    source_state_zeros=[0.]*n
    source_state_zeros[current_node]=1

    end_state_zeros=[0.]*n
    end_state_zeros[end_node]=1.
    vector=source_state_zeros+end_state_zeros
    return vector


      

  #return a list of list converted from state to vectors
  def list_of_vecotrs(self,new_obses_t):
    list_new_obss_t=new_obses_t.tolist()
    #convert to integer
    list_new_obss_t=[int(v) for v in list_new_obss_t]
    vector_list=[]
    for state in list_new_obss_t:
      s,f=self.state_dec(state)
      vector=self.state_to_vector(s,f)
      vector_list.append(vector)
    return vector_list

  def dqn_clipped_loss(self,current_q, target_q, target_q_a_max, gamma):
      # max[current_q, delta^2 + target_q] + max[delta,gamma*(target_q_a_max)^2]
      delta = current_q - target_q
      left_max = T.max(current_q, T.pow(delta, 2) + target_q)
      right_max = T.max(delta, gamma * T.pow(target_q_a_max, 2))
      loss = T.mean(left_max + right_max)

      return loss



