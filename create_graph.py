
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pyproj
from torch.distributions import Categorical,Normal,Bernoulli
from IPython.display import clear_output



class create_graph:
  def __init__(self):
    super(create_graph,self).__init__()
    self.data={}
  
  def get_neighbors(self,node):
    neighbors=[i for i in self.h.neighbors(node)]
    return len(neighbors)

  def get_g(self):

    # Opening JSON file 
    f = open('final.json') 
      
    # returns JSON object as  
    # a dictionary 
    data_str_key = json.load(f)
    #solve the issue of string keys and convert it to numbers
    
    for k,v in data_str_key.items():
      self.data[int(k)]=v
    self.h = nx.Graph()
    for key in self.data.keys():
      if self.data[key]['type']=='way':
        for i in range(len(self.data[key]['nodes'])-1):
          if 'tags' in self.data[key] and 'name' in self.data[key]['tags']:
              self.h.add_edge(self.data[key]['nodes'][i],self.data[key]['nodes'][i+1],parent=self.data[key]['id'],label=self.data[key]['tags']['name'])
          else:
            self.h.add_edge(self.data[key]['nodes'][i],self.data[key]['nodes'][i+1],parent=self.data[key]['id'])
    
    geod = pyproj.Geod(ellps='WGS84')
    # Compute distance among the two nodes indexed[s] indexed[d] using LON and LAT
    for s,d in self.h.edges():
      azimuth1, azimuth2, distance = geod.inv(self.data[s]['lon'],self.data[s]['lat'],self.data[d]['lon'],self.data[d]['lat'])
      self.h.edges[s,d]['weight'] = distance
    
    nodex={}
    for node in self.h.nodes:
      nodex[node]=self.get_neighbors(node)

    mx = max(nodex.values())
    [k for k, v in nodex.items() if v == mx]


    starter_node=[2206595456,
     2206595457,
     2871518853,
     2871518854,
     2871518855,
     2556217356,
     8286989452,
     2556217358,
     6267765394,
     588544148,
     1251974676,
     1251974678,
     1799221657,
     299832604,
     1799221660,
     299832606,
     1799200156,
     3700188574,
     1799221665,
     1731824802,
     153531430,
     8309646505,
     8309646506,
     754682796,
     1129756716,
     687588783,
     687588785,
     208456626,
     1799221682,
     1011880503,
     5239132089,
     637882815,
     153343425,
     8311054278,
     2206595455,
     4963500103,
     1429379270,
     1011881546,
     3107640523,
     3107640522,
     2709866189,
     2364408910,
     2364408911,
     8309625418,
     8309625419,
     8309625420,
     1250468692,
     1131363286,
     1250468696,
     1250468701,
     2556169573,
     5124157158,
     5124157159,
     418502504,
     5124157157,
     5124157160,
     2003461227,
     5124157161,
     1250409837,
     1130166767,
     1250409840,
     2115095921,
     2115095922,
     2003461235,
     299831408,
     2003461237,
     588148339,
     1130166770,
     2206595452,
     2003461246,
     2003461247]
    node_list=[]
    for sn in starter_node:
      node_list.append(sn)
      for node in self.h.neighbors(sn):
        node_list.append(node)
        for snode in self.h.neighbors(node):
          node_list.append(snode)
    dictt={}
    for node in node_list:
      c=0
      for subnode in node_list:
        if subnode in self.h[node]:
          c+=1
      dictt[node]=c

    #########################

    self.g = self.h.subgraph(dictt.keys())
    return self.g
    
  def plot_g(self):
    tt=nx.get_edge_attributes(self.g, 'weight')
    graph_labels=nx.get_edge_attributes(self.g, 'weight')
    # Plotting the Graph
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(self.g)
    edge_labels = nx.get_edge_attributes(self.g, 'weight')
    for label in edge_labels:
      edge_labels[label]=round(edge_labels[label],2)
    nx.draw(self.g, pos, node_size=100)
    nx.draw_networkx_edge_labels(self.g, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(self.g, pos, font_size=10)
    plt.show()




cp=create_graph()
g=cp.get_g()
cp.plot_g()