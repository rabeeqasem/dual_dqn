

from ddqn import agent

if __name__ == '__main__':
    agt=agent()
    agt.train()
    agt.plot_result()
    agt.test()

