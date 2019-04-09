import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

reward_file = '../hard_grid_reward_Boltzmann01.csv'

with open(reward_file,'r') as fr:
	content_tmp = fr.readline()
	
	content_tmp = fr.readline()
	iteration = content_tmp[11:-1].split(",")
	
	content_tmp = fr.readline()
	valueI_reward = content_tmp[16:-1].split(",")

	content_tmp = fr.readline()
	policyI_reward = content_tmp[17:-1].split(",")
	
	content_tmp = fr.readline()
	Q_reward = content_tmp[19:-1].split(",")
	
	print('iteration = ' + str(iteration))
	print('valueI_reward = ' + str(valueI_reward))
	print('policyI_reward = ' + str(policyI_reward))
	print('Q_reward = ' + str(Q_reward))
	
# Q_reward = float(Q_reward)
Q_reward2 = list(map(float, Q_reward))
# hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
# hp1 = plt.plot(iteration, valueI_reward, c = 'b')
# plt.xlabel('iterations', fontname = 'rewards new roman', fontsize = 13)
# plt.ylabel('rewards', fontname = 'rewards new roman', fontsize = 13)
# # plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
# plt.tight_layout()
# plt.grid()
# # plt.show(hf1)
# hf1.savefig('hard_grid_valueI_reward.png')
# plt.close(hf1)

# hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
# hp1 = plt.plot(iteration, policyI_reward, c = 'b')
# plt.xlabel('iterations', fontname = 'rewards new roman', fontsize = 13)
# plt.ylabel('rewards', fontname = 'rewards new roman', fontsize = 13)
# # plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
# plt.tight_layout()
# plt.grid()
# # plt.show(hf1)
# hf1.savefig('hard_grid_policyI_reward.png')
# plt.close(hf1)
print('Q_reward = ' + str(max(Q_reward2)))
print('Q_reward = ' + str(min(Q_reward2)))
print('Q_reward = ' + str(max(Q_reward2) - min(Q_reward2)))
print(np.arange(0, 2000, 200))

hf1 = plt.figure(figsize = (6, 3.5), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, Q_reward2, c = 'b', label = "Perfect")
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 12)
plt.ylabel('reward', fontname = 'times new roman', fontsize = 12)
plt.xlim(0,1000)
plt.ylim(0,max(Q_reward2))
plt.xticks(np.arange(0-1, 1000, 200))
plt.yticks(np.arange(min(Q_reward2), max(Q_reward2), 200))
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('hard_grid_Q_reward_Boltzmann01.png')
plt.close(hf1)