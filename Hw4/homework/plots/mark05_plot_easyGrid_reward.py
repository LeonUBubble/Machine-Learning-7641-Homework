import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

reward_file = '../easy_grid_reward.csv'

with open(reward_file,'r') as fr:
	content_tmp = fr.readline()
	
	content_tmp = fr.readline()
	iteration = content_tmp[11:-1].split(",")
	
	content_tmp = fr.readline()
	valueI_reward = content_tmp[24:-1].split(",")

	content_tmp = fr.readline()
	policyI_reward = content_tmp[25:-1].split(",")
	
	content_tmp = fr.readline()
	Q_reward = content_tmp[19:-1].split(",")
	
	print('iteration = ' + str(iteration))
	print('valueI_reward = ' + str(valueI_reward))
	print('policyI_reward = ' + str(policyI_reward))
	print('Q_reward = ' + str(Q_reward))
	
	
hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, valueI_reward, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Reward', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
plt.ylim(-100,100)
plt.xlim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_valueI_reward.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, policyI_reward, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Reward', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
plt.ylim(-100,100)
plt.xlim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_policyI_reward.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, Q_reward, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Reward', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_Q_reward.png')
plt.close(hf1)