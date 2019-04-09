import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
#
# action_file = '../hard_grid_number_of_action.csv'

action_file = '../hard_grid_action_EpsilonG01.csv'
with open(action_file,'r') as fr:
	content_tmp = fr.readline()
	
	content_tmp = fr.readline()
	iteration = content_tmp[11:-1].split(",")
	iteration = list(map(eval, iteration))

	
	content_tmp = fr.readline()
	valueI_action = content_tmp[16:-1].split(",")
	valueI_action = list(map(eval, valueI_action))

	content_tmp = fr.readline()
	policyI_action = content_tmp[17:-1].split(",")
	policyI_action = list(map(eval, policyI_action))

	content_tmp = fr.readline()
	Q_action = content_tmp[11:-1].split(",")
	Q_action = list(map(eval, Q_action))
	
	print('iteration = ' + str(iteration))
	print('valueI_action = ' + str(valueI_action))
	print('policyI_action = ' + str(policyI_action))
	print('Q_action = ' + str(Q_action))
	

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, valueI_action, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Actions', fontname = 'times new roman', fontsize = 13)
#plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
plt.ylim(0,100)
#plt.tight_layout()
plt.grid()
plt.show(hf1)
hf1.savefig('hard_grid_valueI_action.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, policyI_action, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Actions', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
plt.ylim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('hard_grid_policyI_action.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, Q_action, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('Actions', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
plt.ylim(0,2000)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('hard_grid_Q_action.png')
plt.close(hf1)
