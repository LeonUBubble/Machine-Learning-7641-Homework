import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

time_file = '../hard_grid_time_Boltzmann09.csv'

with open(time_file,'r') as fr:
	content_tmp = fr.readline()
	
	content_tmp = fr.readline()
	iteration = content_tmp[11:-1].split(",")
	
	content_tmp = fr.readline()
	valueI_time = content_tmp[16:-1].split(",")

	content_tmp = fr.readline()
	policyI_time = content_tmp[17:-1].split(",")
	
	content_tmp = fr.readline()
	Q_time = content_tmp[11:-1].split(",")
	
	print('iteration = ' + str(iteration))
	print('valueI_time = ' + str(valueI_time))
	print('policyI_time = ' + str(policyI_time))
	print('Q_time = ' + str(Q_time))
	
# Q_time = float(Q_time)
Q_time2 = list(map(float, Q_time))
# hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
# hp1 = plt.plot(iteration, valueI_time, c = 'b')
# plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
# plt.ylabel('times', fontname = 'times new roman', fontsize = 13)
# # plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
# plt.tight_layout()
# plt.grid()
# # plt.show(hf1)
# hf1.savefig('hard_grid_valueI_time.png')
# plt.close(hf1)

# hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
# hp1 = plt.plot(iteration, policyI_time, c = 'b')
# plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
# plt.ylabel('times', fontname = 'times new roman', fontsize = 13)
# # plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
# plt.tight_layout()
# plt.grid()
# # plt.show(hf1)
# hf1.savefig('hard_grid_policyI_time.png')
# plt.close(hf1)
print('Q_time = ' + str(max(Q_time2)))
print('Q_time = ' + str(min(Q_time2)))
print('Q_time = ' + str(max(Q_time2) - min(Q_time2)))
print(np.arange(0, 2000, 200))

hf1 = plt.figure(figsize = (6, 3.5), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, Q_time2, c = 'b', label = "Perfect")
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 12)
plt.ylabel('time', fontname = 'times new roman', fontsize = 12)
plt.xlim(0,1000)
plt.ylim(0,max(Q_time2))
plt.xticks(np.arange(0-1, 1000, 200))
plt.yticks(np.arange(min(Q_time2), max(Q_time2), 200))
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('hard_grid_Q_time_Boltzmann09.png')
plt.close(hf1)