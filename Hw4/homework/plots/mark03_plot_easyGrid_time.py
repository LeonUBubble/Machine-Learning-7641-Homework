import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

time_file = '../easy_grid_time.csv'

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
	
	
hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, valueI_time, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('times', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_valueI_time.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, policyI_time, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('times', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_policyI_time.png')
plt.close(hf1)

hf1 = plt.figure(figsize = (5, 4), facecolor = 'w', dpi = 100)
hp1 = plt.plot(iteration, Q_time, c = 'b')
plt.xlabel('iterations', fontname = 'times new roman', fontsize = 13)
plt.ylabel('times', fontname = 'times new roman', fontsize = 13)
# plt.legend(('EM', 'Kmeans'), loc = 4, fontsize = 13)
# plt.ylim(0,100)
plt.tight_layout()
plt.grid()
# plt.show(hf1)
hf1.savefig('easy_grid_Q_time.png')
plt.close(hf1)