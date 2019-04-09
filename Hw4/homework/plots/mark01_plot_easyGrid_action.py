import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

action_file = '../easy_grid_number_of_action.csv'

with open(action_file,'r') as fr:
	content_tmp = fr.readline()
	
	content_tmp = fr.readline()
	iteration = content_tmp[11:-1].split(",")
	
	content_tmp = fr.readline()
	valueI_action = content_tmp[16:-1].split(",")

	content_tmp = fr.readline()
	policyI_action = content_tmp[17:-1].split(",")
	
	content_tmp = fr.readline()
	Q_action = content_tmp[11:-1].split(",")
	
	print('iteration = ' + str(iteration))
	print('valueI_action = ' + str(valueI_action))
	print('policyI_action = ' + str(policyI_action))
	print('Q_action = ' + str(Q_action))
	