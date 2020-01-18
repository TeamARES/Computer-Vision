import numpy as np
from datetime import datetime
import time
'''
def funct (big_ara, threshold):
	res_list=[]
	for point in big_ara :
		if point[2]>=threshold :
			res_list.append(point)

	return res_list

#a = np.asarray([[1,2,3],[11,22,33],[2,3,4],[22,33,44]])
'''

while True:
	np.random.seed(0)
	
	a = np.random.rand(300000,3)*10402
	g = a[np.where(a[:,2]>6)]
	
	print(g)
	

