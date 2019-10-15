import copy
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
 
global MAX # 用于初始化隶属度矩阵U
MAX = 10000.0

global Epsilon  # 结束条件
Epsilon = 0.0000001

def import_data_format_iris(file):
    """
	导入数据（函数名未更改）。

	Parameters
    ----------
	file：'./psd/'文件夹下的文件。如，'./psd/pig.txt'。

	Returns
    -------
	data：样本。
	cluster：标签。
    """
    data = []
    cluster = []
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")  # 对每一行以逗号为分割，返回一个list
            current_dummy = []
            for j in range(0, len(current)-1):
                current_dummy.append(float(current[j]))  # current_dummy存放data
            data.append(current_dummy)
            cluster.append(current[-1].replace('\t',''))
	#cluster = [cluster[i].replace('\t','') for i in range(len(cluster))]
    print("加载数据完毕")

    return data ,cluster
 

def print_matrix(list):
	""" 
	显示多维列表。
	
	Parameters
    ----------
	list：要显示的列表。
	"""
	for i in range(0, len(list)):
		print (list[i])
 

def initialize_U(data, cluster_number):
	"""
	随机初始化隶属度矩阵U。
	
	Parameters
    ----------
	data：样本。
	cluster_number：聚类类别数。

	Returns
    -------
	U：随机初始化后的隶属度矩阵U。
	"""
	global MAX
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)

	return U
 

def distance(point, center):
	"""
	计算样本与聚类中心之间的欧几里得距离。
	
	Parameters
    ----------
	point：一个样本。
	center：一个聚类中心。

	Returns
    -------
	dummy：二者的距离。
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	dummy = math.sqrt(dummy)

	return dummy


def end_conditon(U, U_old):
    """
	迭代结束条件。
	当隶属度矩阵U停止变化时，结束迭代。
	
	Parameters
    ----------
	U：更新后的隶属度矩阵。
	U_old：更新前的隶属度矩阵。
	"""
    global Epsilon
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):
		    if abs(U[i][j] - U_old[i][j]) > Epsilon :
			    return False
    return True
 

def normalise_U(U):
	"""
	在聚类结束时对隶属度矩阵U模糊化。
	每个样本的隶属度最大值变为1，其余的变为0。

	Parameters
    ----------
	U：隶属度矩阵。

	Returns
    -------
	U：归一化后的隶属度矩阵。
	"""
	for i in range(0, len(U)):
		maximum = max(U[i])
		for j in range(0, len(U[0])):
			if U[i][j] != maximum:
				U[i][j] = 0
			else:
				U[i][j] = 1
	return U


def fuzzy(data, cluster_number, m, init_method='random'):
	"""
	模糊C均值算法。计算聚类中心和归一化隶属矩阵。
	m的最佳取值范围为[1.5，2.5]。
	
	Parameters
    ----------
	data：样本。
	cluster_number：聚类类别数。
	m：2。
	init_method：初始化方法。

	Returns
    -------
	U：隶属度矩阵。
	C：聚类中心。
	"""
	# 初始化隶属度矩阵U
	# 1. random：对U随机初始化。
	# 2. half：论文中的方法，U=0.5并得到两个相等的聚类中心C。
	if init_method == 'random':
		U = initialize_U(data, cluster_number)
	elif init_method == 'half':
		U = (np.zeros((len(data),cluster_number)) + 0.5).tolist()
	# print_matrix(U)
	# 循环更新U
	while (True):
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
		# 计算聚类中心
		C = []
		# 第j类
		for j in range(0, cluster_number):
			current_cluster_center = []
			# 第i个样本特征点
			for i in range(0, len(data[0])):
				dummy_sum_num = 0.0
				dummy_sum_dum = 0.0
				# 第k个样本
				for k in range(0, len(data)):
    				# 分子
					dummy_sum_num += (U[k][j] ** m) * data[k][i]
					# 分母
					dummy_sum_dum += (U[k][j] ** m)
				# 第i列的聚类中心
				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j类的所有聚类中心
			C.append(current_cluster_center)
 
		# 创建一个距离向量, 用于计算U矩阵。
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)
 
		# 更新U
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0, cluster_number):
    				# 分母
					dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / dummy
 
		if end_conditon(U, U_old):
			print ("结束聚类")
			break
	print ("标准化 U")
	U = normalise_U(U)
	return U, C

if __name__ == '__main__':
	start = time.time()
	data, label = import_data_format_iris("./psd/pig.txt")		# 加载数据
	"""
	这行代码应该本在记录原始音频PSD的位置，但由于缺少这一步，聚类一直是错误的。
	耗费了一周的时间，找到了原因。为了警示犯过的错误，放在了这里。
	"""
	data = 10 * np.log10(np.array(data))		# 将计算的PSD值转换为dB单位
	# 数据归一化，没用上
	#data_normed = data_array / data_array.max(axis=0)

	# 调用模糊C均值函数，随机初始化隶属度矩阵U，得到两个不同的聚类中心
	U, C = fuzzy(data, 2, 2, init_method='random')		
	# 调用模糊C均值函数，令隶属度矩阵U=0.5，得到两个相同的聚类中心
	#U, C = fuzzy(data, 2, 2, init_method='half')		
	print ("用时：{0}".format(time.time() - start))

	# 绘制样本
	fs = np.linspace(0,8000,65)
	plt.figure()
	# 绘制cough类PSD曲线
	# TODO:不同类样本数变化后，需要修改相关参数
	for i in range(len(data[0:36])):
		plt.plot(fs,data[i],lw=0.5,color='red')
	# 绘制other类PSD曲线
	for i in range(len(data[36:72])):
		plt.plot(fs,data[36+i],lw=0.5,color='blue')
	
	# 绘制聚类中心
	plt.plot(fs,C[0],lw=4,color='black')
	plt.plot(fs,C[1],lw=4,color='yellow')
	plt.xlabel('Frequency(Hz)')
	plt.ylabel('Power Spectral Density(dB)')
	plt.grid()
	plt.show()
	