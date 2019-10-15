import librosa
import os
import fcm
import numpy as np
import matplotlib.pylab as plt
import random
import data_process as dp

def trans_R2res(R):
    """
	统计9次比较的结果。
    5次以上'd<=e'，判断为'cough'类；5次一下'd>e'，判断为'other'类。

	Parameters
    ----------
	R：9次比较后的['cough','other']矩阵。

	Returns
    -------
	result：最终的['cough','other']矩阵。
	"""
    # 将['cough','other']转换为['1','0']
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] == 'cough':
                R[i][j] = 1
            elif R[i][j] == 'other':
                R[i][j] = 0
    R_array = np.array(R).T
    # 9个结果中，若5个及以上判为cough（即5个以上1），则为cough；反之为other
    result = []
    for i in range(len(R_array)):
        right = 0
        for j in range(len(R_array[0])):
            if R_array[i][j] == 1:
                right += 1
        if right >= 5:
            result.append(1)
        else:
            result.append(0)
    #print(result)
    # 将['1','0']转换为['cough','other']
    for i in range(len(result)):
        if result[i] == 1:
            result[i] = 'cough'
        elif result[i] == 0:
            result[i] = 'other'

    return result


def plot_C(data,C):
    """
    绘制样本与聚类中心C。

	Parameters
    ----------
	data：样本。
    C：一个聚类中心C。
    """    
    fs = np.linspace(0,8000,65)
    plt.figure()
    # 绘制cough类PSD曲线
    for i in range(len(data[0:36])):
        plt.plot(fs,data[i],lw=0.5,color='red')
    # 绘制other类PSD曲线
    for i in range(len(data[36:72])):
        plt.plot(fs,data[36+i],lw=0.5,color='blue')

    plt.plot(fs,C[0],lw=4,color='yellow')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Power Spectral Density(dB)')
    plt.grid()
    plt.show()


def main_paper():
    """
    文献中的方法
    """
    data, label = fcm.import_data_format_iris("./psd/pig.txt")
    data = 10 * np.log10(np.array(data))
    data_cough = data[0:36]  # 用于计算D，
    
    # TODO:可以随机选取20个样本
    #random.seed(1)
    #index = random.sample(range(len(data_cough)),20)
    #print(index)

    _, C = fcm.fuzzy(data, 2, 2, init_method='half')
    e = dp.calculate_e(data,C[0])       # 计算阈值e
    D = dp.calculate_D(data,data_cough)
    res = dp.compare_d_e(e,D)
    dp.checker(res,label)       # 输出结果
    plot_C(data,C)      # 绘制样本与聚类中心


def main_paper_improved():
    """
    文献中对main_paper()方法的改进
    """
    psd_path = './psd/'
    files = os.listdir(psd_path)
    files.remove('pig.txt')     # 剔除pig.txt文件
    R = []      # 9次d与e的比较结果

    for file in files:
        data, label = fcm.import_data_format_iris(os.path.join(psd_path,file))
        print(file.replace('.txt',''))
        data = 10 * np.log10(np.array(data))
        data_cough = data[0:36]
        _, C = fcm.fuzzy(data, 2, 2, init_method='half')
        e = dp.calculate_e(data,C[0])
        D = dp.calculate_D(data,data_cough)
        res = dp.compare_d_e(e,D)
        print('------------------------------')
        #plot_C(data,C)      # 绘制样本与聚类中心
        R.append(res)
    
    result = trans_R2res(R)
    dp.checker(result,label)


if __name__ == "__main__":
    # 文献方法
    main_paper()
    # 文献改进方法
    #main_paper_improved()
    