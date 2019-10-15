import os
import librosa
import librosa.display
from scipy import signal
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import fcm

def bandpass(y,fs1,fs2,sr):
    """
	带通滤波器。

	Parameters
    ----------
	y：输入信号。
    fs1：带通滤波器的下限截止频率。
    fs2：带通滤波器的上限截止频率。
    sr：样本的采样频率。

	Returns
    -------
	filtedData：输出信号。
	"""
    b, a = signal.butter(10, [2.0*fs1/sr,2.0*fs2/sr],'bandpass')
    filtedData = signal.filtfilt(b,a,y)

    return filtedData


def lowpass(y,fs,sr):
    """
	低通滤波器。

	Parameters
    ----------
	y：输入信号。
    fs：低通滤波器的截止频率。
    sr：样本的采样频率。

	Returns
    -------
    filtedData：输出信号。
    """
    b, a = signal.butter(10,2.0*fs/sr, 'lowpass')  
    filtedData = signal.filtfilt(b, a,y) 

    return filtedData


def PSD_calculate_write(y,f,classtype):
    """
	计算信号的PSD，并记录在文件中。

	Parameters
    ----------
	y：输入信号。
    f：写入的文件。
    classtype：信号的类别。如，'cough'或'other'。
    """
    # 计算信号y的PSD
    PSD, _ = plt.mlab.psd(y,128,16000,scale_by_freq=False)
    for i in range(0,len(PSD)):
        f.write(str(PSD[i])+',\t')
    f.write(classtype+'\n')


def calculate_e(data,C):
    """
    计算阈值e。

	Parameters
    ----------
	data：所有样本。
    C：一个聚类中心。

	Returns
    -------
    e：阈值e。
    """
    e = 0
    for i in range(len(data)):
        e += fcm.distance(data[i],C)
    e = e / len(data)

    return e


def calculate_D(data,data_cough):
    """
    计算样本与全部咳嗽样本的距离均值。

	Parameters
    ----------
	data：所有样本。
    data_cough：咳嗽样本。

	Returns
    -------
    D：每个样本与咳嗽样本的距离均值。
    """
    D = []
    for i in range(len(data)):
        d = 0
        for j in range(len(data_cough)):
            d += fcm.distance(data[i],data_cough[j])
        #D.append(d/(len(data_cough)-1))
        D.append(d/len(data_cough))
    
    return D


def compare_d_e(e,D):
    """
    比较d与e的大小。
    'd<=e'判断为'cough'类，'d>e'判断为'other'类。

	Parameters
    ----------
	e：阈值e。
    D：包含所有样本d的矩阵。

	Returns
    -------
    result：d与e比较后的结果。
    """
    result = []
    # d与e比较，统计result
    for i in range(len(D)):
        if D[i] <= e :
            result.append('cough')
        elif D[i] > e:
            result.append('other')
    
    return result


def checker(result,label):
    """
    对比结果与标签并统计准确率。

	Parameters
    ----------
	result：d与e比较后的结果。
    label：样本标签。
    """  
    right = 0
    # result与label比较，得到正确的结果
    for i in range(len(result)):
        if result[i] == label[i]:
            right += 1
    print('正确的个数是：',right)
    print('准确率：{}%' .format(right/len(label) * 100))


if __name__ == "__main__":
    audio_path = './data/'
    classtype = [os.path.basename(i) for i in glob(audio_path + '*')]

    path_cough = os.path.join(audio_path,classtype[0])
    files_cough = os.listdir(path_cough)
    path_other = os.path.join(audio_path,classtype[1])
    files_other = os.listdir(path_other)

    # 带通滤波器截止频率
    fre_band = [[6000,7000],[2000,5000],[2000,4000],
                [4000,6000],[2300,3200],[5000,7000]]
    # 低通滤波器截止频率
    fre_low = [7000,6000,2000]
    
    plt.figure()
    # 样本经过带通滤波器
    for i in range(len(fre_band)):
        print(fre_band[i][0],fre_band[i][1])
        for file in files_cough:
            y, sr = librosa.load(os.path.join(path_cough,file),sr=16000)
            y = bandpass(y,fre_band[i][0],fre_band[i][1],sr)
            plt.psd(y,128,16000,scale_by_freq=False,lw=0.8,color='red')
        for file in files_other:
            y, sr = librosa.load(os.path.join(path_other,file),sr=16000)
            y = bandpass(y,fre_band[i][0],fre_band[i][1],sr)
            plt.psd(y,128,16000,scale_by_freq=False,lw=0.8,color='blue') 
        plt.xlabel('Frequency(Hz)') 
        plt.show()      
    # 样本经过低通滤波器
    for i in range(0,len(fre_low)):
        print(fre_low[i])
        for file in files_cough:
            y, sr = librosa.load(os.path.join(path_cough,file),sr=16000)
            y = lowpass(y,fre_low[i],sr)
            plt.psd(y,128,16000,scale_by_freq=False,lw=0.8,color='red')
        for file in files_other:
            y, sr = librosa.load(os.path.join(path_other,file),sr=16000)
            y = lowpass(y,fre_low[i],sr)
            plt.psd(y,128,16000,scale_by_freq=False,lw=0.8,color='blue')            
        plt.xlabel('Frequency(Hz)')
        plt.show()
