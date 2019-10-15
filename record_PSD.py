import os
import librosa
import data_process as dp
from glob import glob

def record_raw_PSD():
    """
    记录原始音频的PSD。
    保存为'./psd/pig.txt'。
    """
    f = open('./psd/pig.txt','w')
    for file in files_cough:
        y, _ = librosa.load(os.path.join(path_cough,file),sr=16000)
        dp.PSD_calculate_write(y,f,classtype[0])

    for file in files_other:
        y, _ = librosa.load(os.path.join(path_other,file),sr=16000)
        dp.PSD_calculate_write(y,f,classtype[1])
    f.close()
    print('over')


def record_filtrate_PSD():
    """
    记录样本通过滤波器后的功率谱密度。
    保存为'./psd/pig_{}_{}.txt'。
    """
    # 记录样本通过带通滤波器后的功率谱密度
    for i in range(len(fre_band)):
        print(fre_band[i][0],fre_band[i][1])
        f = open('./psd/pig_{}_{}.txt' .format(fre_band[i][0],fre_band[i][1]),'w')
        for file in files_cough:
            y, sr = librosa.load(os.path.join(path_cough,file),sr=16000)
            y = dp.bandpass(y,fre_band[i][0],fre_band[i][1],sr)
            dp.PSD_calculate_write(y,f,classtype[0])
        
        for file in files_other: 
            y, sr = librosa.load(os.path.join(path_other,file),sr=16000)
            y = dp.bandpass(y,fre_band[i][0],fre_band[i][1],sr)
            dp.PSD_calculate_write(y,f,classtype[1])
        f.close()    

    # 记录样本通过低通滤波器后的功率谱密度
    for i in range(0,len(fre_low)):
        print(fre_low[i])
        f = open('./psd/pig_0_{}.txt' .format(fre_low[i]),'w')
        for file in files_cough:
            y, sr = librosa.load(os.path.join(path_cough,file),sr=16000)
            y = dp.lowpass(y,fre_low[i],sr)
            dp.PSD_calculate_write(y,f,classtype[0])
        
        for file in files_other:
            y, sr = librosa.load(os.path.join(path_other,file),sr=16000)
            y = dp.lowpass(y,fre_low[i],sr)
            dp.PSD_calculate_write(y,f,classtype[1])
        f.close()
    print('over')  


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

    record_raw_PSD()
    #record_filtrate_PSD()