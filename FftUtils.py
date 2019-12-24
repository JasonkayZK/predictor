import matplotlib.pyplot as plt
from scipy import fftpack
from cmath import phase
import numpy as np
import random

class FftUtils(object):
    """
        一些和FFT相关的方法
    """
    #矩形窗
    @staticmethod
    def Wr(omega, N):
        return np.sin(N*omega/2) / np.sin(omega/2)

    #Blackman
    @staticmethod
    def Wb(omega, N):
        w = np.zeros(omega.size)
        b = [0.42,0.5,0.08]
        for i in np.arange(0, len(b)):
            w = w + b[i]/2*(FftUtils.Wr(omega-2*np.pi/N*i, N)+FftUtils.Wr(omega+2*np.pi/N*i, N))
        return w

    #Blackman-Harris
    @staticmethod
    def Wbh(omega, N):
        w = np.zeros(omega.size)
        b = [0.35875,0.48829,0.14128,0.01168]
        for i in np.arange(0, len(b)):
            w = w + b[i]/2*(FftUtils.Wr(omega-2*np.pi/N*i, N)+FftUtils.Wr(omega+2*np.pi/N*i, N))
        return w

    #Nuttall(3)
    @staticmethod
    def Wn3(omega, N):
        w = np.zeros(omega.size)
        b = [0.338946,0.481973,0.161054,0.018027]
        for i in np.arange(0, len(b)):
            w = w + b[i]/2*(FftUtils.Wr(omega-2*np.pi/N*i, N)+FftUtils.Wr(omega+2*np.pi/N*i, N))
        return w

    #矩形窗
    @staticmethod
    def wr(x):
        return x

    #Blackman
    @staticmethod
    def wb(x):
        n = np.arange(0, x.size)
        w = np.zeros(x.size)
        b = [0.42,-0.5,0.08]
        for i in np.arange(0, len(b)):
            w = w + b[i]*np.cos(2*np.pi*i*n/x.size)
        return w*x

    #Blackman-Harris
    @staticmethod
    def wbh(x):
        n = np.arange(0, x.size)
        w = np.zeros(x.size)
        b = [0.35875,-0.48829,0.14128,-0.01168]
        for i in np.arange(0, len(b)):
            w = w + b[i]*np.cos(2*np.pi*i*n/x.size)
        return w*x

    #Nuttall(3)
    @staticmethod
    def wn2(x):
        n = np.arange(0, x.size)
        w = np.zeros(x.size)
        b = [0.338946,-0.481973,0.161054,-0.018027]
        for i in np.arange(0, len(b)):
            w = w + b[i]*np.cos(2*np.pi*i*n/x.size)
        return w*x

    @staticmethod
    def plotFFT(y, sampling_rate = 1 / 8000, fft_size = 2 ** 16, labels='waveform', colors='g'):
        """
            给定一个t时长的波形， 给定采样频率， fft的精度， 计算fft并作图

            param:
                t: 采样时长
                sampling_rate: fft的范围
                fft_size: fft精度

            result: 
                作出频谱图
        """
        if fft_size >= 2 ** 22:
            return

        ys = y[:fft_size]
        yf = np.fft.rfft(ys)/fft_size
        freq = np.linspace(0, (1 / sampling_rate) / 2, len(yf))
        plt.plot(freq, abs(yf), colors, label=labels, )
        # freqs = np.array(map(lambda x : x/1e3, freq))
        # yfp = 20*np.log10(np.clip(np.abs(yf),1e-20,1e100))

    @staticmethod
    def findBasic(y, time = np.arange(0, 10, 1.0/8000), sampling_rate = 1.0/8000):
        N = 10000
        alpha=np.arange(-0.5,0.5,0.001)[1:]
        y1=FftUtils.Wb(2*np.pi*(-alpha-0.5)/N, N)
        y2=FftUtils.Wb(2*np.pi*(-alpha+0.5)/N, N)
        beta=(np.abs(y2)-np.abs(y1))/(np.abs(y2)+np.abs(y1))
        a = np.polyfit(beta, alpha, 7)
        # print(a)
        g = 2*N/(np.abs(y2)+np.abs(y1))
        b = np.polyfit(alpha, g, 7)
        # print(b)


        t = time
        x = FftUtils.wb(y)
        y = 2/t.size*fftpack.fft(x)
        k = 0
        max_y = 0
        for i in np.arange(0, t.size//2):
            if abs(y[i]) > max_y:
                max_y = abs(y[i])
                k = i
        if k > 0 and abs(y[k-1]) > abs(y[k+1]):
            k1 = k-1
            k2 = k
        else:
            k1 = k
            k2 = k+1

        beta = (abs(y[k2])-abs(y[k1]))/(abs(y[k2])+abs(y[k1]))
        alpha = a[0]*beta**7+a[2]*beta**5+a[4]*beta**3+a[6]*beta
        f = (alpha+k1+0.5)*1/ (len(t) * sampling_rate)
        # print(f)
        rms = (abs(y[k2])+abs(y[k1]))/2*(b[1]*alpha**6+b[3]*alpha**4+b[5]*alpha**2+b[7])
        # print(rms)
        if alpha <= 0:
            xita = phase(y[k1])-np.pi*(alpha+0.5)
        else:
            xita = phase(y[k2])-np.pi*(alpha-0.5)
        # print(xita)
        return f, rms, xita

    @staticmethod
    def mendBasicByFrame(y, time = np.arange(0, 10, 1.0/8000), windowWidth = 1, sampling_rate = 1.0/8000):
        """
            分窗口拆分得到不同窗口下的基波波形， 最后拼接后返回

            @param:
                time: 总时间
                windowWidth: 每个小窗口时间， 单位：秒
                sampling_rate: 采样率

            @return:
                源数据y的基波
        """
        # 0. 构建元数据
        windowTime = np.arange(0, windowWidth, sampling_rate) # 窗口时间
        # print(len(windowTime))
        result = [] # 数据结果容器


        # 将y 按照窗的大小(windowWidth)分为二维数组
        window_y = np.array(y).reshape((int(len(time) / (windowWidth / sampling_rate)), int(windowWidth / sampling_rate)))
        # print(np.shape(y))

        # 对于y的每一个窗, 求取一个基波
        # print(len(window_y))
        for i in range(len(window_y)):
            f, rms, xita = FftUtils.findBasic(np.array(window_y[i]), time = windowTime, sampling_rate = sampling_rate)
            normal = rms*np.cos(2*np.pi*f*windowTime+xita)
            result.append(normal)
        
        return np.array(result).reshape(len(time))


     

if __name__ == '__main__':
    from LstmUtils import LstmUtils
# windows
    # ua_normal, ub_normal, uc_normal, ia_harm_normal, ib_harm_normal, ic_harm_normal = LstmUtils.readRealData('E:\\Spyder_python\\paper_proj\\data\\realdata\\湘力\\电机正常1\\motor_normal_solved_14.txt')
    # ua_pianxin, ub_pianxin, uc_pianxin, ia_harm_pianxin, ib_harm_pianxin, ic_harm_pianxin = LstmUtils.readRealData('E:\\Spyder_python\\paper_proj\\data\\realdata\\湘力\\电机断条1\\motor_duantiao_solved_10.txt')

# Linux
    ua_normal, ub_normal, uc_normal, ia_harm_normal, ib_harm_normal, ic_harm_normal = LstmUtils.readRealData('/home/zk/workspace/NN_Python/paper_proj/data/realdata/motor_normal_solved_31.txt')
    ua_pianxin, ub_pianxin, uc_pianxin, ia_harm_pianxin, ib_harm_pianxin, ic_harm_pianxin = LstmUtils.readRealData('/home/zk/workspace/NN_Python/paper_proj/data/realdata/motor_pianxin2_5.txt')

    sampling_rate = 1 / 8000
    fft_size = 2 ** 16
    t = np.arange(0, 100, sampling_rate)

    # FFT 提取基波    
    # f, rms, xita = FftUtils.findBasic(np.array(ia_harm), time=t, sampling_rate=sampling_rate)
    # ia_normal = rms*np.cos(2*np.pi*f*t+xita)
    # plt.plot(ia_harm, label="Origin")
    # plt.plot(ia_normal, label="Non-basic")
    # plt.plot(ia_harm - ia_normal, 'y', label='Harm')    
    # plt.legend()
    # plt.show()
    
    # FFT 分析
    # FftUtils.plotFFT(ia_harm, labels='ia_harm', sampling_rate = 5000, fft_size = 2 ** 16)
    # plt.show()

    # 滤波分析
    # f, rms, xita = FftUtils.findBasic(np.array(ia_harm), time=t, sampling_rate=sampling_rate)
    # ia_normal = rms*np.cos(2*np.pi*f*t+xita)

    # FftUtils.plotFFT(ia_harm, labels='ia_full', sampling_rate = 5000, fft_size = 2 ** 16)
    # # FftUtils.plotFFT(ia_normal, labels='ia_basic', sampling_rate = 5000, fft_size = 2 ** 16)
    # FftUtils.plotFFT(ia_harm - ia_normal, labels='ia_harm', sampling_rate = 5000, fft_size = 2 ** 16)
    # plt.legend()
    # plt.show()


####################################  实际数据测试区域 ################################

  # 正常情况下:
    # ia_f, ia_rms, ia_xita = FftUtils.findBasic(np.array(ib_harm_pianxin), time=t, sampling_rate=sampling_rate)
    # fullBasic = ia_rms*np.cos(2*np.pi*ia_f*t+ia_xita)

    # windowBasic = FftUtils.mendBasicByFrame(ib_harm_pianxin, t, 1)
    # plt.plot(ib_harm_pianxin - fullBasic, 'g', label='full frame harm')
    # plt.plot(ib_harm_pianxin - windowBasic, 'r', label='window frame harm')
    # plt.legend()
    # plt.show()

  # 正常情况和其他情况对比:

    # plt.plot(ia_harm_normal, 'r', label='normal_ua')
    # plt.plot(ia_harm_pianxin, 'g', label='pianxin_ua')
    # plt.legend()
    # plt.show()

    windowNormalBasic = FftUtils.mendBasicByFrame(ia_harm_normal, t, 1)
    windowPianxinBasic = FftUtils.mendBasicByFrame(ia_harm_pianxin, t, 1)

    ia_harm_normal = ia_harm_normal - windowNormalBasic
    ia_harm_pianxin = ia_harm_pianxin - windowPianxinBasic

  # 正常测试
    FftUtils.plotFFT(ia_harm_normal, labels='ia_normal', sampling_rate=sampling_rate, fft_size=fft_size, colors='g')
    FftUtils.plotFFT(ia_harm_pianxin, labels='ia_pianxin', sampling_rate=sampling_rate, fft_size=fft_size, colors='r')
    # plt.title('normal')
    # plt.legend()
    # plt.show()

  # 归一化测试
    ia_harm_normal, ia_harm_normal_min, ia_harm_normal_max = LstmUtils.Normalize(ia_harm_normal)
    ia_harm_pianxin, ia_harm_pianxin_min, ia_harm_pianxin_max = LstmUtils.Normalize(ia_harm_pianxin)
   
    # FftUtils.plotFFT(ia_harm_normal, labels='ia_normal', sampling_rate=sampling_rate, fft_size=fft_size, colors='g')
    # FftUtils.plotFFT(ia_harm_pianxin, labels='ia_pianxin', sampling_rate=sampling_rate, fft_size=fft_size, colors='r')
    # plt.title('Normalized')
    # plt.legend()
    # plt.show()
  
  # 反归一化测试
    ia_harm_normal = LstmUtils.FNoramlize(ia_harm_normal, high=ia_harm_normal_max, low=ia_harm_normal_min)
    ia_harm_pianxin = LstmUtils.FNoramlize(ia_harm_pianxin, high=ia_harm_pianxin_max, low=ia_harm_pianxin_min)
    
    FftUtils.plotFFT(ia_harm_normal, labels='ia_normal', sampling_rate=sampling_rate, fft_size=fft_size, colors='y')
    FftUtils.plotFFT(ia_harm_pianxin, labels='ia_pianxin', sampling_rate=sampling_rate, fft_size=fft_size, colors='blue')
    plt.title('FNormalized')
    plt.legend()
    plt.show()






