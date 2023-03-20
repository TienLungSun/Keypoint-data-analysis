import numpy as np
from scipy.signal import convolve

def MorletWavelet(fc):
    F_RATIO = 7
    Zalpha2 = 3.3

    sigma_f = fc / F_RATIO
    sigma_t = 1 / (2 * np.pi * sigma_f)
    A = 1 / np.sqrt(sigma_t * np.sqrt(np.pi))
    max_t = np.ceil(Zalpha2 * sigma_t)

    t = np.arange(-max_t, max_t + 1)

    v1 = 1 / (-2 * sigma_t ** 2)
    v2 = 2j * np.pi * fc
    MW = A * np.exp(t * (t * v1 + v2))

    return MW


def tfa_morlet(td, fs, fmin, fmax, fstep):
    TFmap = np.array([])
    for fc in np.arange(fmin, fmax + fstep, fstep):
        MW = MorletWavelet(fc / fs)
        cr = convolve(td, MW, mode='same')

        TFmap = np.vstack([TFmap, abs(cr)]) if TFmap.size else abs(cr)

    return TFmap


def coarse_grain(ts, scale):  #比較容易了解版本
    seg = int(np.floor(len(ts)/scale))
    ts1 = np.zeros(seg) #new time series
    for i in range(seg):
        head = i*scale 
        tail = head+scale-1
        seg = ts[head:tail+1]
        ts1[i] = np.mean(seg)
    return ts1

def sample_entropy(ts, Mdim, r_ratio):
    n = len(ts)
    r = r_ratio*np.std(ts)
    SE=np.zeros(Mdim) # Mdim 個 SE
    count_m = np.zeros(Mdim+1) # 計算 Midm SE 時需要用到 Midm+1, 因此要多一個
    
    for i in range(n-Mdim):  # index (min,max)=(0,n-1), i+Mdim=n-1 => i=n-Mdim-1
        for j in range(i+1, n-Mdim): # j+Mdim 要 <= n-1, 所以 range到 n-Mdim
            m=0  # 0~Mdim 
            while(m<=Mdim and abs(ts[i+m]-ts[j+m]) <= r):
                count_m[m] += 1
                m = m+1
                
    for m in range(Mdim):
        if(count_m[m] ==0 or count_m[m+1] ==0):
            #SE[m] = -np.log(1/((n-m)*(n-m-1))) # a large number
            SE[m] = 0
        else:
            SE[m] = -np.log(count_m[m+1]/count_m[m]) 
    return SE