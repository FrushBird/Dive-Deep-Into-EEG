from mne.time_frequency import psd_array_multitaper
import math
import numpy as np
import torch
import random
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import torch.nn as nn
import d2l.torch as d2l
import sklearn


def compute_psd(data_chan, sample_rate=250, cut_down=80):
    # 计算功率谱密度
    psd, freqs = psd_array_multitaper(data_chan, sample_rate, adaptive=True, normalization='full', verbose=0)
    psd = np.reshape(psd, (-1))
    freqs = freqs[freqs < cut_down]
    psd = psd[0:len(freqs)]
    return psd, freqs


def compute_difE(data_chan, simple=True):
    if simple:
        # 微分熵可以作为特定频段上能量强度的表征（对数)
        # 此处使用简化公式，见‘Differential Entropy Feature for EEG-based Vigilance Estimation’，采用简化公式计算微分熵
        # 先求方差
        variance = np.var(data_chan, ddof=1)
        difE = math.log(2 * math.pi * math.e * variance) / 2
    else:
        # 此处不使用简化公式，而采用平均功率谱密度计算difE
        psd, freqs = compute_psd(data_chan)
        psd_avg = np.average(psd)
        difE = np.log2(psd_avg)

    return difE


def compute_power(data_chan, lower_freq=0.5, upper_freq=4):
    # 积分功率谱密度获取特定频带上的功率
    psd, freqs = compute_psd(data_chan)
    df = upper_freq - lower_freq
    idx_band = np.logical_and(freqs >= lower_freq, freqs <= upper_freq)
    power_band = simps(psd[idx_band], dx=df)
    return power_band


def SlidBin(data_chan, step=20, bin=15):
    # Cut integral data into pieces.Set up your own window&step size.
    data_chan = np.reshape(data_chan, (-1))
    data_cuts = []
    for i in range(0, len(data_chan) - bin, step):
        data_cuts.append(data_chan[i:i + bin])

    return np.array(data_cuts)


def ReorganizeByMontage(data_chans):
    sparse = np.zeros_like(data_chans)
    # Here is a reorganized example from BCIC 2008 dataset4A
    data_from_montage = np.array([
        sparse, sparse, sparse, data_chans[0], sparse, sparse, sparse,
        sparse, data_chans[1], data_chans[2], data_chans[3], data_chans[4], data_chans[5], sparse,
        data_chans[6], data_chans[7], data_chans[8], data_chans[9], data_chans[10], data_chans[11], data_chans[12],
        sparse, data_chans[13], data_chans[14], data_chans[15], data_chans[16], data_chans[17], sparse,
        sparse, sparse, data_chans[19], data_chans[20], data_chans[21], sparse, sparse,
        sparse, sparse, sparse, data_chans[22], sparse, sparse, sparse,
    ])
    return data_from_montage


def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

def makeiter(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             num_workers=0)  # 这里可调成4
    dataiter = iter(dataloader)
    return dataiter


def evaluate_f1_score_gpu(net, dataset_test, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    samples = dataset_test.samples
    labels = dataset_test.labels
    y_hats = []
    for x in samples:
        x = x.float().reshape(1, x.shape[0], x.shape[1])
        x = x.to(device)
        y_hat = net(x)
        y_hat = d2l.argmax(y_hat)
        y_hat = y_hat.cpu().numpy().tolist()
        y_hats.append(y_hat)
    return sklearn.metrics.f1_score(labels, y_hats, average='micro')


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(x, list):
                # Required for BERT Fine-tuning (to be covered later)
                x = x.float()
                x = [x.to(device) for x in x]

            else:
                x = x.float()
                x = x.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(x), y), d2l.size(y))

    return metric[0] / metric[1]
