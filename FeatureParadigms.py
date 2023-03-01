import mne
import numpy as np

from HookAndUtils import SlidBin, compute_difE, ReorganizeByMontage


def product_Paragram_CNN(raw, labels, num_chan=64):
    # raw: raw class in mne, labels: a list contains event marks
    (events_value, events_anno) = mne.events_from_annotations(raw)
    events_relative = np.array([i for i in events_value if i[2] in labels])
    labels_list = events_relative[:, 2]
    samples_list = []
    for event in events_relative:
        sample = []
        for i in range(num_chan):
            # ERD:Event-Relevant-Data
            erd_chan = raw[i][0][event[0]:(event[0] + 4.8 * 250)]
            erd_chan_cuts = SlidBin(erd_chan)
            features_chan = np.array([compute_difE(cuts, simple=False) for cuts in erd_chan_cuts]).reshape(-1)
            sample.append(features_chan)
        samples_list.append(np.array(sample))

    return samples_list, labels_list


def product_Paragram_RNN(raw, labels, num_chan=64):
    # raw: raw class in mne, labels: a list contains event marks
    (events_value, events_anno) = mne.events_from_annotations(raw)
    events_relative = np.array([i for i in events_value if i[2] in labels])
    labels_list = events_relative[:, 2]
    samples_list = []
    for event in events_relative:
        sample = []
        # 'i' is a coefficient to adjust pieces of the temporal sample
        for i in range(4):
            sample_piece_chans = []
            for j in range(num_chan):
                # ERD:Event-Relative-Data
                erd_chan = raw[j][0][(event[0] + i * 250):(event[0] + (i + 1) * 250)]
                erd_chan_cuts = SlidBin(erd_chan)
                features_chan = np.array([compute_difE(cuts, simple=False) for cuts in erd_chan_cuts]).reshape(-1)
                sample_piece_chans.append(features_chan)
            sample_piece = ReorganizeByMontage(sample_piece_chans)

            sample.append(sample_piece)

        samples_list.append(np.array(sample))

    return samples_list, labels_list


def product_Paragram_MultiModality(raw, labels, num_chan=22):
    """
    Product multiModalities samples for RNN training, Return samples with size N*H*(montage's shape).
    N is the number of modalities, H is the number of hidden layers, montage size is decided by your electrodes map
    raw: raw class in mne, labels: a list contains event marks
    """

    (events_value, events_anno) = mne.events_from_annotations(raw)
    events_relative = np.array([i for i in events_value if i[2] in labels])
    labels_list = events_relative[:, 2]

    # 脑电信号的多模态样本有多种实现，这里选择最简单的一种：基于频率
    num_modality = 5
    lfs = [1, 4, 8, 13, 30]
    hfs = [4, 8, 13, 30, 50]

    samples_list = []
    for event in events_relative:
        sample = []
        # 'i' is a coefficient to adjust pieces of the temporal sample
        for modality in range(num_modality):
            raw_c = raw.copy()
            raw_c.filter(l_freq=lfs[modality], h_freq=hfs[modality])
            sample_modality = []
            for i in range(4):
                sample_piece_chans = []
                for j in range(num_chan):
                    # ERD:Event-Relative-Data
                    erd_chan = raw[j][0][(event[0] + i * 250):(event[0] + (i + 1) * 250)]
                    erd_chan_cuts = SlidBin(erd_chan)
                    features_chan = np.array([compute_difE(cuts, simple=False) for cuts in erd_chan_cuts]).reshape(-1)
                    sample_piece_chans.append(features_chan)
                sample_piece = ReorganizeByMontage(sample_piece_chans)
                sample_modality.append(sample_piece)
            sample.append(np.array(sample_modality))
        samples_list.append(np.array(sample))

    return samples_list, labels_list
