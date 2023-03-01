
if __name__ == '__main__':
    import mne
    import numpy as np
    import os
    import matplotlib
    from FeatureParadigms import product_Paragram_CNN

    matplotlib.use('Qt5Agg')

    # 读入raw数据
    path = r'D:\WXMsWH\Warehouse\Datas\BCIC4\BCICIV_2a_gdf\train'
    file = os.listdir(path)[0]
    dir_file = os.path.join(path, file)
    raw = mne.io.read_raw(fname=dir_file)

    # 设置EOG通道
    eog_name = ['EOG-left', 'EOG-central', 'EOG-right']
    eog_type = ['eog', 'eog', 'eog']
    dict_eog = dict(zip(eog_name, eog_type))
    raw.set_channel_types(dict_eog)

    # 将EEG通道改成国际导联标准的命名
    ch_names = ['Fz',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2',
                'POz']
    ch_names_raw = ['EEG-Fz',
                    'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4',
                    'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8',
                    'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                    'EEG-14', 'EEG-Pz', 'EEG-15',
                    'EEG-16']
    # 修改通道名称需要传入一个字典，前面是通道的原名，后面是新名字
    dict_rename = dict(zip(ch_names_raw, ch_names))
    raw.rename_channels(dict_rename)
    raw.set_montage("standard_1020", on_missing='warn')

    # 设置重参考
    raw_ref = raw.copy()
    # 重参考方法要求先导入数据
    raw_ref.load_data()
    raw_ref.set_eeg_reference(ref_channels=['EOG-left', 'EOG-central', 'EOG-right'])

    # 带通滤波
    raw_ref_filter = raw_ref.copy()
    raw_ref_filter.filter(l_freq=0.5, h_freq=60)
    raw_ref_filter.plot_psd(fmax=80)

    # ICA部分,除杂
    ica = mne.preprocessing.ICA(n_components=22)
    ica.fit(raw_ref_filter)
    # 用脑地形图看ICA成分的
    ica.plot_components()

    # Extract and organize features into samples
    samples, labels = product_Paragram_CNN(raw=raw_ref_filter, labels=[7, 8, 9, 10])

    # Save Samples
    dir_dataset = r'Datas'
    for i in range(len(samples)):
        np.save(samples[i],
                os.path.join(dir_dataset, 'sample{}+label{}'.format(i, labels[i]))
                )
