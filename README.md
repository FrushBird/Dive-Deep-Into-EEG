# Incomplete Guide for Dive-Into-EEG
If you are interested in neural decoding, or you are a crazy fans of neuroprosthetics and brain-computer interfaces, you should dive into EEG!
Dive Into EEG is a litter program for decoding EEG signal, which is super ideal for fresh birds who wants to try neural decoding. Here let me introduce you a basic skeleton:

1. You should clean your EEG raw signal first. It is recommended to clean EEG signals with EEGlab in MATLAB. But, since we are using Python, the best language in this wolrd, we do it in Python xD. Preprocessing.py contains an EEG preprocess script realized by mne( https://mne.tools/stable/index.html ), and it would create samples for training. You can set up your own preprocessing scheme following example or instruction of mne( https://mne.tools/stable/auto_tutorials/index.html ).

2. Seconed you should extract features from EEG signals, then arrange them in meaningful sequences. Many hooks and useful tools, like computing PSD, Differential entropy and ohters, are included in the HookAndUtils.py. Please make good use of them to extract features or you can set up your own methods. Several organization paradigms, including schemes for CNN,RNN and mulit-modalities analyse, are provided in FeatureParadigms.py. You can modify it as you wish, if you have some new idea to dig out EEG informations.

3. Now you have transformed raw data into samples available for training. But before training, establish your deep learning models in '/Models'. If you dont know how to start a dl model, see tutorials of PyTorch. Also I provided a little example :).

4. Ultimately, we can start trining! A simple but typical train process of deep learning is provided in Main.py. Also you can modify it as you need.

5. I ll give out example dataset as soon as I found a really suitble dataset for training (Most public datasets are so suck now :/). Since it is training, just use your own dataset.
