from torch.nn import functional as F
from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn



# nn中大写的是层，如Conv2d，nn.function中小写的是函数


class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        # if i == 0 and not first_block:
        # 不是很懂为什么要对第一个块特别处理
        if i == 0:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def make_ResNet18(input_channal, output_size=24):
    b1 = nn.Sequential(nn.Conv2d(input_channal, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 256), nn.Dropout(0.5), nn.Linear(256, 64),
                        nn.Dropout(0.2), nn.Linear(64, 32), nn.Dropout(0.2), nn.Linear(32, 16), nn.Dropout(0.2),
                        nn.Linear(16, output_size))
    return net


class ResNet18(nn.Module):
    # 简洁版的resnet18，使用pytorch的官方实现
    def __init__(self, size_output, pretrain=False,Cl = False ,input_ch = 1):
        super(ResNet18, self).__init__()
        # 还有一种方案，是在最前面加入卷积层，将潜变量化为三通道
        # 取掉model的后1层
        model = resnet18(pretrained=pretrain)
        if input_ch == 1:
            if Cl:
                self.layers = nn.Sequential(nn.Conv2d(1, 3, 1, 1), *list(model.children())[:-1], nn.Flatten(),
                                            nn.Linear(512, size_output),
                                            nn.Softmax(),
                                            # 用于CL时,启用softmax；用于AE训练时，关闭softmax
                                            )
            else:
                self.layers = nn.Sequential(nn.Conv2d(1, 3, 1, 1), *list(model.children())[:-1], nn.Flatten(),
                                            nn.Linear(512, size_output),
                                            # nn.Softmax(),
                                            # 用于CL时,启用softmax；用于AE训练时，关闭softmax
                                            )

        else:

            if Cl:
                self.layers = nn.Sequential(*list(model.children())[:-1], nn.Flatten(),
                                            nn.Linear(512, size_output),
                                            nn.Softmax(),
                                            # 用于CL时,启用softmax；用于AE训练时，关闭softmax
                                            )
            else:
                self.layers = nn.Sequential(*list(model.children())[:-1], nn.Flatten(),
                                            nn.Linear(512, size_output),
                                            # nn.Softmax(),
                                            # 用于CL时,启用softmax；用于AE训练时，关闭softmax
                                            )

    def forward(self, inputs):
        if len(inputs.shape) != 4:
            inputs = inputs.reshape(len(inputs), -1, 64, 96)
            # 输入至少三维，batch_size，H，W

        # 20221221更新，在初始化的时候就设置好是不是单通道的
        return self.layers(inputs)


class RNNModel(nn.Module):
    '''
    Input rnn layers and it would return an RNN model.
    '''

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


if __name__ == '__main__':
    pass
