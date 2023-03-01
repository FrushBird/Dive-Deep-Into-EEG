if __name__ == '__main__':

    import torch.optim
    from tqdm import tqdm
    import torch.nn as nn
    from Dataset.Examples import DatasetTemplate_train, DatasetTemplate_test
    from HookAndUtils import setup_seed, makeiter, evaluate_accuracy_gpu, evaluate_f1_score_gpu
    from Models.Example import ResNet18

    setup_seed(seed=1)
    net = ResNet18(4)
    num_epochs, batch_size, lr, device = 200, 256, 0.001, 'cuda'


    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    dataset_train = DatasetTemplate_train()
    dataset_test = DatasetTemplate_test()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # you can write down a summary here
    record_acc = []
    record_f1 = []

    for epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(dataset_train, batch_size)
        test_iter = makeiter(dataset_test, batch_size)
        net.train()
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x = x.float()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        test_f1_score = evaluate_f1_score_gpu(net, dataset_test)
        record_acc.append(test_acc)
        record_f1.append(test_f1_score)

    print("test_acc:", max(record_acc))
    print("f1_score:", max(record_f1))

