# 不在使用LSTMCell来构建 LSTM网络,直接使用LSTM模型
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.optim as opt
from torch.utils.data import DataLoader, Dataset


# 1. load data

class GetSet(Dataset):
    def __init__(self, data_path, label_path):
        super().__init__()
        self.data = torch.load(data_path)
        self.label = torch.load(label_path)

    def __getitem__(self, item):
        _data = self.data[item]
        _label = self.label[item]
        return _data, _label

    def __len__(self):
        return len(self.data)


# train_data_set = GetSet("..\\data\\source_data.pt", "..\\data\\source_label.pt")
# test_data_set = GetSet("..\\data\\source_data_test.pt", "..\\data\\source_label_test.pt")
# noise data
train_data_set = GetSet("TheVariant\\source_data_noise.pt", "TheVariant\\source_label_noise.pt")
test_data_set = GetSet("TheVariant\\test_data_noise.pt", "TheVariant\\test_label_noise.pt")
train_data = DataLoader(train_data_set, batch_size=50, shuffle=True)
test_data = DataLoader(test_data_set, batch_size=10, shuffle=False)


# 2. create module
class LstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.LstmNet = nn.LSTM(input_size=1, hidden_size=51, num_layers=2, batch_first=True)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h0 = torch.zeros(2, input.shape[0], self.LstmNet.hidden_size, dtype=torch.float64)
        c0 = torch.zeros(2, input.shape[0], self.LstmNet.hidden_size, dtype=torch.float64)
        output, (ht, ct) = self.LstmNet(input,
                                        (h0, c0))  # 输入数据的格式为 Tensor([batch_num, sequence_len, input_features]),(h0,co)
        # output的输出格式为 Tensor([batch_num, sequence_len, hidden_size])
        for output_t in output.split(1, dim=1):
            outputs += [self.linear(torch.squeeze(output_t, dim=1))]
        prediction_source = torch.unsqueeze(outputs[-1], dim=1)   # 用上面的最后一个元素作为输入实现预测
        for _ in range(future): # 单步预测
            prediction_output, (ht, ct) = self.LstmNet(prediction_source, (ht, ct))
            output_pred = self.linear(torch.squeeze(prediction_output, dim=1))
            outputs += [output_pred]
            prediction_source = torch.unsqueeze(output_pred, dim=2)
        outputs = torch.cat(outputs, dim=1)
        return outputs


model = LstmModel()
model.double()
criterion = nn.MSELoss()
optimizer = opt.LBFGS(model.parameters(), lr=0.8)
model.zero_grad()  # zero model grad
# train step
MaxEpoch = 5
loss_record = []
for epoch in range(MaxEpoch):
    print(f'step:{epoch}')


    def closure():
        running_loss = 0
        for index, data in enumerate(train_data):
            optimizer.zero_grad()
            data_train, label_train = data
            data_train = torch.unsqueeze(data_train, dim=2)
            output = model(data_train)  # output的格式是一个List{999} 每一个包含20个数据
            loss = criterion(output, label_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            running_loss += loss.item()
        print(f'loss: {running_loss}')
        loss_record.append(running_loss)
        return running_loss


    optimizer.step(closure)
    with torch.no_grad():
        test_loss = 0
        future = 1000
        for index, data in enumerate(test_data):
            data_test, label_test = data
            data_test = torch.unsqueeze(data_test, dim=2)
            output_test = model(data_test, future=future)
            loss = criterion(output_test[:, :-future], label_test)
            y_pred = output_test[:, future-1:]
            test_loss += loss.item()

            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # save some fig
            plt.plot(np.arange(data_test.size(1)), torch.squeeze(data_test[0, :], dim=0), 'r', linewidth=2.0)
            plt.plot(np.arange(data_test.size(1), data_test.size(1) + future), np.squeeze(y_pred[0, :]),
                     'r:', linewidth=2.0)
            plt.plot(np.arange(data_test.size(1)), torch.squeeze(data_test[9, :], dim=0), 'g', linewidth=2.0)
            plt.plot(np.arange(data_test.size(1), data_test.size(1) + future), np.squeeze(y_pred[9, :]),
                     'g:', linewidth=2.0)
            plt.plot(np.arange(data_test.size(1)), torch.squeeze(data_test[5, :], dim=0), 'b', linewidth=2.0)
            plt.plot(np.arange(data_test.size(1), data_test.size(1) + future), np.squeeze(y_pred[5, :]),
                     'b:', linewidth=2.0)
            plt.savefig("RewritePredictLSTM%d.pdf" % epoch)

        print(f"test loss is : {test_loss}")

plt.figure(figsize=(30, 15))
plt.title('train process with loss', fontsize=30)
plt.xlabel('step', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(len(loss_record)), loss_record, 'r', linewidth=2)
plt.savefig('TrainLoss.pdf')
