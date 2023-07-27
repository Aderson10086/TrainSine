# modeling 2d data
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


# 1. create data

def create_data(num_points=1000):
    t = torch.linspace(start=0, end=5 / 3 * np.pi, steps=num_points, dtype=torch.float64)
    x = np.cos(t)
    y = np.sin(t)
    # plt.plot(x, y)
    # plt.show()
    _data = torch.cat([torch.unsqueeze(x, dim=1), torch.unsqueeze(y, dim=1)], dim=1)
    _source_data = _data[:-1, :]
    _source_label = _data[1:, :]
    return _source_data, _source_label


source_data, source_label = create_data()


class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=31, num_layers=2, batch_first=False)
        self.linear = nn.Linear(self.lstm.hidden_size, self.lstm.input_size)

    def forward(self, input, future=0):
        outputs = []
        h0 = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size, dtype=torch.float64)
        c0 = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size, dtype=torch.float64)
        output, (ht, ct) = self.lstm(input, (h0, c0))
        for output_t in output.split(1, dim=0):
            outputs += [self.linear(output_t)]
        # prediction
        prediction_source = torch.unsqueeze(input[-1, :], dim=0)

        for _ in range(future):
            output, (ht, ct) = self.lstm(prediction_source, (ht, ct))
            prediction_output = self.linear(output)
            prediction_source = prediction_output
            outputs += [prediction_output]
        outputs = torch.cat(outputs, dim=0)
        return outputs


model = lstm_model()
model.double()
model.zero_grad()

train_data = source_data
train_label = source_label

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)

maxEpoch = 10
loss_record = []
future = 300
for step in range(maxEpoch):
    print(f'step:{step}')


    def closure():
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_label)
        loss_record.append(loss.detach().numpy())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        loss.backward()
        print(f'loss:{loss}')
        return loss


    optimizer.step(closure)
    with torch.no_grad():
        output_prediction = model(train_data, future=future)
        plt.figure(figsize=(20, 20))
        plt.plot(train_data[:, 0], train_data[:, 1], 'k', linewidth=2)
        plt.plot(output_prediction[-future:, 0], output_prediction[-future:, 1], 'r:', linewidth=2)
        plt.savefig("Circle_Predict%d.pdf" % step)

plt.figure(figsize=(30, 20))
plt.title('The train step with logarithmic loss', fontsize=20)
plt.xlabel('step', fontsize=10)
plt.ylabel('log(loss)', fontsize=10)
plt.plot(np.arange(len(loss_record)), np.log(loss_record), 'k-', linewidth=2.0)
plt.savefig("Loss.pdf")