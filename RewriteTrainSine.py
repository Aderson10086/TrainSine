import torch
import matplotlib
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
matplotlib.use('TkAgg')

# create data
cuda = torch.device('cuda')


def generate_data(time, length, number):
    x = np.empty((number, length), 'float64')
    x[:] = np.array(range(length)) + np.random.randint(-4 * time, 4 * time, number).reshape(number, 1)
    data = np.sin(x / 1.0 / time).astype('float64')
    data = torch.tensor(data)
    # use the current data to predict the next
    source_data = data[:, :-1]
    source_label = data[:, 1:]
    return source_data, source_label


# source_data_, source_label_ = generate_data(20, 1000, 50)
# torch.save(source_data_, open('..\\data\\source_data_test.pt', 'wb'))
# torch.save(source_label_, open('..\\data\\source_label_test.pt', 'wb'))


# define dataset
class GetSet(Dataset):
    def __init__(self, source_data, source_label, from_where='workspace'):
        super(GetSet, self).__init__()
        if from_where == 'workspace':
            self.source_data = source_data
            self.source_label = source_label
        elif from_where == 'file':
            self.source_data = torch.load(source_data)
            self.source_label = torch.load(source_label)
        else:
            raise ValueError('the value of from_where is wrong, only workspace or file')

    def __getitem__(self, item):
        __data = self.source_data[item]
        __label = self.source_label[item]
        return __data, __label

    def __len__(self):
        return len(self.source_data)


# for index, data in enumerate(train_data, start=0):
#     print(f'index: {index}   data:{data}')
#     plt.figure(index)
#     plt.plot(np.arange(0, data[0].size(1)), torch.squeeze(data[0],dim=0))
#     plt.show()
#     print("h")

# create model
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        if torch.cuda.is_available():
            self.lstmcell1 = nn.LSTMCell(1, 51, device=cuda)
            self.lstmcell2 = nn.LSTMCell(51, 51, device=cuda)
            self.linear = nn.Linear(51, 1, device=cuda)
        else:
            self.lstmcell1 = nn.LSTMCell(1, 51)
            self.lstmcell2 = nn.LSTMCell(51, 51)
            self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        output = 0.0
        if torch.cuda.is_available():
            h_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=cuda)
            c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=cuda)
            h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=cuda)
            c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=cuda)
        else:
            h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
            c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
            h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
            c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstmcell1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for _ in range(future):
            h_t, c_t = self.lstmcell1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


# model
seq = model()
seq.double()
seq.zero_grad()  # zero grad
epochMax = 14
criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(seq.parameters(), lr=0.8)
# optimizer = torch.optim.SGD(seq.parameters(), lr=0.01,momentum=0.9)
# get the data
train_data = DataLoader(GetSet("..\\data\\source_data.pt", "..\\data\\source_label.pt", from_where='file'),
                        batch_size=20, shuffle=True)
test_data = DataLoader(GetSet("..\\data\\source_data_test.pt", "..\\data\\source_label_test.pt", from_where='file'),
                       batch_size=50, shuffle=True)
loss_record = []
for epoch in tqdm(range(epochMax)):
    print(f'step:{epoch}')


    def closure():
        running_loss = 0.0
        for _, data in enumerate(train_data):
            train, train_label = data
            optimizer.zero_grad()
            outputs = seq(train.cuda(cuda))
            loss = criterion(outputs.cuda(cuda), train_label.cuda(cuda))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq.parameters(), max_norm=10, norm_type=2) # 梯度裁剪，防止梯度爆炸

            running_loss += loss.cpu().item()
        print(f'loss: {running_loss}')
        loss_record.append(running_loss)
        return running_loss


    optimizer.step(closure)
    # test
    test_running_loss = 0.0
    with torch.no_grad():
        future = 1000
        for _, data_test in enumerate(test_data):
            test, test_label = data_test
            outputs_test = seq(test.cuda(cuda), future)
            loss = criterion(outputs_test.cuda(cuda)[:, :-future], test_label.cuda(cuda))
            test_running_loss += loss.cpu().item()
            y_pred = outputs_test.cpu().detach().numpy()
            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # save some fig
            plt.plot(np.arange(test.size(1)), torch.squeeze(test[0, :], dim=0), 'r', linewidth=2.0)
            plt.plot(np.arange(test.size(1), test.size(1) + future), np.squeeze(y_pred[0, future-1:]),
                     'r:', linewidth=2.0)
            plt.plot(np.arange(test.size(1)), torch.squeeze(test[10, :], dim=0), 'g', linewidth=2.0)
            plt.plot(np.arange(test.size(1), test.size(1) + future), np.squeeze(y_pred[10, future-1:]),
                     'g:', linewidth=2.0)
            plt.plot(np.arange(test.size(1)), torch.squeeze(test[30, :], dim=0), 'b', linewidth=2.0)
            plt.plot(np.arange(test.size(1), test.size(1) + future), np.squeeze(y_pred[30, future-1:]),
                     'b:', linewidth=2.0)
            plt.savefig("RewritePredict%d.pdf" % epoch)
    print(f'test loss:{test_running_loss}')

plt.figure(figsize=(20, 20))
plt.title('The train step with loss', fontsize=20)
plt.xlabel('step', fontsize=10)
plt.ylabel('loss', fontsize=10)
plt.plot(np.arange(len(loss_record)), loss_record, 'k-', linewidth=2.0)
plt.savefig("Loss.pdf")