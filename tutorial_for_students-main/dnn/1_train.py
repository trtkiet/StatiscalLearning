import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import gen_data


sample_size = 100
n = 16
mu_background = 0
mu_object = 0


train_data_1, train_label_1 = gen_data.generate_train(sample_size, n, 0.0, 0.5, flag=1)
train_data_2, train_label_2 = gen_data.generate_train(sample_size, n, 0.0, 0.0, flag=0)

train_data = np.vstack((train_data_1, train_data_2))
train_label = np.vstack((train_label_1, train_label_2))

train_data = torch.tensor(train_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.float32)

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(n, 128)
        self.relu = nn.ReLU()
        self.hidden_2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.output = nn.Linear(64, n)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.relu_2(self.hidden_2(x))
        x = self.sigmoid(self.output(x))
        return x

model = DNN()

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100  # number of epochs to run
batch_size = 50  # size of each batch
batch_start = torch.arange(0, len(train_data), batch_size)


for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            train_data_batch = train_data[start:start + batch_size]
            train_label_batch = train_label[start:start + batch_size]
            # forward pass
            y_pred = model(train_data_batch)
            loss = loss_fn(y_pred, train_label_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            acc = (y_pred.round() == train_label_batch).float().mean()
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )

torch.save(model, './model/model.pth')

