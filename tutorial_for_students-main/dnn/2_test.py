import numpy as np
import torch
import torch.nn as nn
import gen_data

n = 16
mu_background = 0
mu_object = 0

X_obs = gen_data.generate_test(n, mu_background, mu_object)
X_obs = torch.tensor(X_obs, dtype=torch.float32)


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
model = torch.load('./model/model.pth')

prediction = model(X_obs)
prediction = prediction.detach().numpy() # convert to numpy array

# print(prediction)
classification_result = (prediction > 0.5).astype(int)

print(classification_result)

# NOTE: By running this experiment few times, tou will see that in some cases, even we set
# mu_object = mu_background = 0 at the begining (i.e., there is NO truly object region),
# the DNN still classify some data points as 1.0. This indicates the results of DNN is UNRELIABLE


