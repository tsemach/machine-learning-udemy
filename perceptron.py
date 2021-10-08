#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import print_assert_equal
import seaborn as sns
import torch
import torch.nn as nn
from sklearn import datasets
import pandas as pd

#%%
# create two cluster where the centers points of the clusters are define by centers variable

n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

x_data = torch.tensor(X).float()
y_data = torch.tensor(y.reshape(n_pts, 1)).float()

print('x_data.shape = ', x_data.shape, 'x-min:', x_data.min(), 'x-max:', x_data.max())
print('y_data.shape = ', y_data.shape, 'y-min:', y_data.min(), 'y-max:', y_data.max())

#%%
def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1])
  plt.scatter(X[y==1, 0], X[y==1, 1])

#%%
scatter_plot()

#%%
def plot_yesno(data: torch.Tensor, header: str, xlabel = '', ylabel = ''):
  sns.set_style('darkgrid')    
  plt.xlabel('total =' + xlabel)
  plt.ylabel('total =' + ylabel)
  sns.histplot(data.numpy(), bins=2, kde=False)  

#%%
x1 = x_data[:, 0]
x2 = x_data[:, 1]
plot_yesno(x_data, 'x_data 2D bin', str(len(x1[x1 > 0])), str(len(x1[x1 < 0])))

#%%
plot_yesno(x_data, 'x_data 2D bin', str(len(y_data[y_data > 0])), str(len(y_data[y_data < 0])))

#%%
plot_yesno(y_data, 'y_data 2D bin')
# plot_yesno(y_data, 'y_data 2D bin', str(len(y_data[y_data == 0])), str(len(y_data[y_data == 1])))

#%%
class Model(nn.Module):

  def __init__(self, input_size, output_size):
    super().__init__() 
    self.linear = nn.Linear(input_size, output_size)


  def forward(self, x):
    x = self.linear(x)
    pred = torch.sigmoid(x)

    return pred


  def predict(self, x):
    pred = self.forward(x)
    if pred >= 0.5:
      return 1
    else:
      return 0

#%%
torch.manual_seed(2)
model = Model(2, 1)
print('model paramters:', list(model.parameters()))

[w, b] = model.parameters()

# view w as a tuple of size 2
w1, w2 = w.view(2)

#%%
def get_params(model: Model):
  [w, b] = model.parameters()

  # view w as a tuple of size 2
  w1, w2 = w.view(2)

  return (w1.item(), w2.item(), b[0].item())

#%%
w1, w2, b1 = get_params(model)
print(w1, w2, b1)

#%%
def plot_fit(title):
  
  plt.title = title
  w1, w2, b1 = get_params(model)
  x1 = np.array([-2, 2])
  x2 = (w1*x1 + b1)/(-w2)
  plt.plot(x1, x2, 'r')
  scatter_plot()
  plt.show()

#%%
plot_fit('Initial Model')

#%%
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#%%
epochs = 1000
losses = []

for i in range(epochs):
  y_pred = model.forward(x_data)
  loss = criterion(y_pred, y_data)
  print("epoch:", i, "loss:", loss.item())
  losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

#%%
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()

#%%
plot_fit("Trained Model")


#%%
point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')
plot_fit("Trained Model")
print("Red point positive probability = {}".format(model.forward(point1).item())) 
print("Black point positive probability = {}".format(model.forward(point2).item())) 
print("Red point belongs in class {}".format(model.predict(point1))) 
print("Black point belongs in class = {}".format(model.predict(point2))) 
# %%
