from test import test
from train import train
from model import Model
import torch
from plot import plot

model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)

model, loss_history = train(model, 200)
plot(loss_history)

"""111"""