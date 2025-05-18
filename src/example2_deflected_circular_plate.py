import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
from deepxde.backend import torch

dde.config.set_random_seed(2024)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Read and organize input data
data = np.load("../data/deflected_circular_plate_data.npy", allow_pickle="TRUE")
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]

uz_mean, uz_std = np.mean(gt_disp), np.std(gt_disp)

# Extract sampling points
idx = np.random.choice(np.where(coors)[0], 5000, replace=False)
pde_pts, pde_pts_disp = coors[idx, :], gt_disp[idx, :]

geom = dde.geometry.PointCloud(points=pde_pts)

loss = [
    dde.PointSetBC(pde_pts, pde_pts_disp, component=0),
]

# Model variables
q = -0.001  # MPa
h = 0.05  # Half plate thickness
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


# Define the governing equations to constrain the networks
def pde(x, y):
    Nuz = y[:, 0:1]

    E = torch.tanh(E_) + 1
    nu = (torch.tanh(nu_) + 1.0) / 4
    D = 2 * h**3 * E / (3 * (1**2 - nu**2))

    d2uz_dx2 = dde.grad.hessian(Nuz, x, i=0, j=0)
    d2uz_dy2 = dde.grad.hessian(Nuz, x, i=1, j=1)
    d2uz_dxy = dde.grad.hessian(Nuz, x, i=0, j=1)

    d4uz_dx4 = dde.grad.hessian(d2uz_dx2, x, i=0, j=0)
    d4uz_dy4 = dde.grad.hessian(d2uz_dy2, x, i=1, j=1)
    d4uz_dx2y2 = dde.grad.hessian(d2uz_dxy, x, i=0, j=1)

    biharmonic = d4uz_dx4 + 2 * d4uz_dx2y2 + d4uz_dy4 - q / D

    return biharmonic


data = dde.data.PDE(geom, pde, loss, anchors=pde_pts)


# Nondimensionalize network output variables
def output_transform(x, y):
    Nuz = y[:, 0:1]
    Nuz = Nuz * uz_std + uz_mean

    return torch.concat([Nuz], axis=1)


net = dde.nn.FNN([3] + [45] * 5 + [1], "swish", "Glorot normal")

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
external_trainable_variables = [E_, nu_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    lr=1e-3,
    decay=["step", 15000, 0.15],
    loss_weights=[1] + [1e3],
    external_trainable_variables=external_trainable_variables,
)
losshistory, train_state = model.train(epochs=100000, callbacks=[variables])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

import re

lines = open("variables.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
E_true = 1
nu_true = 0.3
E_pred = np.tanh(vkinfer[:, 0]) + 1.0
nu_pred = (np.tanh(vkinfer[:, 1]) + 1.0) / 4

print(
    "E prediction: ",
    E_pred[-1],
    "percentage error (%): ",
    np.linalg.norm(E_true - E_pred[-1]) / np.linalg.norm(E_true) * 100,
)
print(
    "nu prediction: ",
    nu_pred[-1],
    "percentage error (%): ",
    np.linalg.norm(nu_true - nu_pred[-1]) / np.linalg.norm(nu_true) * 100,
)
