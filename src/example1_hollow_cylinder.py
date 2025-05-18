import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
from deepxde.backend import torch
from deepxde.nn import activations

dde.config.set_random_seed(2024)

if torch.cuda.is_available():
    torch.cuda.set_device(0)


# Create a class for parallel network in DeepXDE
class MPFNN(dde.nn.PFNN):
    def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
        self.first_layer_sizes = layer_sizes
        self.second_layer_sizes = second_layer_sizes
        self.activation = activations.get(activation)

        self.firstFNN = dde.nn.PFNN(
            self.first_layer_sizes, self.activation, kernel_initializer
        )
        self.secondFNN = dde.nn.PFNN(
            self.second_layer_sizes, self.activation, kernel_initializer
        )

    def forward(self, inputs):
        x = inputs

        if self._input_transform is not None:
            x = self._input_transform(x)

        x_firstFNN = self.firstFNN(x)
        x_secondFNN = self.secondFNN(x)

        x = torch.cat((x_firstFNN, x_secondFNN), dim=1)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x


# Read and organize input data
data = np.load("../data/hollow_cylinder_data.npy", allow_pickle="TRUE")
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]
bnd_coors, bond_gt_disp = (
    data.item()["boundary_coordinates"],
    data.item()["boundary_displacements"],
)
radius = data.item()["radius"]

scale = 1e4
gt_disp = gt_disp * scale
bond_gt_disp = bond_gt_disp * scale

ux_mean, uy_mean = np.mean(gt_disp[:, 0]), np.mean(gt_disp[:, 1])
ux_std, uy_std = np.std(gt_disp[:, 0]), np.std(gt_disp[:, 1])

# Extract sampling points
idx1 = np.random.choice(np.where(radius < 2)[0], 500, replace=False)
idx2 = np.random.choice(np.where((radius > 2) & (radius < 3))[0], 400, replace=False)
idx3 = np.random.choice(np.where((radius > 3) & (radius < 4))[0], 300, replace=False)
idx4 = np.random.choice(np.where((radius > 4) & (radius <= 5))[0], 200, replace=False)
idx5 = np.random.choice(np.where(bnd_coors)[0], 150, replace=False)

pde_pts = np.vstack(
    (coors[idx1, :], coors[idx2, :], coors[idx3, :], coors[idx4, :], bnd_coors[idx5, :])
)
pde_pts_disp = np.vstack(
    (
        gt_disp[idx1, :],
        gt_disp[idx2, :],
        gt_disp[idx3, :],
        gt_disp[idx4, :],
        bond_gt_disp[idx5, :],
    )
)

geom = dde.geometry.PointCloud(points=pde_pts)

losses = [
    dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1]),
]

# Model variables
p_in = 1e-5 * scale
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


# Compute strain tensor from displacements
def strain(x, y):
    ux, uy = y[:, 0:1], y[:, 1:2]

    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    exy = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)
    )

    return exx, eyy, exy


# Compute stess tensor from displacements
def stress(x, y):
    exx, eyy, exy = strain(x, y)

    E = (torch.tanh(E_) + 1.0) / 2
    nu = (torch.tanh(nu_) + 1.0) / 4

    sxx = E / (1 - nu**2) * (exx + nu * eyy)
    syy = E / (1 - nu**2) * (eyy + nu * exx)
    sxy = E / (1 + nu) * exy

    return sxx, syy, sxy, E


# Define the governing equations to constrain the networks
def pde(x, y):
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)

    mx = sxx_x + sxy_y
    my = sxy_x + syy_y

    sxx, syy, sxy, E = stress(x, y)

    nx = torch.cos(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    ny = torch.sin(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    srr = sxx * torch.square(nx) + syy * torch.square(ny) + sxy * 2 * nx * ny

    stress_x = sxx - Nsxx
    stress_y = syy - Nsyy
    stress_xy = sxy - Nsxy
    stress_rr = srr - Nsrr

    return mx, my, stress_x, stress_y, stress_xy, stress_rr


data = dde.data.PDE(
    geom,
    pde,
    losses,
    anchors=pde_pts,
)


# Nondimensionalize network output variables
def output_transform(x, y):
    Nux, Nuy = y[:, 0:1], y[:, 1:2]
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]

    Nux = x[:, 0:1] * (Nux * ux_std + ux_mean)
    Nuy = x[:, 1:2] * (Nuy * uy_std + uy_mean)

    rad = torch.sqrt(torch.square(x[:, 0:1]) + torch.square(x[:, 1:2]))

    Nsxx = (rad - 5) * Nsxx
    Nsyy = (rad - 5) * Nsyy
    Nsxy = x[:, 0:1] * x[:, 1:2] * Nsxy
    Nsxy = (rad - 5) * Nsxy

    Nsrr = ((rad - 1) * Nsrr - 1) * p_in / -4 * (rad - 5)

    return torch.concat([Nux, Nuy, Nsxx, Nsyy, Nsxy, Nsrr], axis=1)


net = MPFNN([2] + [45] * 5 + [2], [2] + [45] * 5 + [4], "swish", "Glorot normal")

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
    loss_weights=[1] * 2 + [1e1] * 4 + [1],
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
E_true = 1.35e-1
nu_true = 0.3
E_pred = (np.tanh(vkinfer[:, 0]) + 1.0) / 2
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
