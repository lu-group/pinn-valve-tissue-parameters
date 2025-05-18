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
data = np.load("../data/3D_cone_data.npy", allow_pickle="TRUE")
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]

ux_mean, ux_std = np.mean(gt_disp[:, 0]), np.std(gt_disp[:, 0])
uy_mean, uy_std = np.mean(gt_disp[:, 1]), np.std(gt_disp[:, 1])
uz_mean, uz_std = np.mean(gt_disp[:, 2]), np.std(gt_disp[:, 2])

# Extract sampling points
idx1 = np.random.choice(np.where(coors[:, 2:3] > 0.75)[0], 2000, replace=False)
idx2 = np.random.choice(
    np.where((coors[:, 2:3] > 0.5) & (coors[:, 2:3] < 0.75))[0], 1500, replace=False
)
idx3 = np.random.choice(
    np.where((coors[:, 2:3] > 0.25) & (coors[:, 2:3] < 0.5))[0], 1000, replace=False
)
idx4 = np.random.choice(
    np.where((coors[:, 2:3] > 0) & (coors[:, 2:3] < 0.25))[0], 500, replace=False
)

pde_pts = np.vstack((coors[idx1, :], coors[idx2, :], coors[idx3, :], coors[idx4, :]))
pde_pts_disp = np.vstack(
    (gt_disp[idx1, :], gt_disp[idx2, :], gt_disp[idx3, :], gt_disp[idx4, :])
)

geom = dde.geometry.PointCloud(points=pde_pts)

loss = [dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1, 2])]

# Model variables
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


# Compute strain tensor from displacements
def strain(x, y):
    ux, uy, uz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    exy = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)
    )
    exz = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=2) + dde.grad.jacobian(uz, x, i=0, j=0)
    )
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    eyz = 0.5 * (
        dde.grad.jacobian(uy, x, i=0, j=2) + dde.grad.jacobian(uz, x, i=0, j=1)
    )
    ezz = dde.grad.jacobian(uz, x, i=0, j=2)

    return exx, exy, exz, eyy, eyz, ezz


# Compute stress tensor from displacements
def stress(x, y):
    exx, exy, exz, eyy, eyz, ezz = strain(x, y)

    E = (torch.tanh(E_) + 1.5) * 3000
    nu = (torch.tanh(nu_) + 1) * 0.25

    c1 = E / (1 + nu)
    c2 = E / (1 - 2 * nu)

    sxx = c1 * (exx + c2 * (exx + eyy + ezz))
    sxy = c1 * exy
    sxz = c1 * exz
    syy = c1 * (eyy + c2 * (exx + eyy + ezz))
    syz = c1 * eyz
    szz = c1 * (ezz + c2 * (exx + eyy + ezz))

    return sxx, sxy, sxz, syy, syz, szz, E


# Define the governing equations to constrain the networks
def pde(x, y):
    sxx, sxy, sxz, syy, syz, szz, E = stress(x, y)
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxz_z = dde.grad.jacobian(Nsxz, x, i=0, j=2)

    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    syz_z = dde.grad.jacobian(Nsyz, x, i=0, j=2)

    sxz_x = dde.grad.jacobian(Nsxz, x, i=0, j=0)
    syz_y = dde.grad.jacobian(Nsyz, x, i=0, j=1)
    szz_z = dde.grad.jacobian(Nszz, x, i=0, j=2)

    mx = sxx_x + sxy_y + sxz_z
    my = sxy_x + syy_y + syz_z
    mz = sxz_x + syz_y + szz_z

    stress_xx = sxx - Nsxx
    stress_yy = syy - Nsyy
    stress_zz = szz - Nszz
    stress_xy = sxy - Nsxy
    stress_xz = sxz - Nsxz
    stress_yz = syz - Nsyz

    return [
        mx,
        my,
        mz,
        stress_xx,
        stress_yy,
        stress_zz,
        stress_xy,
        stress_xz,
        stress_yz,
    ]


net = MPFNN([3, 32, 16, 8, 3], [3, 32, 16, 8, 6], "swish", "Glorot normal")


# Nondimensionalize network output variables
def output_transform(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    Nux = Nux * ux_std + ux_mean
    Nuy = Nuy * uy_std + uy_mean
    Nuz = Nuz * uz_std + uz_mean

    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    return torch.concat([Nux, Nuy, Nuz, Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz], axis=1)


net.apply_output_transform(output_transform)

data = dde.data.PDE(
    geom,
    pde,
    loss,
    anchors=pde_pts,
)

model = dde.Model(data, net)
external_trainable_variables = [E_, nu_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    lr=1e-3,
    decay=["step", 5000, 0.66],
    loss_weights=[1e-4] * 9 + [1],
    external_trainable_variables=external_trainable_variables,
)
losshistory, train_state = model.train(
    epochs=250000, display_every=1000, callbacks=[variables]
)
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
E_true = 5000
nu_true = 0.3
E_pred = (np.tanh(vkinfer[:, 0]) + 1.5) * 3000
nu_pred = (np.tanh(vkinfer[:, 1]) + 1) * 0.25

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
