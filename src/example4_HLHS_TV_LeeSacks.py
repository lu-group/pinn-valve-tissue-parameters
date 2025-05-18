import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
from deepxde.backend import torch
from deepxde.nn import activations

dde.config.set_random_seed(2024)

if torch.cuda.is_available():
    torch.cuda.set_device(0)


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
data = np.load("../data/HLHS_TV_data.npy", allow_pickle="TRUE")
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]

coors_t0 = np.hstack((coors, np.zeros((len(coors), 1))))
coors_t1 = np.hstack((coors, 0.1 * np.ones((len(coors), 1))))

ux_mean, uy_mean, uz_mean = (
    np.mean(gt_disp[:, 0]),
    np.mean(gt_disp[:, 1]),
    np.mean(gt_disp[:, 2]),
)
ux_std, uy_std, uz_std = (
    np.std(gt_disp[:, 0]),
    np.std(gt_disp[:, 1]),
    np.std(gt_disp[:, 2]),
)

# Extract sampling points
idx1 = np.random.choice(np.where(coors_t0)[0], 2500, replace=False)
idx2 = np.random.choice(np.where(coors_t1)[0], 2500, replace=False)
pde_pts = np.vstack((coors_t0[idx1, :], coors_t1[idx2, :]))
pde_pts_disp = np.vstack((np.zeros((len(coors_t0[idx1, :]), 3)), gt_disp[idx2, :]))

geomtime = dde.geometry.PointCloud(points=pde_pts)

loss = [
    dde.PointSetBC(
        coors_t0[idx1, :], np.zeros((len(coors[idx1, :]), 3)), component=[0, 1, 2]
    ),
    dde.PointSetBC(
        coors_t1[idx2, :], coors_t1[idx2, :3] + gt_disp[idx2, :], component=[9, 10, 11]
    ),
]

# Model variables
c0_ = dde.Variable(1.0)
c1_ = dde.Variable(1.0)
c2_ = dde.Variable(1.0)


def stress(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
    duxdy = dde.grad.jacobian(Nux, x, i=0, j=1)
    duxdz = dde.grad.jacobian(Nux, x, i=0, j=2)

    duydx = dde.grad.jacobian(Nuy, x, i=0, j=0)
    duydy = dde.grad.jacobian(Nuy, x, i=0, j=1)
    duydz = dde.grad.jacobian(Nuy, x, i=0, j=2)

    duzdx = dde.grad.jacobian(Nuz, x, i=0, j=0)
    duzdy = dde.grad.jacobian(Nuz, x, i=0, j=1)
    duzdz = dde.grad.jacobian(Nuz, x, i=0, j=2)

    Fxx = duxdx + 1.0
    Fxy = duxdy
    Fxz = duxdz

    Fyx = duydx
    Fyy = duydy + 1.0
    Fyz = duydz

    Fzx = duzdx
    Fzy = duzdy
    Fzz = duzdz + 1.0

    detF = (
        Fxx * (Fyy * Fzz - Fyz * Fzy)
        - Fxy * (Fyx * Fzz - Fyz * Fzx)
        + Fxz * (Fyx * Fzy - Fyy * Fzx)
    )

    detF = torch.where(torch.le(detF, 0), 0.001, detF)

    adjFxx = Fyy * Fzz - Fyz * Fzy
    adjFxy = -(Fxy * Fzz - Fxz * Fzy)
    adjFxz = Fxy * Fyz - Fxz * Fyy

    adjFyx = -(Fyx * Fzz - Fyz * Fzx)
    adjFyy = Fxx * Fzz - Fxz * Fzx
    adjFyz = -(Fxx * Fyz - Fxz * Fyz)

    adjFzx = Fyx * Fzy - Fzx * Fyy
    adjFzy = -(Fxx * Fzy - Fxy * Fzx)
    adjFzz = Fxx * Fyy - Fxy * Fyx

    invFxx = adjFxx / detF
    invFxy = adjFxy / detF
    invFxz = adjFxz / detF

    invFyx = adjFyx / detF
    invFyy = adjFyy / detF
    invFyz = adjFyz / detF

    invFzx = adjFzx / detF
    invFzy = adjFzy / detF
    invFzz = adjFzz / detF

    c0 = (torch.tanh(c0_) + 1.0) * 100
    c1 = (torch.tanh(c1_) + 1.0) * 100
    c2 = (torch.tanh(c2_) + 1.0) * 10

    Cxx = Fxx**2 + Fyx**2 + Fzx**2
    Cyy = Fxy**2 + Fyy**2 + Fzy**2
    Czz = Fxz**2 + Fyz**2 + Fzz**2

    I1 = Cxx + Cyy + Czz

    # I1 has to be less than 3.8
    I1 = torch.where((I1 - 3) > 0.8, 3.8, I1)

    coeff = c0 + c1 * c2 * (I1 - 3) * torch.exp(c2 * (I1 - 3) ** 2)

    Pxx = coeff * Fxx
    Pyy = coeff * Fyy
    Pzz = coeff * Fzz

    Pxy = coeff * Fxy
    Pyx = coeff * Fyx
    Pxz = coeff * Fxz
    Pzx = coeff * Fzx
    Pyz = coeff * Fyz
    Pzy = coeff * Fzy

    # Cauchy stress
    sxx = invFxx * Pxx + invFxy * Pyx + invFxz * Pzx
    sxy = invFxx * Pxy + invFxy * Pyy + invFxz * Pzy
    sxz = invFxx * Pxz + invFxy * Pyz + invFxz * Pzz
    syy = invFyx * Pxy + invFyy * Pyy + invFyz * Pzy
    syz = invFyx * Pxz + invFyy * Pyz + invFyz * Pzz
    szz = invFzx * Pxz + invFzy * Pyz + invFzz * Pzz

    return sxx, sxy, sxz, syy, syz, szz


def pde(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    sxx, sxy, sxz, syy, syz, szz = stress(x, y)

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxz_z = dde.grad.jacobian(Nsxz, x, i=0, j=2)

    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    syz_z = dde.grad.jacobian(Nsyz, x, i=0, j=2)

    sxz_x = dde.grad.jacobian(Nsxz, x, i=0, j=0)
    syz_y = dde.grad.jacobian(Nsyz, x, i=0, j=1)
    szz_z = dde.grad.jacobian(Nszz, x, i=0, j=2)

    rho = 1e-6
    d2x_dt2 = dde.grad.hessian(Nux, x, i=3, j=3)
    d2y_dt2 = dde.grad.hessian(Nuy, x, i=3, j=3)
    d2z_dt2 = dde.grad.hessian(Nuz, x, i=3, j=3)

    mx = sxx_x + sxy_y + sxz_z - rho * d2x_dt2
    my = sxy_x + syy_y + syz_z - rho * d2y_dt2
    mz = sxz_x + syz_y + szz_z - rho * d2z_dt2

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


net = MPFNN([4, 32, 16, 8, 3], [4, 32, 16, 8, 6], "swish", "Glorot normal")


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

    Nux_new = Nux + x[:, 0:1]
    Nuy_new = Nuy + x[:, 1:2]
    Nuz_new = Nuz + x[:, 2:3]

    return torch.concat(
        [Nux, Nuy, Nuz, Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz, Nux_new, Nuy_new, Nuz_new],
        axis=1,
    )


def hausdorff_distance(y_true, y_pred):

    distances = torch.cdist(y_pred, y_true, p=2)
    avg_distances_1 = torch.mean(
        torch.min(distances, dim=1).values
    )  # Max of min distances from 1 to 2
    avg_distances_2 = torch.mean(
        torch.min(distances, dim=0).values
    )  # Max of min distances from 2 to 1
    error = 0.5 * (avg_distances_1 + avg_distances_2)

    return error


net.apply_output_transform(output_transform)
data = dde.data.PDE(geomtime, pde, loss, anchors=pde_pts)

model = dde.Model(data, net)
loss = ["MSE"] * 10 + [hausdorff_distance]
model = dde.Model(data, net)
external_trainable_variables = [c0_, c1_, c2_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    loss=loss,
    lr=1e-3,
    decay=["step", 15000, 0.61],
    loss_weights=[1e-4] * 10 + [1],
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
c0_pred = (np.tanh(vkinfer[:, 0]) + 1) * 100
c1_pred = (np.tanh(vkinfer[:, 1]) + 1) * 100
c2_pred = (np.tanh(vkinfer[:, 2]) + 1) * 10

print("c0 prediction: ", c0_pred[-1])
print("c1 prediction: ", c1_pred[-1])
print("c2 prediction: ", c2_pred[-1])
