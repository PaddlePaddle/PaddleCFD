import os
from pathlib import Path

import paddle

"""
This is a modified version of ns_2d.py from https://github.com/zongyi-li/fourier_neural_operator
"""
import argparse
import math
from timeit import default_timer

import scipy.io
from random_fields import GaussianRF
from tqdm import tqdm

from ppcfd.models.g_fno.paddle_utils import set_runtime_device


def navier_stokes_2d(w0, f, domain_size, visc, T, delta_t=0.0001, record_steps=1):
    N = w0.size()[-1]
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)
    w_h = paddle.fft.rfft2(w0)
    f_h = paddle.fft.rfft2(f)
    if len(f_h.size()) < len(w_h.size()):
        f_h = paddle.unsqueeze(f_h, 0)
    record_time = math.floor(steps / record_steps)
    k_y = paddle.cat(
        (
            paddle.arange(start=0, end=k_max, step=1, device=w0.device),
            paddle.arange(start=-k_max, end=0, step=1, device=w0.device),
        ),
        0,
    ).repeat(N, 1)
    k_x = k_y.transpose(0, 1)
    k_x = k_x[..., : k_max + 1]
    k_y = k_y[..., : k_max + 1]
    lap = 4 * math.pi**2 * (k_x**2 + k_y**2) / domain_size**2
    lap[0, 0] = 1.0
    dealias = paddle.unsqueeze(
        paddle.logical_and(
            paddle.abs(k_y) <= 2.0 / 3.0 * k_max, paddle.abs(k_x) <= 2.0 / 3.0 * k_max
        ).float(),
        0,
    )
    sol = paddle.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = paddle.zeros(record_steps, device=w0.device)
    c = 0
    t = 0.0
    for j in range(steps):
        psi_h = w_h / lap
        q = 2.0 * math.pi / domain_size * k_y * 1.0j * psi_h
        q = paddle.fft.irfft2(q, s=(N, N))
        v = -2.0 * math.pi / domain_size * k_x * 1.0j * psi_h
        v = paddle.fft.irfft2(v, s=(N, N))
        w_x = 2.0 * math.pi / domain_size * k_x * 1.0j * w_h
        w_x = paddle.fft.irfft2(w_x, s=(N, N))
        w_y = 2.0 * math.pi / domain_size * k_y * 1.0j * w_h
        w_y = paddle.fft.irfft2(w_y, s=(N, N))
        F_h = paddle.fft.rfft2(q * w_x + v * w_y)
        F_h = dealias * F_h
        w_h = (
            -delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h
        ) / (1.0 + 0.5 * delta_t * visc * lap)
        t += delta_t
        if (j + 1) % record_time == 0:
            w = paddle.fft.irfft2(w_h, s=(N, N))
            sol[..., c] = w
            sol_t[c] = t
            c += 1
    return sol, sol_t


parser = argparse.ArgumentParser()
parser.add_argument("--nu", type=float, required=True)
parser.add_argument("--s", type=int, default=256)
parser.add_argument("--T", type=int, required=True, help="Time horizon")
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--bsize", type=int, default=20)
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument(
    "--ntest", type=int, required=True, help="Number of superresolution examples"
)
parser.add_argument("--period", type=int, required=True, help="Period if sym is true")
parser.add_argument(
    "--sym", action="store_true", default=True, help="Use a symmetric forcing term"
)
parser.add_argument("--domain_size", type=float, default=1)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="runtime device, e.g. auto, cpu, gpu, xpu, npu, gcu",
)
args = parser.parse_args()
device = set_runtime_device(args.device)
s = args.s
N = args.N
GRF = GaussianRF(2, s, args.domain_size, alpha=2.5, tau=7, device=device)
t = paddle.linspace(0, args.domain_size, s + 1, device=device)
t = t[0:-1]
X, Y = paddle.meshgrid(t, t, indexing="ij")
if args.sym:
    f = 0.1 * (
        paddle.cos(args.period * math.pi * X) + paddle.cos(args.period * math.pi * Y)
    )
else:
    f = 0.1 * (paddle.sin(2 * math.pi * (X + Y)) + paddle.cos(2 * math.pi * (X + Y)))
record_steps = args.T * 4
a = paddle.zeros(N, s, s)
u = paddle.zeros(N, s, s, record_steps)
bsize = args.bsize
c = 0
t0 = default_timer()
for j in tqdm(range(N // bsize)):
    w0 = GRF.sample(shape=bsize)
    sol, sol_t = navier_stokes_2d(
        w0, f, args.domain_size, args.nu, args.T, 0.0001, record_steps
    )
    a[c : c + bsize, ...] = w0
    u[c : c + bsize, ...] = sol
    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)
a_super = a[-args.ntest :]
u_super = u[-args.ntest :]
space_sub = s // 64
time_sub = 4
a = a[..., ::space_sub, ::space_sub]
u = u[..., ::space_sub, ::space_sub, ::time_sub]
if args.sym:
    data_name = f"ns_V{args.nu}_N{args.N}_T{args.T}_cos{args.period}{'_' + args.suffix if args.suffix is not None else ''}.mat"
else:
    data_name = f"ns_V{args.nu}_N{args.N}_T{args.T}_sin{'_' + args.suffix if args.suffix is not None else ''}.mat"
super_name = data_name[:-4] + "_super.mat"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
save_dir = os.path.join(args.save_path, data_name)
super_dir = os.path.join(args.save_path, super_name)
scipy.io.savemat(
    save_dir,
    mdict={"a": a.cpu().numpy(), "u": u.cpu().numpy(), "t": sol_t.cpu().numpy()},
)
scipy.io.savemat(
    super_dir,
    mdict={
        "a": a_super.cpu().numpy(),
        "u": u_super.cpu().numpy(),
        "t": sol_t.cpu().numpy(),
    },
)
