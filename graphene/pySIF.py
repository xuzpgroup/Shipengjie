from ase.io import read, write
import matplotlib.pyplot as plt
import numpy as np
from ase.build import make_supercell
from scipy.optimize import minimize
import alphashape
import math

# written by Shi Pengjie at 2022/08/01

def crack_position(atom, atom0,al=0.6):
    # Coordinates origin setting
    pos = atom.get_positions()[:, :2]
    pos[:, 0] -= pos[100, 0]
    pos[:, 1] -= pos[100, 1]
    pos0 = atom0.get_positions()[:, :2]
    pos0[:, 0] -= pos0[100, 0]
    pos0[:, 1] -= pos0[100, 1]

    # Adjust the coordinates because of the PBC condition at y direction
    displacement = pos - pos0
    ly = np.array(atom.get_cell())[1, 1]
    I_up = np.argwhere(displacement[:, 1] > ly/3).flatten()
    I_down = np.argwhere(displacement[:, 1] < -ly/3).flatten()
    pos[I_up, 1] -= ly
    pos[I_down, 1] += ly

    # Extract the crack close to the crack tip & crack tip
    alpha_shape = alphashape.alphashape(pos, al)
    # print(0.5)
    bd = [np.min(pos[:, 0]), np.max(pos[:, 0]), np.min(pos[:, 1]), np.max(pos[:, 1])]
    boundary = np.array(alpha_shape.boundary.coords)
    I = np.argwhere((boundary[:, 0] > bd[0] + 5) & (boundary[:, 0] < bd[1] - 5) &
                    (boundary[:, 1] > bd[2] + 5) & (boundary[:, 1] < bd[3] - 5)).flatten()

    index_crack = []
    for ii in I:
        cc = np.argwhere(
            (np.abs(pos[:, 0] - boundary[ii, 0]) < 1e-5) & (np.abs(pos[:, 1] - boundary[ii, 1]) < 1e-5)).flatten()[0]
        index_crack.append(cc)
    index_crack = np.array(index_crack)
    """
    tip_total = pos0[index_crack, 0].argsort()[-2:][::-1]
    if pos0[index_crack[tip_total[0]], 0] < pos0[index_crack[tip_total[1]], 0]:
        tip = index_crack[tip_total[0]]
    elif pos0[index_crack[tip_total[0]], 0] >= pos0[index_crack[tip_total[1]], 0]:
        tip = index_crack[tip_total[1]]
    """
    tip = index_crack[np.argmax(pos0[index_crack, 0])]

    dx = pos0[index_crack, 0] - pos0[tip, 0]
    dy = pos0[index_crack, 1] - pos0[tip, 1]
    ly0 = atom0.get_cell()[1, 1]
    dy[dy > ly0 / 2] -= ly0
    dy[dy < -ly0 / 2] += ly0
    r_tip = np.sqrt((dx) ** 2 + (dy) ** 2)
    index_crack = index_crack[r_tip < 50]

    # Extract the crack direction
    p = np.polyfit(pos0[index_crack, 0], pos0[index_crack, 1], 1)
    theta = np.arctan(p[0])

    # re-adjust the coordinates with the orgin is crack tip
    pos[:, 0] -= pos[tip, 0]
    pos[:, 1] -= pos[tip, 1]
    pos0[:, 0] -= pos0[tip, 0]
    pos0[:, 1] -= pos0[tip, 1]

    return pos, pos0, index_crack, theta, tip


def extract_circle(pos0, ly0, r_min, r_max):
    dy = pos0[:, 1].copy()
    dy[dy > ly0 / 2] -= ly0
    dy[dy < -ly0 / 2] += ly0
    dx = pos0[:, 0].copy()
    r = np.sqrt((dx) ** 2 + (dy) ** 2)
    circle_index = np.argwhere((r < r_max) & (r > r_min)).flatten()
    return circle_index


def position_remap(pos0, pos, circle_index, ly0, ly):
    pos0_circle = pos0[circle_index, :]
    pos_circle = pos[circle_index, :]

    I = np.argwhere(pos0_circle[:, 1] > ly0 / 2).flatten()
    pos0_circle[I, 1] -= ly0
    I = np.argwhere(pos0_circle[:, 1] < -ly0 / 2).flatten()
    pos0_circle[I, 1] += ly0

    I = np.argwhere(pos_circle[:, 1] > ly / 2).flatten()
    pos_circle[I, 1] -= ly
    I = np.argwhere(pos_circle[:, 1] < -ly / 2).flatten()
    pos_circle[I, 1] += ly
    displacement_circle = pos_circle - pos0_circle

    return pos0_circle, displacement_circle


def rotate_circle(p, theta):
    rotate_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(p, rotate_matrix.T)


def get_r_theta(pos0_circle):
    r = np.sqrt(pos0_circle[:, 0] ** 2 + pos0_circle[:, 1] ** 2)
    theta_circle = []
    for p in pos0_circle:
        theta_circle.append(math.atan2(p[1], p[0]))
    theta_circle = np.array(theta_circle)
    return r, theta_circle


def extract_data(r_circle, theta_circle, displacement_circle, data_range):
    I_half = np.argwhere(np.abs(theta_circle) < np.pi * data_range).flatten()
    theta_circle = theta_circle[I_half]
    displacement_circle = displacement_circle[I_half, :]
    r_circle = r_circle[I_half]
    return r_circle, theta_circle, displacement_circle


def g11_n(n,kai,theta):
    c1 = (kai + n/2 + (-1)**n)*np.cos(n/2*theta) - n/2 * np.cos((n/2-2)*theta)
    return c1


def g12_n(n,kai,theta):
    c1 = (kai-n/2-(-1)**n)*np.sin(n/2*theta) + n/2 *np.sin((n/2-2)*theta)
    return c1


def g21_n(n,kai,theta):
    c1 = -(kai+n/2-(-1)**n)*np.sin(n/2*theta) + n/2 * np.sin((n/2-2)*theta)
    return c1


def g22_n(n,kai,theta):
    c1 = (kai-n/2+(-1)**n)*np.cos(n/2*theta) + n/2* np.cos((n/2-2)*theta)
    return c1


def g11(n,kai,r,theta):
    c = np.zeros((theta.shape[0],n+1))
    for nn in range(n+1):
        c[:,nn] = g11_n(nn,kai,theta)*r**(nn/2)
    return c


def g12(n,kai,r,theta):
    c = np.zeros((theta.shape[0],n+1))
    for nn in range(n+1):
        c[:,nn] = g12_n(nn,kai,theta)*r**(nn/2)
    return c


def g21(n,kai,r,theta):
    c = np.zeros((theta.shape[0],n+1))
    for nn in range(n+1):
        c[:,nn] = g21_n(nn,kai,theta)*r**(nn/2)
    return c


def g22(n,kai,r,theta):
    c = np.zeros((theta.shape[0],n+1))
    for nn in range(n+1):
        c[:,nn] = g22_n(nn,kai,theta)*r**(nn/2)
    return c


def Williams_expansion(n,kai,r,theta):
    kk = theta.shape[0]
    C = np.zeros((r.shape[0]*2,2*n+2))
    C[:kk,:(n+1)] = g11(n,kai,r,theta)
    C[:kk,(n+1):] = g21(n,kai,r,theta)
    C[kk:,:(n+1)] = g12(n,kai,r,theta)
    C[kk:,(n+1):] = g22(n,kai,r,theta)
    return C


def solve_A(n,kai,r,theta,disp):
    disp_concat = np.concatenate([disp[:, 0], disp[:, 1]])
    C = Williams_expansion(n, kai, r, theta)
    Y = np.matmul(C.T, disp_concat)
    CC = np.matmul(C.T, C)
    return np.linalg.solve(CC,Y)


def A_list(n):
    A = np.zeros((2*n+2,))
    A[1], A[n+2] = 1, 1
    return A


def loss_kai(kai0, n, r, theta, disp):
    A_true = solve_A(n, kai0, r, theta, disp)
    disp_pred = np.matmul(Williams_expansion(n, kai0, r, theta), A_true)
    disp_concat = np.concatenate([disp[:, 0], disp[:, 1]])
    return np.sum((disp_concat - disp_pred)**2)