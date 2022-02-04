#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:36:58 2019

@author: Jens
"""

from __future__ import print_function

"""
This file is used to calculate the time integration of the 2D advection equation:
    
    df/dt = -Vx df/dx - Vy df/dy
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation


def minmod(a, b):
    """
    Minmod slope limiter

    INPUT
    =====
    a : array
    b : array

    OUTPUT
    ======
    minmod_array : array containing the individual minmod slopes
    """

    n = a.shape[0]

    minmod_array = np.zeros(n)

    for i in range(n):
        if abs(a[i]) < abs(b[i]) and a[i]*b[i] > 0:
            minmod_array[i] = a[i]
        elif abs(a[i]) > abs(b[i]) and a[i]*b[i] > 0:
            minmod_array[i] = b[i]
        else:
            pass

    return minmod_array


def maxmod(a, b):
    """
    Maxmod slope limiter

    INPUT
    =====
    a : array
    b : array

    OUTPUT
    ======
    maxmod_array : array containing the individual maxmod slopes
    """

    n = a.shape[0]

    A = np.greater(np.multiply(a, b), np.zeros(n)).astype(int)
    B = np.greater(np.absolute(a), np.absolute(b)).astype(int)

    C = np.where(B == 1, np.multiply(a, B), np.multiply(b, np.where(B == 1, 0, 1)))
    maxmod_array = np.multiply(C, A)

    #    maxmod_array = np.zeros(n)
    #    for i in range(n):
    #        if abs(a[i]) < abs(b[i]) and a[i]*b[i] > 0:
    #            maxmod_array[i] = b[i]
    #        elif abs(a[i]) > abs(b[i]) and a[i]*b[i] > 0:
    #            maxmod_array[i] = a[i]
    #        else:
    #            pass

    return maxmod_array


def advection_2D_MUSCL(f, V0, dt, t, h, slope_type, dim):
    """
    Estimation for the left and right Riemann solution 2D advection equation
    for the finite volume scheme (MUSCL) along one axis.

    INPUT
    =====
    f : 2D array containing the current shape
    V0 : float (velocity)
    dt : float (time interval)
    t : current time
    h : float (cell size)
    slope_type : str (type of the slope limiter to use: minmod, superbee)
    dim : int (dimension along which to solve, either 0 or 1)

    OUTPUT
    ======
    f_next : array containing the updated shape
    """

    fs = f.shape[0]
    slope = np.zeros((fs, fs))
    flux = np.zeros((fs, fs))
    f_next = np.zeros((fs, fs))

    # Calculating the slope in each cell using minmod slope limiter
    if slope_type == "minmod":
        if dim == 0:
            for i in range(fs):
                slope[i][:] = minmod((f[i][:] - f[i - 1][:])/h, (f[(i + 1)%fs][:] - f[i][:])/h)
        else:
            for i in range(fs):
                slope[:][i] = minmod((f[:][i] - f[:][i - 1])/h, (f[:][(i + 1)%fs] - f[:][i])/h)
    elif slope_type == "superbee":
        if dim == 0:
            for i in range(fs):
                A = minmod((f[(i + 1)%fs][:] - f[i][:])/h, 2*(f[i][:] - f[i - 1][:])/h)
                B = minmod((f[i][:] - f[i - 1][:])/h, 2*(f[(i + 1)%fs][:] - f[i][:])/h)
                slope[i][:] = maxmod(A, B)
        else:
            for i in range(fs):
                A = minmod((f[:][(i + 1)%fs] - f[:][i])/h, 2*(f[:][i] - f[:][i - 1])/h)
                B = minmod((f[:][i] - f[:][i - 1])/h, 2*(f[:][(i + 1)%fs] - f[:][i])/h)
                slope[:][i] = maxmod(A, B)
    else:
        raise Exception("Unknown slope limiter '" + slope_type +
                        "'! Use either 'minmod' or 'superbee'.")

    # Calculating the flux
    if dim == 0:
        if V0 > 0:  # from the left
            for i in range(fs):
                flux[i][:] = (f[i - 1][:] + (1/2)*h*(1 - V0*dt/h)*slope[i - 1][:])*V0
        else:  # from the right
            for i in range(fs):
                flux[i][:] = (f[i][:] - (1/2)*h*(1 + V0*dt/h)*slope[i][:])*V0
    else:
        if V0 > 0:  # from the left
            for i in range(fs):
                flux[:][i] = (f[:][i] + (1/2)*h*(1 - V0*dt/h)*slope[:][i])*V0
        else:  # from the right
            for i in range(fs):
                flux[:][i] = (f[:][i] - (1/2)*h*(1 + V0*dt/h)*slope[:][i])*V0

    if dim == 0:
        for i in range(fs):
            f_next[i][:] = f[i][:] + (dt/h)*(flux[i][:] - flux[(i + 1)%fs][:])
    else:
        for i in range(fs):
            f_next[:][i] = f[:][i] + (dt/h)*(flux[:][i] - flux[:][(i + 1)%fs])

    return f_next


def advection_2D_integration(n_steps, f, V0_arr, h, dt, slope_limiter):
    """
    Performs the time integration of the 2D advection equation for a given
    initial shape, velocity, cell size and time step using the operator
    splitting technique

    INPUT
    =====
    n_steps : int (number of steps)
    f : 2D array containing the (initial) shape of the function to advect
    V0_arr : array (advection velocity in x and y direction)
    h : float (cell size)
    dt : float (time step)
    slope_limiter : str

    OUTPUT
    ======
    result : 3D array containing the final result of the advection (the third
             dimension corresponds to the different times)
    """

    fs = f.shape[0]
    result = np.zeros((fs, fs, n_steps))
    result[:, :, 0] = f
    t = 0

    for i in range(1, n_steps):
        t += dt
        temp_res1 = advection_2D_MUSCL(f, V0_arr[0], dt/2, t, h, slope_limiter, 0)
        temp_res2 = advection_2D_MUSCL(temp_res1, V0_arr[1], dt, t, h, slope_limiter, 1)
        temp_res = advection_2D_MUSCL(temp_res2, V0_arr[0], dt/2, t, h, slope_limiter, 0)
        result[:, :, i] = temp_res
        f = temp_res

    return result


# def step_function(x):
#    """
#    Step function for the initial shape
#
#    INPUT
#    =====
#    x : array
#
#    OUTPUT
#    ======
#    f : array
#    """
#
#    f = np.zeros(x.shape)
#    for i in range(x.shape[0]):
#        if x[i] < 0.4:
#            f[i] = 1
#        elif 0.4 < x[i] and x[i] < 0.6:
#            f[i] = 2
#
#    return f

def gaussian(x, y):
    """
    Gaussian function for the initial shape

    INPUT
    =====
    x : 2D array
    y : 2D array

    OUTPUT
    ======
    f : 2D array
    """

    f = 1 + np.exp(-((x - 0.5)**2 + (y - 0.5)**2)/(0.1**2))

    return f


def animate_results(X, Y, Z, fps, slope_type):
    """
    Animation and plotting of certain tasks

    INPUT
    =====
    X : x coordinates
    Y : y coordinates
    Z : 3D array containing the shapes for the different times
    fps : int
    slope_type : str

    OUTPUT
    ======
    ani : animation reference
    """

    global Xdata
    global Ydata

    Xdata = X
    Ydata = Y

    time_str = "Step: " + str(0) + "/" + str(n_steps)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.canvas.set_window_title("AST246: Hydro project - Optional task (2D Advection equation)")
    #    line1, = ax.plot_surface([], [], [], color='grey', label='Original shape')
    line = [ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.jet, label='Operator splitting with ' + slope_type)]

    #    lines = [line1, line2]

    #    def init():
    #        for line in lines:
    #            line.set_data([],[],[])
    #        return lines

    def update(i, z, line):
        time_str = "Step: " + str(i) + "/" + str(n_steps)
        timer.set_text(time_str)
        line[0].remove()
        line[0] = ax.plot_surface(X, Y, z[:, :, i], cmap=cm.jet)

    minz = np.amin(Z)
    maxz = np.amax(Z)
    ax.set_zlim([minz - abs(maxz - minz)/2, 1.2*maxz])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')

    timer = ax.set_title(time_str)

    ani = animation.FuncAnimation(fig, update, fargs=(Z, line), frames=n_steps, interval=1000/fps)

    return ani


if __name__ == '__main__':
    global x
    global h
    global n_steps
    global f_ini

    Nx = 100  # number of points / cells
    x, h = np.linspace(0, 1, Nx, retstep=True)
    y = x

    # 2D grid
    [X, Y] = np.meshgrid(x, y)

    # initial velocities in x and y direction
    Vx, Vy = 1, 0
    V0 = np.array([Vx, Vy])

    # time step
    dt = 0.5*h**2/np.sqrt((h*Vx)**2 + (h*Vy)**2)

    # number of integration steps
    n_steps = 300

    # initial shape
    f_ini = gaussian(X, Y)

    # slope limiter
    sl = "superbee"

    advection_2D = advection_2D_integration(n_steps, f_ini, V0, h, dt, sl)

    """
    Plotting the desired results results & saving to a file
    """

    fps = 40
    outer_ani = animate_results(X, Y, advection_2D, fps, sl)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=-1)
    outer_ani.save('hydro_task_optional.mp4', writer)
