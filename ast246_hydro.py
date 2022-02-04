#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:36:58 2019

@author: Jens
"""
import time

"""
This file is used to ...
1) calculate the 1st order advection equation for the course AST246 in the 
   HS2019 using finite difference & finite volume methods (Task 1)
2) calculate the 2nd order diffusion equation for the course AST246 in the
   HS2019 using finite difference difference method (Task 2)
3) calculate the 2nd order advection-diffusion equation for the couse AST246
   in the HS2019 using the operator splitting technique (Task 3)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def advection_1D_FD(f, V0, dt, t, h):
    """
    Advection equation using finite differencing method (forward Euler).

    INPUT
    =====
    f : array containing the current shape
    V0 : float (velocity)
    dt : float (time step)
    t : current time
    h : float (cell size)

    OUTPUT
    ======
    f_next : array containing the updated shape
    """
    f_next = np.zeros(f.shape)
    f_next[1:] = f[1:] - dt*V0*(f[1:] - f[:-1])/h
    f_next[0] = f_next[-1]  # Periodic boundary.

    return f_next


def minmod(a, b):
    """
    Minmod slope limiter

    INPUT
    =====
    a : float
    b : float

    OUTPUT
    ======
    a or b : float
    """

    if abs(a) < abs(b) and a*b > 0.0:
        return a
    elif abs(b) < abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0


def maxmod(a, b):
    """
    Maxmod slope limiter

    INPUT
    =====
    a : float
    b : float

    OUTPUT
    ======
    a or b : float
    """
    if abs(a) > abs(b) and a*b > 0.0:
        return a
    elif abs(b) > abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0


def advection_1D_MUSCL(f, V0, dt, t, h, slope_type):
    """
    Estimation for the left and right Riemann solution 1D advection equation
    for the finite volume scheme (MUSCL).

    INPUT
    =====
    f : array containing the current shape
    V0 : float (velocity)
    dt : float (time interval)
    t : current time
    h : float (cell size)
    slope_type : str (type of the slope limiter to use: minmod, superbee)

    OUTPUT
    ======
    f_next : array containing the updated shape
    """

    fs = f.shape[0]
    slope = np.zeros(fs)
    flux = np.zeros(fs)
    f_next = np.zeros(fs)

    # Calculating the slope in each cell using minmod slope limiter
    if slope_type == "minmod":
        for i in range(fs):
            slope[i] = minmod((f[i] - f[i - 1])/h, (f[(i + 1)%fs] - f[i])/h)
    elif slope_type == "superbee":
        for i in range(fs):
            A = minmod((f[(i + 1)%fs] - f[i])/h, 2*(f[i] - f[i - 1])/h)
            B = minmod((f[i] - f[i - 1])/h, 2*(f[(i + 1)%fs] - f[i])/h)
            slope[i] = maxmod(A, B)
    elif slope_type == "van_leer":
        for i in range(fs):
            if (f[i] - f[i - 1]) == 0 or (f[(i + 1)%fs] - f[i]) == 0:
                slope[i] = 0
            else:
                r = (f[i] - f[i - 1])/(f[(i + 1)%fs] - f[i])
                slope[i] = (r + np.absolute(r))/(1 + np.absolute(r))
    else:
        raise Exception("Unknown slope limiter '" + slope_type +
                        "'! Use either 'minmod', 'superbee' or 'van_leer'.")

    # Calculating the flux
    if V0 > 0:  # from the left
        for j in range(fs):
            flux[j] = (f[j - 1] + (1/2)*h*(1 - V0*dt/h)*slope[j - 1])*V0
    else:  # from the right
        for j in range(fs):
            flux[j] = (f[j] - (1/2)*h*(1 + V0*dt/h)*slope[j])*V0

    for i in range(fs):
        f_next[i] = f[i] + (dt/h)*(flux[i] - flux[(i + 1)%fs])

    return f_next


def advection_1D_integration(n_steps, f, V0, h, dt, int_type, *args):
    """
    Performs the time integration of the 1D advection equation for a given
    initial shape, velocity, cell size and time step using a given type
    of integration technique (finite differencing (FD) or finite volume (MUSCL))

    INPUT
    =====
    n_steps : int (number of steps)
    f : array containing the (initial) shape of the function to advect
    V0 : float (velocity)
    h : float (cell size)
    dt : float (time step)
    int_type : str (integration type, either "FD" or "MUSCL")

    OUTPUT
    ======
    result : 2D array containing the final result of the advection (the third
             dimension corresponds to the different times)
    """

    fs = f.shape[0]
    result = np.zeros((fs, n_steps))
    result[:, 0] = f
    t = 0

    if int_type == "FD":
        for i in range(1, n_steps):
            t += dt
            temp_res = advection_1D_FD(f, V0, dt, t, h)
            result[:, i] = temp_res
            f = temp_res
    elif int_type == "MUSCL":
        slope_type = args[0]
        for i in range(1, n_steps):
            t += dt
            temp_res = advection_1D_MUSCL(f, V0, dt, t, h, slope_type)
            result[:, i] = temp_res
            f = temp_res
    else:
        raise Exception("Unknown integration type '" + int_type + "'! Use either 'FD' or 'MUSCL'.")

    return result


def diffusion_rk2(t, D, h, f):
    """
    Calculates the RHS of the diffusion equation in order to use the RK2 method.

    INPUT
    =====
    t : float (current time)
    D : float (diffusion coefficient)
    h : float (cell size)
    f : array containing the shape of the function

    OUTPUT
    =====
    f_temp : float (RHS)
    """

    fs = f.shape[0]
    f_temp = np.zeros(fs)

    for i in range(fs):
        f_temp[i] = (D/h**2)*(f[(i + 1)%fs] + f[i - 1] - 2*f[i])

    return f_temp


def diffusion_1D_FD(f, D, dt, t, h):
    """
    Diffusion equation using finite differencing method (RK2).

    INPUT
    =====
    f : array containing the current shape
    D : float (diffusion coefficient)
    dt : float (time step)
    t : current time
    h : float (cell size)

    OUTPUT
    ======
    f_next : array containing the updated shape
    """

    fs = f.shape[0]
    f_next = np.zeros(fs)

    f_temp = diffusion_rk2(t, D, h, f)
    f_next = f + dt*diffusion_rk2(t + dt/2, D, h, f + (dt/2)*f_temp)  # <- is it f+(dt/2)*f_temp or ...
    #    f_next = f + dt*diffusion_rk2(t+dt/2, D, h, f+(h/2)*f_temp) # <- f+(h/2)*f_temp ??
    #    f_next[0] = f_next[-1]

    return f_next


def diffusion_1D_integration(n_steps, f, D, h, dt, int_type, *args):
    """
    Performs the time integration of the 1D diffusion equation for a given
    initial shape, velocity, cell size and time step using a given type
    of integration technique (finite differencing (FD) or operator splitting (OS))

    INPUT
    =====
    n_steps : int (number of steps)
    f : array containing the (initial) shape of the function to advect
    D : float (diffusion coefficient)
    h : float (cell size)
    dt : float (time step)
    int_type : str (integration type, either "FD" or "OS")

    OUTPUT
    ======
    result : 2D array containing the final result of the diffusion (the second
             dimension corresponds to the different times)
    """

    fs = f.shape[0]
    result = np.zeros((fs, n_steps))
    result[:, 0] = f
    t = 0

    if int_type == "FD":
        for i in range(1, n_steps):
            t += dt
            temp_res = diffusion_1D_FD(f, D, dt, t, h)
            result[:, i] = temp_res
            f = temp_res
    elif int_type == "OS":
        slope_type = args[0]
        for i in range(1, n_steps):
            t += dt
            temp_res1 = diffusion_1D_FD(f, D, dt/2, t, h)
            temp_res2 = advection_1D_MUSCL(temp_res1, V0, dt, t, h, slope_type)
            temp_res = diffusion_1D_FD(temp_res2, D, dt/2, t, h)
            result[:, i] = temp_res
            f = temp_res
    else:
        raise Exception("Unknown integration type '" + int_type + "'! Use either 'FD' or 'OS'.")

    return result


def step_function(x):
    """
    Step function for the initial shape

    INPUT
    =====
    x : array

    OUTPUT
    ======
    f : array
    """
    f = np.zeros(x.shape)
    f[x <= 0.4] = 1
    f[(x > 0.4) & (x <= 0.6)] = 2
    return f


def gaussian(x, sigma=0.1):
    """
    Gaussian function for the initial shape

    INPUT
    =====
    x : array

    OUTPUT
    ======
    f : array
    """

    f = 1 + np.exp(-(x - 0.5)**2/(sigma**2))

    return f


def trigonometric(x, offset):
    """
    Trigonometric function for the initial shape

    INPUT
    =====
    x : array
    offset : float

    OUTPUT
    ======
    f : array
    """

    f = (1/8)*np.sin(10*np.pi*(x - offset))/((x - offset) + 0.001)
    f[x < offset] = 0
    return f


def animate_results(F_task, fps, task, xmin, xmax, *args):
    """
    Animation and plotting of certain tasks

    INPUT
    =====
    F_task : 3D array containing the results
    fps : int
    task : int (1, 2 or 3)
    args : contains additional arguments, namely args = [D, slope_limiters]

    OUTPUT
    ======
    ani : animation reference
    """


    if task == 1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.set_window_title("AST246: Hydro project - First Task (Advection equation)")
        line1, = ax.plot([], [], 'k--', label='Original function')
        line2, = ax.plot([], [], 'r-', label='Finite differencing (Forward Euler)')
        line3, = ax.plot([], [], 'b-', label='Finite volume (MUSCL) with minmod')
        #        line4, = ax.plot([], [], 'g-', label='Finite volume (MUSCL) with superbee')
        #        lines = [line1, line2, line3, line4]
        lines = [line1, line2, line3]
    #        for slope_type in args[2]:
    #            label_str = "Finite volume (MUSCL) with "+slope_type
    #            line, = ax.plot([], [], linestyle='-.', label=label_str)
    #            lines.append(line)
    elif task == 2:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.set_window_title("AST246: Hydro project - Second Task (Diffusion equation)")
        line1, = ax.plot([], [], 'k--', label='Original function')
        lines = [line1]
        for val in args[0]:
            label_str = 'FD (RK2) with D = ' + str(round(val, 5))
            line, = ax.plot([], [], linestyle='-', label=label_str)
            lines.append(line)
    elif task == 3:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.set_window_title("AST246: Hydro project - Third Task (Advection-Diffusion equation)")
        line1, = ax.plot([], [], 'k--', label='Original function')
        lines = [line1]
        for (d, dt) in zip(args[0], args[1]):
            for slope_type in args[2]:
                label_str = 'OS with D = ' + str(round(d, 4)) + ', dt = ' + str(round(dt, 4)) + ' (' + slope_type + ')'
                if slope_type == 'minmod':
                    line, = ax.plot([], [], linestyle='-', label=label_str)
                else:
                    line, = ax.plot([], [], linestyle='--', label=label_str)
                lines.append(line)
    elif task == 4:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.set_window_title("AST246: Hydro project - Second Task (Diffusion equation)")
        line1, = ax.plot([], [], 'k--', label='Original function')
        line2, = ax.plot([], [], 'r-', label='Advection FD (Forward Euler)')
        line3, = ax.plot([], [], 'b-', label='Diffusion FD (RK2)')
        lines = [line1, line2, line3]
    else:
        raise Exception("Unknown task number! Choose either 1, 2 or 3.")

    miny = np.amin(F_task)
    maxy = np.amax(F_task)

    ax.set_ylim([miny - abs(maxy - miny)/2, 1.2*maxy])
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    timer = ax.set_title(f"Step: 0/{n_steps}")
    ax.legend(loc='best')

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(i):
        time_str = f"Step: {i}/{n_steps}"

        for lnum, line in enumerate(lines):
            line.set_data(x, F_task[:, i, lnum])

        timer.set_text(time_str)
        return tuple(lines)

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=1000/fps)

    return ani


if __name__ == '__main__':

    global x
    global V0
    global h
    global n_steps
    global f_ini

    Nx = 500  # number of points / cells. Must be integer multiple of Nx_start.
    Nx_start = 100
    step = Nx//Nx_start  # Step size for plot, such that animations stay the same speed, regardless of Nx.
    xmin, xmax = 0, 1
    x, h = np.linspace(xmin, xmax, Nx, retstep=True)

    V0 = 1  # advection velocity
    cfl = h/V0  # cfl timestep condition
    dt_advec = 0.5*cfl  # time step for the integration

    n_steps = step*500  # number of integration time steps

    # Defining the initial shape
    f_ini = step_function(x)
    # f_ini = gaussian(x)
    # f_ini = trigonometric(x, 0.1)

    # Choosing  the slope limiters for comparison
    sl = ["minmod", "superbee"]

    # Just for the plotting section ...
    f_ini_plt = np.tile(f_ini, [n_steps, 1]).T

    # Defining the number of fps for the animation at the end
    fps = 30

    """
    Task 1: Step 0 -> solving the 1D first order advection equation df/dt = -V0 df/dx
                      using first order finite differencing scheme
    """

    # Do the time integration with the finite differencing
    start = time.perf_counter()
    advection_FD = advection_1D_integration(n_steps, f_ini, V0, h, dt_advec, "FD")
    duration = time.perf_counter() - start
    print(f'1D advection finite differencing: {duration}')

    """
    Task 1: Step 1 -> solving the 1D first order advection equation df/dt = -V0 df/dx 
                      using second order finite volume scheme (MUSCL)
    """

    # Do the time integration with the finite volume scheme
    advection_MUSCL1 = advection_1D_integration(n_steps, f_ini, V0, h, dt_advec, "MUSCL", sl[0])
    # advection_MUSCL2 = advection_1D_integration(n_steps, f_ini, V0, h, dt_advec, "MUSCL", sl[1])
    #    advection_MUSCL3 = advection_1D_integration(n_steps, f_ini, V0, h, dt_advec, "MUSCL", sl[2])

    # Combine the different solutions into one array
    F1 = np.zeros((Nx, n_steps//step, 4))
    F1[:, :, 0] = f_ini_plt[:, ::step]
    F1[:, :, 1] = advection_FD[:, ::step]
    F1[:, :, 2] = advection_MUSCL1[:, ::step]
    # F1[:, :, 3] = advection_MUSCL2
    #    F1[:,:,4] = advection_MUSCL3

    """
    Task 2: solving the 1D second order diffusion equation df/dt = D d^2 f / dx^2
            using the finite difference method
    """

    # D = np.array([1e-3, 5e-2, 5e-1, 1])
    # dt_diff = np.zeros(D.shape[0])
    # dt_diff[:] = 0.5*h**2/(2*np.amax(D))
    #
    # F2 = np.zeros((Nx, n_steps, 1 + D.shape[0]))
    # F2[:, :, 0] = f_ini_plt
    # for i in range(D.shape[0]):
    #     F2[:, :, i + 1] = diffusion_1D_integration(n_steps, f_ini, D[i], h, dt_diff[i], "FD")

    """
    Task 3: solving the 1D second order advection-diffusion equation
            df/dt = -V0 df/dx + D d^2 f / fx^2
            using the operator splitting technique
    """

    # D_advec_diff = [0.01]
    # dt_advec_diff = [0.5*h**2/(2*D_advec_diff[0])]
    # #    dt_advec_diff = [dt_advec]
    #
    # #    print(n_steps*dt_advec_diff[0], dt_advec_diff[0])
    # #    print(n_steps*dt_advec_diff2[0], dt_advec_diff2[0])
    #
    # F3 = np.zeros((Nx, n_steps, 3))
    #
    # F3[:, :, 0] = f_ini_plt
    #
    # j = 1
    # for (d, dt) in zip(D_advec_diff, dt_advec_diff):
    #     for slope_type in sl:
    #         F3[:, :, j] = diffusion_1D_integration(n_steps, f_ini, d, h, dt, "OS", slope_type)
    #         j += 1
    #
    # #    F3[:,:,1] = diffusion_1D_integration(n_steps, f_ini, D_advec_diff[0], h, dt_advec_diff[0], "OS", sl[0])
    # #    F3[:,:,2] = diffusion_1D_integration(n_steps, f_ini, D_advec_diff[0], h, dt_advec_diff[0], "OS", sl[1])
    # #    F3[:,:,3] = diffusion_1D_integration(n_steps, f_ini, D_advec_diff[1], h, dt_advec_diff[1], "OS", sl[0])
    # #    F3[:,:,4] = diffusion_1D_integration(n_steps, f_ini, D_advec_diff[1], h, dt_advec_diff[1], "OS", sl[1])

    """
    Calculate the equivalent diffusion coefficient D for the advection equation
    
    """

    #    F4 = np.zeros((Nx, n_steps, 3))
    #    D_equivalent = np.array([1/2 * V0 * h])
    #    dt_equivalent = np.array([dt_advec])
    #    F4[:,:,0] = f_ini_plt
    #    F4[:,:,1] = advection_1D_integration(n_steps, f_ini, V0, h, dt_advec, "FD")
    #    F4[:,:,2] = diffusion_1D_integration(n_steps, f_ini, D_equivalent, h, dt_equivalent, "FD")

    """
    Plotting the desired results results & saving to a file
    """
    task = 1
    outer_ani = animate_results(F1, fps, task, xmin, xmax)
    plt.show()

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1000)
    # outer_ani.save(f'output/hydro_task{task}_step_Nx{Nx}_FD.mp4', writer, dpi=300)
