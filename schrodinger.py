from scipy import sparse
from timesteppers import StateVector, CrankNicolson
import finite
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_the_v(x,y,V):
    fig = plt.figure()
    xm, ym = np.meshgrid(x, y)
    plt.clf()
    ax = sns.heatmap(V)
    plt.xticks([])
    plt.yticks([])
    plt.show()

class V_well():
    def __init__(self, domain, x_vals, y_vals, value = 0, plot = 0):
        x, y = domain
        x0, x1 = x_vals
        y0, y1 = y_vals

        output = np.zeros((len(x), len(x)))
        for i in range (0,len(x)):
            for j in range(0, len(x)):
                if (x[i][0] <= x0) | (x[i][0] >= x1) | (y[0][j] <= y0) | (y[0][j] >= y1):
                    output[i][j] = value
        self.V = output

        if (plot):
            plot_the_v(x, y, self.V)

class V_double_well():
    def __init__(self, domain, x_vals, y_vals, value = 0, plot = 0):
        x, y = domain
        x0, x1 = x_vals
        y0, y1 = y_vals

        output = np.zeros((len(x), len(x)))
        for i in range (0,len(x)):
            for j in range(0, len(x)):
                if ((x[i][0] >= -x1) & (x[i][0] <= x1)) & ((y[0][j] >= -y1) & (y[0][j] <= y1)):
                    output[i][j] = value 
                    if ((x[i][0] >= -x0) & (x[i][0] <= x0)) & ((y[0][j] >= -y0) & (y[0][j] <= y0)):
                        output[i][j] = 0
        self.V = output

        if (plot):
            plot_the_v(x, y, self.V)
        pass

class ValueZero_Boundary_XML():
    def __init__(self, c, m, ax, l_mat, lower_endpoint_information, higher_endpoint_information):
        self.X = StateVector([c], axis = ax)
        N = len(c)

        point_left, type_left = lower_endpoint_information
        point_right, type_right = higher_endpoint_information

        if type_left.lower() == 'v':
            a = np.zeros(N)
            a[point_left] = 1
            value_left = a
        
        if type_right.lower() == 'v':
            a = np.zeros(N)
            a[point_right] = 1
            value_right = a

        M = sparse.eye(N, N)
        M = M.tocsr()

        L = (-1j/(2*m)) * sparse.csr_matrix(l_mat)
        L = L.tocsr()

        M[point_left,:] = 0
        L[point_left,:] = value_left

        M[point_right,:] = 0 
        L[point_right,:] = value_right

        L.eliminate_zeros()
        self.L = L

        M.eliminate_zeros()
        self.M = M
        pass

class No_Boundary_XML():
    def __init__(self, c, m, ax, l_mat):
        self.X = StateVector([c], axis = ax)
        N = len(c)

        self.M = sparse.eye(N, N)
        self.L = (-1j/(2*m)) * sparse.csr_matrix(l_mat)
        pass

class Potential_No_Boundary():
    def __init__(self, c, V, ax):
        self.X = StateVector([c], axis = ax)
        N = len(c)

        self.M = sparse.eye(N, N)
        self.L = sparse.csr_matrix(1j*V)
        pass

class Schrodinger_Infinite_Well:
    def __init__(self, c, m, spatial_order, domain, p):
        self.t = 0
        self.iter = 0 
        self.c = c
        self.V = p.V

        d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)

        diffx = ValueZero_Boundary_XML(self.c, m, 0, d2x.matrix, [0, 'v'], [-1, 'v'])
        diffy = ValueZero_Boundary_XML(self.c, m, 1, d2y.matrix, [0, 'v'], [-1, 'v'])

        potential_x = ValueZero_Boundary_XML(self.c, m, 0, self.V, [0, 'v'], [-1, 'v'])
        potential_y = ValueZero_Boundary_XML(self.c, m, 1, self.V, [0, 'v'], [-1, 'v'])

        self.ts_x = CrankNicolson(diffx, 0)
        self.ts_y = CrankNicolson(diffy, 1)

        self.ts_px = CrankNicolson(potential_x, 0)
        self.ts_py = CrankNicolson(potential_y, 1)
        pass

    def step(self, dt):
        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.ts_px.step(dt/2)
        self.ts_py.step(dt)
        self.ts_px.step(dt/2)

        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.t = self.t + dt
        self.iter = self.iter + 1
        pass

class Schrodinger_No_Boundary:
    def __init__(self, c, m, spatial_order, domain, p):
        self.t = 0
        self.iter = 0 
        self.c = c
        self.V = p.V

        d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)

        diffx = No_Boundary_XML(self.c, m, 0, d2x.matrix)
        diffy = No_Boundary_XML(self.c, m, 1, d2y.matrix)

        potential_x = Potential_No_Boundary(self.c, self.V, 0)
        potential_y = Potential_No_Boundary(self.c, self.V, 1)

        self.ts_x = CrankNicolson(diffx, 0)
        self.ts_y = CrankNicolson(diffy, 1)

        self.ts_px = CrankNicolson(potential_x, 0)
        self.ts_py = CrankNicolson(potential_y, 1)
        pass

    def step(self, dt):
        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.ts_px.step(dt/2)
        self.ts_py.step(dt)
        self.ts_px.step(dt/2)

        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.t = self.t + dt
        self.iter = self.iter + 1
        pass

class Schrodinger_Quantum_Oscillator():
    def __init__(self, c, m, spatial_order, domain, x, y):
        self.t = 0
        self.iter = 0 
        self.c = c

        d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)

        diffx = No_Boundary_XML(self.c, m, 0, d2x.matrix)
        diffy = No_Boundary_XML(self.c, m, 1, d2y.matrix)

        potential_x = Potential_No_Boundary(self.c, x, 0)
        potential_y = Potential_No_Boundary(self.c, y, 1)

        self.ts_x = CrankNicolson(diffx, 0)
        self.ts_y = CrankNicolson(diffy, 1)

        self.ts_px = CrankNicolson(potential_x, 0)
        self.ts_py = CrankNicolson(potential_y, 1)
        pass

    def step(self, dt):
        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.ts_px.step(dt)
        self.ts_py.step(dt)

        self.ts_x.step(dt/4)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/4)

        self.t = self.t + dt
        self.iter = self.iter + 1
        pass
    pass