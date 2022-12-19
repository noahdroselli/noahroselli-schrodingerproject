from scipy import sparse, special
import finite
import numpy as np
from animate import animate_heat_map
import schrodinger
import matplotlib.pyplot as plt
import seaborn as sns

choices = [ ['infinite',   'particle'],                # 0 
            ['infinite',   'stationary', [4, 3]],      # 1 - change the values [4,3] to get different states

            ['finite',     'particle_at_center_of_V'], # 2
            ['finite',     'particle_at_origin'],      # 3

            ['doublewell', 'particle'],                # 4

            ['oscillator', 'particle'],                # 5
            ['oscillator', 'stationary', [4, 3]],      # 6 - change the values [4,3] to get different states
            ['oscillator', 'elsewhere'] ]              # 7


plotTypes = ['animation', 'timed'] # 0: animate through time, 
                                   # 1: plot final solution after specific time
endTime = 0.25 # how far out in nondimensional time

choice = choices[3] # Indicate the number for which choice you want to make
plotting = plotTypes[0] # 0: animate through time, 1: plot final solution

# Initializion
spatial_order = 8
resolution = 200
alpha = 0.25
m = 1

if choice[0] == 'infinite':
    # Infinite Potential Initialization
    endPoints = 10
    grid_x = finite.UniformNonPeriodicGrid(resolution, (0, endPoints))
    grid_y = finite.UniformNonPeriodicGrid(resolution, (0, endPoints))
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()
    xm, ym, = domain.plotting_arrays()

    c = np.zeros(domain.shape, dtype = 'complex128')

    potential = schrodinger.V_well([x, y], [0, endPoints], [0, endPoints], 1)
    diff = schrodinger.Schrodinger_Infinite_Well(c, m, spatial_order, domain, potential)
    if choice[1] == 'particle':
        # Free Particle Infinite Well - CAN CHANGE INITIAL CONDITION HERE
        r = np.sqrt((x - (endPoints/2))**2 + (y - (endPoints/2))**2)
        IC = np.exp(-r**2*16)
        schrodinger.plot_the_v(x,y,IC)
    if choice[1] == 'stationary':
        # Stationary States Infinite Well 
        nx, ny = choice[2] # Can change these for different energy states
        IC = np.sin(nx*np.pi*x/(endPoints))*np.sin(ny*np.pi*y/(endPoints))

if choice[0] == 'finite':
    endPoints = 10
    grid_x = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    grid_y = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()
    xm, ym, = domain.plotting_arrays()

    c = np.zeros(domain.shape, dtype = 'complex128')

    # Finite Well Potential Initialization
    value_of_potential_height = 10
    potential = schrodinger.V_well([x, y], [3, 7], [2, 6], value_of_potential_height, 0) # Change finite well
    diff = schrodinger.Schrodinger_No_Boundary(c, m, spatial_order, domain, potential)
    if choice[1] == 'particle_at_center_of_V':
        r = np.sqrt((x - 5)**2 + (y - 4)**2) 
        IC = np.exp(-r**2*16)
    if choice[1] == 'particle_at_origin':
        r = np.sqrt((x)**2 + (y)**2)
        IC = np.exp(-r**2*16)

if choice[0] == 'doublewell':
    endPoints = 10
    grid_x = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    grid_y = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()
    xm, ym, = domain.plotting_arrays()

    c = np.zeros(domain.shape, dtype = 'complex128')

    value_of_potential_height = 100
    potential = schrodinger.V_double_well([x, y], [1, 8], [1, 8], value_of_potential_height, 1)
    diff = schrodinger.Schrodinger_Infinite_Well(c, m, spatial_order, domain, potential)
    if choice[1] == 'particle':
        r = np.sqrt((x)**2 + (y)**2)
        IC = np.exp(-r**2*16)

if choice[0] == 'oscillator':
    # Harmonic Oscillator Potential Initialization
    endPoints = 10
    grid_x = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    grid_y = finite.UniformNonPeriodicGrid(resolution, (-endPoints, endPoints))
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()
    xm, ym, = domain.plotting_arrays()

    c = np.zeros(domain.shape, dtype = 'complex128')

    p_x = sparse.diags(np.transpose(x**2/2), [0], shape = (len(x), len(x)))
    p_y = sparse.diags(y**2/2, [0], shape = (len(x), len(x)))

    diff = schrodinger.Schrodinger_Quantum_Oscillator(c, m, spatial_order, domain, p_x, p_y)
    if choice[1] == 'particle':
        r = np.sqrt((x)**2 + (y)**2)
        IC = np.exp(-r**2*16)
    if choice[1] == 'stationary':
        n, m = choice[2]
        a = np.exp(-(x**2 + y**2)/2)
        hermite_x, hermite_y = special.hermite(n), special.hermite(m)
        IC = a * hermite_x(x) * hermite_y(y)
    if choice[1] == 'elsewhere':
        r = np.sqrt((x - (np.random.rand(1) - 0.5)*endPoints)**2 + (y + (np.random.rand(1) - 0.5)*endPoints)**2)
        IC = np.exp(-r**2*16)

c[:] = IC
dt = alpha*min(grid_y.dx, grid_x.dx)

if plotting == 'animation':
    animate_heat_map(c, diff, dt)

if plotting == 'timed':
    while diff.t < endTime:
        diff.step(dt)

    c = np.real(c * np.conj(c))
    c = c / np.linalg.norm(c)

    fig = plt.figure()
    xm, ym = np.meshgrid(x, y)
    plt.clf()
    ax = sns.heatmap(c)
    plt.xticks([], labels=[])
    plt.yticks([], labels=[])
    plt.show()
