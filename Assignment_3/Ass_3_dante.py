import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# We use numpy (for array related operations) and matplotlib (for plotting) 
# because they will help us a lot

def initialize_grid(max_iter_time, plate_length, boundary_value):
    
    # Initialize solution: the grid of u(k, i, j)
    u = np.empty((max_iter_time, plate_length, plate_length))

    # Initial condition everywhere inside the grid
    u_initial = 0.0

    # Boundary conditions (fixed temperature)
    u_top = boundary_value
    u_left = 0.0
    u_bottom = 0.0
    u_right = 0.0

    # Set the initial condition
    u.fill(u_initial)

    # Set the boundary conditions
    u[:, (plate_length-1):, :] = u_top
    u[:, :, :1] = u_left
    u[:, :1, 1:] = u_bottom
    u[:, :, (plate_length-1):] = u_right


    print("\nInitial 2-D grid in spatial dimension for time snapshot t=0 is listed below\n")
    print(u[0,:,:])
    return u

    
#Initialize plate length and max time iterations

plate_length = 50
max_iter_time = 500
boundary_value = 100

initial_grid = initialize_grid(max_iter_time, plate_length, boundary_value)

alpha = 2.0
delta_x = 1

# Calculated params (\Delta t should obey the FTCS condition for stability)
delta_t = (delta_x ** 2)/(4 * alpha)
print("\nUsing a timestep size of Delta t = ", delta_t)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Calculate u iteratively on the grid based on the equation derived above

def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        print("iter:",k)
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]
  
    return u

def plotheatmap(u_k, k):
  # Clear the current plot figure
    plt.clf()
    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
  
    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()
    return plt

# Calculate final grid 
final_grid = calculate(initial_grid)

# Plot the animation for the solution in time steps

def animate(k):
    plotheatmap(final_grid[k], k)


anim = animation.FuncAnimation(plt.figure(), animate, interval=1,frames=max_iter_time, repeat=False)
anim.save("Assignment_3/heat_equation_solution.gif")


#Initialize plate length and max time iterations

plate_length = 50
max_iter_time = 50
boundary_value = 100

initial_unstable_grid = initialize_grid(max_iter_time, plate_length, boundary_value)

alpha = 2.0
delta_x = 1

# Calculated params (\Delta t violates the FTCS condition for stability)

delta_t = 1.5 * (delta_x ** 2)/(4 * alpha)
print("\nUsing a timestep size of Delta t = ", delta_t)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Calculate final grid
final_unstable_grid = calculate(initial_unstable_grid)

# Plot the animation for the solution in time steps

def animate2(k):
    plotheatmap(final_unstable_grid[k], k)


anim = animation.FuncAnimation(plt.figure(), animate2, interval=1,frames=max_iter_time, repeat=False)
anim.save("Assignment_3/heat_equation_solution_unstable.gif")