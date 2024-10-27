import math
import obstacles as ob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time


def plot_circle(circle):
    plt.gca().add_patch(
        plt.Circle(
            (circle.C.x, circle.C.y), circle.r - 0.2, edgecolor="black", facecolor="grey"
        )
    )


def plot_smooth_path(best_location):
    X = [point[0] for point in best_location]
    Y = [point[1] for point in best_location]

    X = [START.x] + X + [END.x]
    Y = [START.y] + Y + [END.y]

    spline = CubicSpline(np.arange(len(X)), X)
    smooth_X = spline(np.linspace(0, len(X) - 1, 100))
    spline = CubicSpline(np.arange(len(Y)), Y)
    smooth_Y = spline(np.linspace(0, len(Y) - 1, 100))

    plt.plot(smooth_X, smooth_Y, '-g', label='Smooth Path')


def plot_graph(best_location, graph):
    plt.close("all")
    plt.figure(figsize=(5, 5))
    plt.grid("on")
    plt.rc("axes", axisbelow=True)
    plt.scatter(END.x, END.y, 100, marker="o", facecolors="k", edgecolors="k")
    plt.scatter(START.x, START.y, 100, marker="o", facecolors="k", edgecolors="k")

    plot_smooth_path(best_location)

    for i in range(DIM):
        plt.scatter(
            best_location[i][0],
            best_location[i][1],
            25,
            marker="o",
            facecolors="blue",
            edgecolors="face",
        )

    plt.xlim(-1, 25)
    plt.ylim(-1, 25)
    plt.title(graph)


def random_initialization(swarm_size):
    particles_loc = np.random.rand(swarm_size, DIM, 2)*20
    particles_vel = np.random.rand(swarm_size, DIM, 2)

    particles_lowest_loss = [
        loss_function(particles_loc[i, :, :]) for i in range(0, len(particles_loc))
    ]
    particles_best_location = np.copy(particles_loc)

    global_lowest_loss = np.min(particles_lowest_loss)
    global_best_location = particles_loc[np.argmin(particles_lowest_loss)].copy()

    return (
        particles_loc,
        particles_vel,
        particles_lowest_loss,
        particles_best_location,
        global_lowest_loss,
        global_best_location,
    )


def loss_function(x):                             
    z = (x[0, 0] - START.x) ** 2 + (x[0, 1] - START.y) ** 2
    for i in range(DIM - 1):                              
        z = z + ((x[i, 0] - x[i + 1, 0]) ** 2 + (x[i, 1] - x[i + 1, 1]) ** 2)
    z = z + (x[DIM - 1, 0] - END.x) ** 2 + (x[DIM - 1, 1] - END.y) ** 2
    return math.sqrt(z)


def is_valid(circles, p):
    to_add = ob.Point(0, 0)
    point_p = ob.Point(p[0], p[1])
    for i in range(len(circles)):
        if circles[i].inside_circle(point_p):
            to_add = ob.Point(
                circles[i].how_to_exit_x(point_p.x) + to_add.x,
                circles[i].how_to_exit_y(point_p.y) + to_add.y,
            )
    return to_add


def particle_swarm_optimization(max_iterations, swarm_size, max_vel, step_size, inertia_l, inertia_u, circles):
    (
        particles_loc,
        particles_vel,
        particles_lowest_loss,
        particles_best_location,
        global_lowest_loss,
        global_best_location,
    ) = random_initialization(swarm_size)

    best_location = []
    print_iteration = 1

    for iteration_i in range(max_iterations):
        A=((iteration_i*((math.log(inertia_u))-(math.log(inertia_l))))/max_iterations)-math.log(inertia_u)
        w=math.exp(-A)
        c1=math.cos(w)
        c2=math.sin(w)
        if iteration_i % 30 == 0:
            print("%i%%" % int(iteration_i / max_iterations * 100))
            print_iteration += 1
        for particle_i in range(swarm_size):
            dim_i = 0
            while dim_i < DIM:
                error_particle_best = (particles_best_location[particle_i, dim_i]- particles_loc[particle_i, dim_i])
                error_global_best = (global_best_location[dim_i] - particles_loc[particle_i, dim_i])
                new_vel = (
                    w * min(1, (dim_i ** (0.5)) / 4)
                    * particles_vel[particle_i, dim_i]
                    + c1 * np.random.rand(1) * error_particle_best
                    + c2 * np.random.rand(1) * error_global_best
                )
                if new_vel[0] < -max_vel:
                    new_vel[0] = -max_vel
                elif new_vel[0] > max_vel:
                    new_vel[0] = max_vel
                if new_vel[1] < -max_vel:
                    new_vel[1] = -max_vel
                elif new_vel[1] > max_vel:
                    new_vel[1] = max_vel

                particles_loc[particle_i, dim_i] = (particles_loc[particle_i, dim_i] + new_vel[:] * step_size)
                particles_vel[particle_i, dim_i] = new_vel[:]

                particle_help = is_valid(circles, particles_loc[particle_i, dim_i, :])
                particles_loc[particle_i, dim_i, 0] += particle_help.x
                particles_loc[particle_i, dim_i, 1] += particle_help.y

                if abs(particle_help.x) > 0.1 or abs(particle_help.y) > 0.1:
                    dim_i -= 1
                dim_i += 1

            # for the new location, check if this is a new local or global best (if it's valid)
            particle_error = loss_function(particles_loc[particle_i, :])
            if particle_error < particles_lowest_loss[particle_i]:  # local best
                particles_lowest_loss[particle_i] = particle_error
                particles_best_location[particle_i, :] = particles_loc[particle_i, :].copy()
            if particle_error < global_lowest_loss:  # global best
                global_lowest_loss = particle_error
                global_best_location = particles_loc[particle_i, :].copy()

        best_location = global_best_location.copy()
 
    return best_location

DIM = 10
START = ob.Point(0, 0)
END = ob.Point(5, 17)


def main():
    print("Please wait:")
    best_location = []

    obstacle1 = ob.Obstacle_Circle(5.5, ob.Point(6.5, 7.5))
    obstacle2 = ob.Obstacle_Circle(3.33, ob.Point(17, 13))
    obstacle3 = ob.Obstacle_Circle(3, ob.Point(11, 18))
    obstacles = [obstacle1, obstacle2, obstacle3]

    start_time = time.time()

    best_location = particle_swarm_optimization(
        max_iterations=250,
        swarm_size=150,
        max_vel=3,
        step_size=1.2,
        inertia_l=0.7,
        inertia_u=1.3,
        circles=obstacles,
    )
    total_path_length = math.sqrt((best_location[0][0] - START.x) ** 2 + (best_location[0][1] - START.y) ** 2)
    for i in range(len(best_location) - 1):
        x1, y1 = best_location[i]
        x2, y2 = best_location[i + 1]
        total_path_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    total_path_length += math.sqrt((best_location[-1][0] - END.x) ** 2 + (best_location[-1][1] - END.y) ** 2)
    print("Total Path Length with PSO:", round(total_path_length, 3))  
    plot_graph(best_location, "PSO")
    for obstacle in obstacles:
        plot_circle(obstacle)
    plt.savefig("pso_w", dpi=300)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")

main()