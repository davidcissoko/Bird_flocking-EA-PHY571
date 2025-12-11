import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
from taichi.types import vector

ti.init(arch=ti.gpu)

# ----------------------------------
# PARAMÈTRES
# ----------------------------------

N = 200
WIDTH, HEIGHT = 20, 20
dt = 0.1

# Paramètres de Reynolds
NEIGHBOR_RADIUS = 3.5
DESIRED_SEPARATION = 0.7
ALIGNMENT_RADIUS = 2.5
MAX_SPEED = 1.5
MAX_FORCE = 0.08
W_SEPARATION = 1.5
W_ALIGNMENT = 1.0
W_COHESION = 0.8
W_AVOIDANCE = 2.0  # poids évitement
Avoidance_RADIUS = 2.0  # rayon d’influence des obstacles
T = 500

# Obstacles (centres et rayons)
obstacle_count = 3
obstacles = ti.Vector.field(2, dtype=ti.f32, shape=obstacle_count)
obstacle_r = ti.field(dtype=ti.f32, shape=obstacle_count)

# ----------------------------------
# STRUCTURES TAICHI
# ----------------------------------

pos = ti.Vector.field(2, dtype=ti.f32, shape=N)  # positions
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)  # velocités

# Buffers temporaires pour les forces
sep_force = ti.Vector.field(2, dtype=ti.f32, shape=N)
ali_force = ti.Vector.field(2, dtype=ti.f32, shape=N)
coh_force = ti.Vector.field(2, dtype=ti.f32, shape=N)
# Buffer force évitement
avoid_force = ti.Vector.field(2, dtype=ti.f32, shape=N)

# ----------------------------------
# KERNELS TAICHI
# ----------------------------------

@ti.kernel
def initialize():
    """Initialisation des positions/vitesses + obstacles"""
    for i in range(N):
        pos[i][0] = ti.random(ti.f32) * WIDTH
        pos[i][1] = ti.random(ti.f32) * HEIGHT
        
        angle = ti.random(ti.f32) * 2.0 * 3.14159265
        vel[i][0] = ti.cos(angle)
        vel[i][1] = ti.sin(angle)
    # Init obstacles (exemple)
    obstacles[0] = ti.Vector([WIDTH * 0.5, HEIGHT * 0.5])
    obstacle_r[0] = 1.2
    obstacles[1] = ti.Vector([WIDTH * 0.2, HEIGHT * 0.8])
    obstacle_r[1] = 0.8
    obstacles[2] = ti.Vector([WIDTH * 0.8, HEIGHT * 0.3])
    obstacle_r[2] = 0.7

@ti.func
def limit_vector(v, max_val: ti.f32):
    # v est un ti.Vector(2, f32)
    norm = ti.sqrt(v[0] * v[0] + v[1] * v[1])
    if norm > max_val and norm > 0.0:
        v = v / norm * max_val
    return v

@ti.func
def distance(a, b) -> ti.f32:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return ti.sqrt(dx * dx + dy * dy)

@ti.kernel
def compute_separation():
    for i in range(N):
        steer_x = 0.0
        steer_y = 0.0
        count = 0
        for j in range(N):
            if i != j:
                d = distance(pos[i], pos[j])
                if 0.0 < d < DESIRED_SEPARATION:
                    steer_x += (pos[i][0] - pos[j][0]) / d
                    steer_y += (pos[i][1] - pos[j][1]) / d
                    count += 1
        if count > 0:
            invc = 1.0 / count
            steer_x *= invc
            steer_y *= invc
        sep_force[i] = limit_vector(ti.Vector([steer_x, steer_y]), MAX_FORCE)

@ti.kernel
def compute_alignment():
    for i in range(N):
        avg_x = 0.0
        avg_y = 0.0
        count = 0
        for j in range(N):
            if i != j:
                d = distance(pos[i], pos[j])
                if 0.0 < d < ALIGNMENT_RADIUS:
                    avg_x += vel[j][0]
                    avg_y += vel[j][1]
                    count += 1
        steer = ti.Vector([0.0, 0.0])
        if count > 0:
            invc = 1.0 / count
            avg_x *= invc
            avg_y *= invc
            n = ti.sqrt(avg_x * avg_x + avg_y * avg_y)
            if n > 1e-6:
                desired_x = avg_x / n * MAX_SPEED
                desired_y = avg_y / n * MAX_SPEED
                steer = ti.Vector([desired_x - vel[i][0], desired_y - vel[i][1]])
        ali_force[i] = limit_vector(steer, MAX_FORCE)

@ti.kernel
def compute_cohesion():
    for i in range(N):
        cx = 0.0
        cy = 0.0
        count = 0
        for j in range(N):
            if i != j:
                d = distance(pos[i], pos[j])
                if 0.0 < d < NEIGHBOR_RADIUS:
                    cx += pos[j][0]
                    cy += pos[j][1]
                    count += 1
        steer = ti.Vector([0.0, 0.0])
        if count > 0:
            invc = 1.0 / count
            cx *= invc
            cy *= invc
            dx = cx - pos[i][0]
            dy = cy - pos[i][1]
            n = ti.sqrt(dx * dx + dy * dy)
            if n > 1e-6:
                desired_x = dx / n * MAX_SPEED
                desired_y = dy / n * MAX_SPEED
                steer = ti.Vector([desired_x - vel[i][0], desired_y - vel[i][1]])
        coh_force[i] = limit_vector(steer, MAX_FORCE)

@ti.kernel
def compute_avoidance():
    """Force d’évitement des obstacles (répulsion ~ 1/d^2)"""
    for i in range(N):
        ax = 0.0
        ay = 0.0
        for k in range(obstacle_count):
            dx = pos[i][0] - obstacles[k][0]
            dy = pos[i][1] - obstacles[k][1]
            d = ti.sqrt(dx * dx + dy * dy)
            if d < obstacle_r[k] + Avoidance_RADIUS:
                # vecteur de fuite pondéré
                inv = 1.0 #/ (d * d)
                ax += dx * inv
                ay += dy * inv
        avoid_force[i] = limit_vector(ti.Vector([ax, ay]), MAX_FORCE * 2.0)

@ti.kernel
def update_boids():
    """Mise à jour des positions et vélocités"""
    for i in range(N):
        # Combinaison pondérée des quatre forces
        force = (
            W_SEPARATION * sep_force[i]
            + W_ALIGNMENT * ali_force[i]
            + W_COHESION * coh_force[i]
            + W_AVOIDANCE * avoid_force[i]
        )
        vel[i] += force
        vel[i] = limit_vector(vel[i], MAX_SPEED)
        pos[i] += vel[i] * dt
        # Bords toroïdaux
        if pos[i][0] < 0:
            pos[i][0] += WIDTH
        if pos[i][0] >= WIDTH:
            pos[i][0] -= WIDTH
        if pos[i][1] < 0:
            pos[i][1] += HEIGHT
        if pos[i][1] >= HEIGHT:
            pos[i][1] -= HEIGHT

# ----------------------------------
# SIMULATION
# ----------------------------------

print('Initialisation...')
initialize()

print('Simulation en cours...')
positions_history = []

for t in range(T):
    compute_separation()
    compute_alignment()
    compute_cohesion()
    compute_avoidance()  # nouveau
    update_boids()
    
    # Copier les positions pour l'animation
    positions_history.append(pos.to_numpy().copy())
    
    if t % 50 == 0:
        print(f'  Itération {t}/{T}')

print('Simulation terminée !')

# ----------------------------------
# AFFICHAGE (animation)
# ----------------------------------
from matplotlib import animation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')
ax.set_facecolor('lightgray')

# Dessin des obstacles (3 cercles)
obs_np = obstacles.to_numpy()
rad_np = obstacle_r.to_numpy()
for k in range(obstacle_count):
    circ = plt.Circle((obs_np[k, 0], obs_np[k, 1]), rad_np[k],
                      color='gray', alpha=0.4, ec='black', lw=1)
    ax.add_patch(circ)

pos0 = positions_history[0]
scat = ax.scatter(pos0[:, 0], pos0[:, 1], s=30, color='tab:blue', alpha=0.7)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(frame):
    data = positions_history[frame]
    scat.set_offsets(data)
    time_text.set_text(f'Frame: {frame}/{T}')
    return scat, time_text

ani = animation.FuncAnimation(fig, update, frames=len(positions_history), 
                             interval=50, blit=True)

plt.title('Boids Simulation (Taichi GPU)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()