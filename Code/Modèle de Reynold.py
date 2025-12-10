import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# PARAMÈTRES
# ----------------------------------

N = 200
WIDTH, HEIGHT = 60, 60
dt = 0.1

# Paramètres de Reynolds
NEIGHBOR_RADIUS = 4.0
DESIRED_SEPARATION = 0.6
MAX_SPEED = 2.0
MAX_FORCE = 0.2   # accélération max
T=1000  # nombre d'itérations
# Obstacles circulaires
obstacles = [
    (30, 30, 5.0),
    
]

# ----------------------------------
# OUTILS
# ----------------------------------

def limit(v, max_val):
    """Limite la norme d'un vecteur."""
    norm = np.linalg.norm(v)
    if norm > max_val:
        return v / norm * max_val
    return v

def distance(a, b):
    return np.linalg.norm(a - b)

# ----------------------------------
# CLASSE BOID (fidèle à Reynolds 1987)
# ----------------------------------

class Boid:
    def __init__(self):
        self.pos = np.array([np.random.uniform(0, WIDTH),
                             np.random.uniform(0, HEIGHT)], dtype=float)
        angle = np.random.uniform(0, 2*np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float)

    # --- comportement 1 : séparation ---
    def separation(self, boids):
        steer = np.zeros(2)
        count = 0
        for other in boids:
            d = distance(self.pos, other.pos)
            if 0 < d < DESIRED_SEPARATION:
                diff = self.pos - other.pos
                diff /= d  # pondération 1/d
                steer += diff
                count += 1
        if count > 0:
            steer /= count
        return limit(steer, MAX_FORCE)

    # --- comportement 2 : alignement ---
    def alignment(self, boids):
        avg_vel = np.zeros(2)
        count = 0
        for other in boids:
            d = distance(self.pos, other.pos)
            if 0 < d < NEIGHBOR_RADIUS:
                avg_vel += other.vel
                count += 1
        if count == 0:
            return np.zeros(2)
        avg_vel /= count
        desired = avg_vel / np.linalg.norm(avg_vel) * MAX_SPEED
        steer = desired - self.vel
        return limit(steer, MAX_FORCE)

    # --- comportement 3 : cohésion ---
    def cohesion(self, boids):
        center = np.zeros(2)
        count = 0
        for other in boids:
            d = distance(self.pos, other.pos)
            if 0 < d < NEIGHBOR_RADIUS:
                center += other.pos
                count += 1
        if count == 0:
            return np.zeros(2)
        center /= count
        desired = center - self.pos
        desired = desired / np.linalg.norm(desired) * MAX_SPEED
        steer = desired - self.vel
        return limit(steer, MAX_FORCE)

    # --- priorité 0 : évitement obstacle ---
    def avoid_obstacles(self):
        steer = np.zeros(2)
        for (ox, oy, R) in obstacles:
            op = np.array([ox, oy])
            d = distance(self.pos, op)
            if d < R + 2.0:
                # vecteur de fuite
                away = (self.pos - op)
                steer += away / (d**2 + 1e-6)
        return limit(steer, MAX_FORCE * 2)

    # --- mise à jour complète ---
    def update(self, boids):
        # Priorité 0 : obstacles
        force = self.avoid_obstacles()
        if np.linalg.norm(force) < 1e-6:  # sinon = obstacle impératif
            # Priorité 1 : séparation
            sep = self.separation(boids)
            if np.linalg.norm(sep) > 0:
                force = sep
            else:
                # Priorité 2 : alignement
                ali = self.alignment(boids)
                # Priorité 3 : cohésion
                coh = self.cohesion(boids)
                force = ali + coh

        force = limit(force, MAX_FORCE)

        self.vel += force
        self.vel = limit(self.vel, MAX_SPEED)

        self.pos += self.vel * dt

        # Bords toroïdaux
        self.pos[0] %= WIDTH
        self.pos[1] %= HEIGHT


# ----------------------------------
# SIMULATION
# ----------------------------------

boids = [Boid() for _ in range(N)]

positions_history = []

for t in range(T):
    for b in boids:
        b.update(boids)
    positions_history.append(np.array([b.pos.copy() for b in boids]))


# ----------------------------------
# AFFICHAGE (animation with FuncAnimation)
# ----------------------------------
from matplotlib import animation

fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')

# draw obstacles (static)
for (ox, oy, R) in obstacles:
    circle = plt.Circle((ox, oy), R, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)

# initial scatter
pos0 = positions_history[0]
# ensure shape (N,2)
scat = ax.scatter(pos0[:,0], pos0[:,1], s=20, color='tab:blue')

def update(frame):
    data = positions_history[frame]
    scat.set_offsets(data)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=50, blit=True)
plt.show()
