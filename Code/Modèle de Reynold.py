import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
ti.init(arch=ti.gpu)
# ----------------------------------
# PARAMÈTRES
# ----------------------------------

N = 120
WIDTH, HEIGHT = 20, 20
dt = 0.1

# Paramètres de Reynolds
NEIGHBOR_RADIUS = 3.5
DESIRED_SEPARATION = 0.7
ALiGNMENT_RADIUS = 2.5
Avoidance_RADIUS = 2.0
MAX_SPEED = 1.5
MAX_FORCE = 0.08  # accélération max
#poids des différents comportements
W_SEPARATION = 1.5
W_ALIGNMENT = 1
W_COHESION = 0.8
W_AVOIDANCE = 2.0
T=500  # nombre d'itérations
# Obstacles circulaires
obstacles = [
    
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

    # --- comportement 1 : évitement ---
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
            if 0 < d < ALiGNMENT_RADIUS:
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
            if d < R + Avoidance_RADIUS:
                # vecteur de fuite
                away = (self.pos - op)
                steer += away
               # steer += away / (d**2 + 1e-6)
        return limit(steer, MAX_FORCE * 2)

  
    def update(self,boids):
        # Les trois comportements s'additionnent TOUJOURS
        sep = self.separation(boids)
        ali = self.alignment(boids)
        coh = self.cohesion(boids)
        # Évitement des obstacles (prioritaire)
        avoid = self.avoid_obstacles()
        force = W_SEPARATION*sep+W_ALIGNMENT*ali + W_COHESION*coh+ W_AVOIDANCE*avoid
      
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
