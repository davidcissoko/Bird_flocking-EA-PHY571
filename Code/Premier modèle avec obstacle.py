import math as m
import numpy as np
import random
import matplotlib.pyplot as plt

L=7 #size
r=1 #interaction radius
delta_t=1 #time interval

eta=2 #dispersion
v=0.03 #distance entre deux collision
N=300 #nombre de particule
rho=N/L**2 #densité
T=100 #nombre d'itération
def evolution(position,angle,N):
    delta_teta = np.random.uniform(-eta/2, eta/2, size=N)
    new_angle=np.zeros(N)
    new_position=np.zeros((N,2)) 
    for i in range(N):
        voisins=[]
        for j in range(N):
            if i!=j:
                dist=np.sqrt((position[i,0]-position[j,0])**2+(position[i,1]-position[j,1])**2)
                if dist<r:
                    voisins.append(j)
        somme_sin=0
        somme_cos=0
        for voisin in voisins:
            somme_sin+=np.sin(angle[voisin])
            somme_cos+=np.cos(angle[voisin])
        moyenne_angle=np.arctan2(somme_sin,somme_cos)
        new_angle[i]=moyenne_angle+delta_teta[i]
        new_position[i,0]=position[i,0]+v*np.cos(new_angle[i])*delta_t
        new_position[i,1]=position[i,1]+v*np.sin(new_angle[i])*delta_t
        new_position[i,0]%=L
        new_position[i,1]%=L
    return new_position,new_angle


def evolution_boids(position, angle, N, obstacles=None, r_align=1.0, r_sep=0.3, obs_infl=1.0,
                   w_align=1.0, w_sep=1.5, w_obs=2.0, noise=eta):
    """
    Version 'boids' : combine alignment, separation and obstacle avoidance.

    - position: (N,2)
    - angle: (N,)
    - obstacles: list of tuples (x,y, radius)
    - r_align: rayon d'alignement
    - r_sep: rayon de séparation (éviter collisions entre boids)
    - obs_infl: rayon d'influence des obstacles (multiplicateur)
    - w_align, w_sep, w_obs: poids relatifs des comportements
    - noise: amplitude du bruit angulaire

    Retourne new_position, new_angle
    """
    new_angle = np.zeros(N)
    new_position = np.zeros((N,2))

    # Convert angles to unit vectors for easier vector sums
    vx = np.cos(angle)
    vy = np.sin(angle)

    for i in range(N):
        # alignment: average direction of neighbours
        align_x = 0.0
        align_y = 0.0
        sep_x = 0.0
        sep_y = 0.0
        obs_x = 0.0
        obs_y = 0.0
        count_align = 0
        for j in range(N):
            if i == j:
                continue
            # distance with periodic boundaries (minimum image)
            dx = position[j,0] - position[i,0]
            dy = position[j,1] - position[i,1]
            # periodic wrapping
            if dx > L/2:
                dx -= L
            elif dx < -L/2:
                dx += L
            if dy > L/2:
                dy -= L
            elif dy < -L/2:
                dy += L
            dist = np.hypot(dx, dy)
            if dist < r_align and dist > 1e-12:
                align_x += vx[j]
                align_y += vy[j]
                count_align += 1
            if dist < r_sep and dist > 1e-12:
                # separation: repulsive vector (stronger when closer)
                sep_x -= dx / (dist*dist + 1e-6)
                sep_y -= dy / (dist*dist + 1e-6)

        if count_align > 0:
            align_x /= count_align
            align_y /= count_align
            # normalize
            norm = np.hypot(align_x, align_y)
            if norm > 1e-12:
                align_x /= norm
                align_y /= norm
        else:
            align_x = vx[i]
            align_y = vy[i]

        # obstacle avoidance
        if obstacles is not None:
            for (ox, oy, orad) in obstacles:
                # compute vector from obstacle center to boid (consider periodic images)
                dxo = position[i,0] - ox
                dyo = position[i,1] - oy
                # consider periodic images by shifting by +/-L where closer
                if dxo > L/2:
                    dxo -= L
                elif dxo < -L/2:
                    dxo += L
                if dyo > L/2:
                    dyo -= L
                elif dyo < -L/2:
                    dyo += L
                disto = np.hypot(dxo, dyo)
                infl_radius = orad * obs_infl
                if disto < infl_radius and disto > 1e-12:
                    # repulsion vector away from obstacle center
                    strength = (infl_radius - disto) / infl_radius
                    obs_x += (dxo / disto) * strength
                    obs_y += (dyo / disto) * strength

        # combine behaviors with weights
        total_x = w_align * align_x + w_sep * sep_x + w_obs * obs_x
        total_y = w_align * align_y + w_sep * sep_y + w_obs * obs_y
        # fallback to current velocity if total is near zero
        if np.hypot(total_x, total_y) < 1e-8:
            total_x = vx[i]
            total_y = vy[i]

        # add noise
        ang = np.arctan2(total_y, total_x) + np.random.uniform(-noise/2, noise/2)
        new_angle[i] = ang
        new_position[i,0] = position[i,0] + v * np.cos(ang) * delta_t
        new_position[i,1] = position[i,1] + v * np.sin(ang) * delta_t
        # periodic boundary
        new_position[i,0] %= L
        new_position[i,1] %= L

    return new_position, new_angle

def initialisation(N):
    position=np.zeros((N,2))
    angle=np.zeros(N)
    for i in range(N):
        position[i,0]=random.uniform(0,L)
        position[i,1]=random.uniform(0,L)
        angle[i]=random.uniform(0,2*np.pi)
    return position,angle
# --- Simulation using boids model with obstacles ---
position, angle = initialisation(N)
positions_history = [position.copy()]

# Define obstacles: list of (x, y, radius)
# Example: two circular obstacles
obstacles = [
    (L*0.5, L*0.5, 0.6),
    (L*0.2, L*0.8, 0.4)
]

# Simulation loop (use evolution_boids)
for t in range(T):
    position, angle = evolution_boids(position, angle, N, obstacles=obstacles,
                                      r_align=r, r_sep=0.2, obs_infl=2.0,
                                      w_align=1.0, w_sep=2.0, w_obs=3.0, noise=eta)
    positions_history.append(position.copy())

plt.figure(figsize=(8,8))
# Nombre de pas à afficher pour les trajectoires
num_steps = min(20, len(positions_history))
hist = np.array(positions_history[-num_steps:])  # shape (num_steps, N, 2)

# Tracé des trajectoires pour chaque particule (courbe continue)
for i in range(N):
    traj = hist[:, i, :].copy()
    # Dérouler les rebonds périodiques pour avoir une trajectoire continue
    unwrapped = traj.copy()
    for k in range(1, unwrapped.shape[0]):
        dx = unwrapped[k, 0] - unwrapped[k-1, 0]
        dy = unwrapped[k, 1] - unwrapped[k-1, 1]
        if dx > L/2:
            dx -= L
        elif dx < -L/2:
            dx += L
        if dy > L/2:
            dy -= L
        elif dy < -L/2:
            dy += L
        unwrapped[k, 0] = unwrapped[k-1, 0] + dx
        unwrapped[k, 1] = unwrapped[k-1, 1] + dy
    # Recaler la trajectoire pour que le dernier point corresponde à la position finale (dans [0,L])
    offset = position[i] - unwrapped[-1]
    unwrapped += offset
    # Préparer les coordonnées modulo pour l'affichage
    xplot = unwrapped[:,0] % L
    yplot = unwrapped[:,1] % L
    # Construire des segments en insérant des NaN là où il y a un grand saut dû au recadrage
    new_x = [xplot[0]]
    new_y = [yplot[0]]
    for k in range(1, len(xplot)):
        dxp = xplot[k] - xplot[k-1]
        dyp = yplot[k] - yplot[k-1]
        if np.hypot(dxp, dyp) > L/4:
            new_x.append(np.nan)
            new_y.append(np.nan)
        new_x.append(xplot[k])
        new_y.append(yplot[k])
    plt.plot(new_x, new_y, color='b', linewidth=0.5, alpha=0.6)

# Flèches plus petites pour indiquer la direction actuelle
plt.quiver(position[:,0], position[:,1], np.cos(angle), np.sin(angle), angles='xy', scale_units='xy', scale=20, width=0.003, color='r')

# Tracé des obstacles
ax = plt.gca()
for (ox, oy, orad) in obstacles:
    circle = plt.Circle((ox, oy), orad, color='k', alpha=0.3)
    ax.add_patch(circle)
    # influence radius (dashed)
    infl = plt.Circle((ox, oy), orad*2.0, color='k', alpha=0.1, fill=False, linestyle='--')
    ax.add_patch(infl)

plt.xlim(0, L)
plt.ylim(0, L)
plt.title('Boids with Obstacles - Directions and Last {} Steps Trajectories'.format(num_steps))
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()
