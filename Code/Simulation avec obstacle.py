import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

L=20 #size
r=1 #interaction radius
delta_t=1 #time interval

eta=2 #dispersion
v=0.06 #distance entre deux collision (augmentée pour accélérer la simulation)
N=300 #nombre de particule
rho=N/L**2 #densité
T=600 #nombre d'itération

# Define obstacles: list of (x, y, radius)
obstacles = [
    (L*0.5, L*0.5, 1.2),
    (L*0.2, L*0.8, 0.8),
    (L*0.8, L*0.3, 0.7)
]
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


def evolution_boids(position, angle, N, obstacles=None, r_align=1.0, r_sep=0.3, r_cohesion=2.0,
                   obs_infl=1.0, w_align=1.0, w_sep=1.5, w_cohesion=0.5, w_obs=2.0,
                   max_turn=np.pi/6, noise=eta):
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
        coh_x = 0.0
        coh_y = 0.0
        count_align = 0
        count_coh = 0
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
            if dist < r_cohesion and dist > 1e-12:
                # cohesion: vector towards neighbors' center (relative)
                coh_x += dx
                coh_y += dy
                count_coh += 1
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

        if count_coh > 0:
            coh_x /= count_coh
            coh_y /= count_coh
            coh_norm = np.hypot(coh_x, coh_y)
            if coh_norm > 1e-12:
                coh_x /= coh_norm
                coh_y /= coh_norm
        else:
            coh_x = 0.0
            coh_y = 0.0

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

        # combine behaviors with weights (include cohesion)
        total_x = w_align * align_x + w_sep * sep_x + w_cohesion * coh_x + w_obs * obs_x
        total_y = w_align * align_y + w_sep * sep_y + w_cohesion * coh_y + w_obs * obs_y

        # determine desired angle
        if np.hypot(total_x, total_y) < 1e-8:
            desired_ang = np.arctan2(vy[i], vx[i])
        else:
            desired_ang = np.arctan2(total_y, total_x)
        # add noise
        desired_ang += np.random.uniform(-noise/2, noise/2)

        # limit turning rate
        cur_ang = np.arctan2(vy[i], vx[i])
        delta = (desired_ang - cur_ang + np.pi) % (2*np.pi) - np.pi
        if delta > max_turn:
            delta = max_turn
        elif delta < -max_turn:
            delta = -max_turn
        ang = cur_ang + delta
        new_angle[i] = ang
        new_position[i,0] = position[i,0] + v * np.cos(ang) * delta_t
        new_position[i,1] = position[i,1] + v * np.sin(ang) * delta_t
        # periodic boundary
        new_position[i,0] %= L
        new_position[i,1] %= L

    return new_position, new_angle

def initialisation(N, obstacles=None, margin=0.05):
    """
    Initialise positions and angles. Avoid placing boids inside any obstacle.

    - obstacles: list of (ox,oy,orad)
    - margin: additional clearance distance from obstacle surface
    """
    position = np.zeros((N,2))
    angle = np.zeros(N)
    placed = 0
    attempts = 0
    max_attempts = N * 1000
    while placed < N and attempts < max_attempts:
        attempts += 1
        x = random.uniform(0, L)
        y = random.uniform(0, L)
        ok = True
        if obstacles is not None:
            for (ox, oy, orad) in obstacles:
                dx = x - ox
                dy = y - oy
                # minimum image for periodic domain
                if dx > L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
                if dy > L/2:
                    dy -= L
                elif dy < -L/2:
                    dy += L
                dist = np.hypot(dx, dy)
                if dist < (orad + margin):
                    ok = False
                    break
        if ok:
            position[placed,0] = x
            position[placed,1] = y
            angle[placed] = random.uniform(0, 2*np.pi)
            placed += 1
    if placed < N:
        raise RuntimeError(f"Could not place all boids without overlapping obstacles after {attempts} attempts")
    return position, angle

# --- Precompute simulation frames then animate playback for smooth rendering ---
position, angle = initialisation(N, obstacles=obstacles)

print('Precomputing frames...')
positions_hist = np.zeros((T, N, 2), dtype=float)
angles_hist = np.zeros((T, N), dtype=float)
pos = position.copy()
ang = angle.copy()
for t in range(T):
    pos, ang = evolution_boids(pos, ang, N, obstacles=obstacles,
                               r_align=r, r_sep=0.3, r_cohesion=2.0, obs_infl=2.0,
                               w_align=1.0, w_sep=2.0, w_cohesion=0.6, w_obs=3.0,
                               max_turn=np.pi/6, noise=eta)
    # hard-collision correction: if any boid ended up inside an obstacle (due to large v or dt),
    # project it back to the obstacle surface and set angle away from center.
    for (ox, oy, orad) in obstacles:
        dx = pos[:,0] - ox
        dy = pos[:,1] - oy
        # minimum image for periodic domain
        dx = (dx + L/2) % L - L/2
        dy = (dy + L/2) % L - L/2
        dist = np.hypot(dx, dy)
        inside = dist < orad
        if np.any(inside):
            idx = np.where(inside)[0]
            for i_b in idx:
                if dist[i_b] < 1e-8:
                    # rare degenerate case: push radially by random angle
                    theta = random.random() * 2*np.pi
                    nx, ny = np.cos(theta), np.sin(theta)
                else:
                    nx, ny = dx[i_b]/dist[i_b], dy[i_b]/dist[i_b]
                # place on surface + small epsilon
                pos[i_b,0] = (ox + nx*(orad + 1e-3)) % L
                pos[i_b,1] = (oy + ny*(orad + 1e-3)) % L
                # set angle to point away from obstacle
                ang[i_b] = np.arctan2(ny, nx)
    positions_hist[t] = pos
    angles_hist[t] = ang
print('Precompute done.')

# prepare figure for playback
fig, ax = plt.subplots(figsize=(8,8))
ax.set_facecolor('whitesmoke')
ax.set_aspect('equal', 'box')

# draw obstacles (static)
for (ox, oy, orad) in obstacles:
    circle = plt.Circle((ox, oy), orad, color='#444444', alpha=0.35)
    ax.add_patch(circle)
    infl = plt.Circle((ox, oy), orad*2.0, color='#444444', alpha=0.12, fill=False, linestyle='--')
    ax.add_patch(infl)

# initial quiver from first frame
pos0 = positions_hist[0]
ang0 = angles_hist[0]
quiv = ax.quiver(pos0[:,0], pos0[:,1], np.cos(ang0), np.sin(ang0), angles='xy', scale_units='xy', scale=5, width=0.01, color='#d62728', pivot='middle', headwidth=5, headlength=7)

# time text
time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top')

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_title('Boids simulation (playback)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update(frame):
    pos = positions_hist[frame]
    ang = angles_hist[frame]
    quiv.set_offsets(pos)
    quiv.set_UVC(np.cos(ang), np.sin(ang))
    time_text.set_text(f'step: {frame+1}/{T}')
    return [quiv, time_text]

ani = animation.FuncAnimation(fig, update, frames=T, interval=80, blit=False)
# Automatic save (mp4) after precompute+animation creation. Requires ffmpeg installed.
out_file = 'boids_playback.mp4'
try:
    print(f"Saving animation to {out_file} (this may take a moment)...")
    ani.save(out_file, fps=int(1000/80))
    print(f"Saved video: {out_file}")
except Exception as e:
    print("Could not save video automatically. Install ffmpeg or configure matplotlib writers. Error:", e)

plt.show()
