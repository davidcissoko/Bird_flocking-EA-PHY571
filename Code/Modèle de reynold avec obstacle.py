import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

L=20 # size

# Parameters introduced from the article / practical guidance
# R_interaction: interaction radius (alignment/cohesion), weighted by 1/r^2 in practice
# R_separation: small separation radius
# V_max: maximum speed (units/sec)
# F_max: maximum acceleration magnitude (force limiter)
R_INTERACTION = 3.0  # typical 2-4 units
R_SEPARATION = 0.8    # typical 0.5-1.0 units
V_MAX = 2.0           # typical 1-3 units/sec
F_MAX = 0.1           # typical 0.05-0.2 units/sec^2

delta_t = 1.0  # time interval (sec)

eta = 2.0  # angular/noise scale (kept for small perturbations)
# initial nominal speed (start a bit below V_MAX)
V_INIT = V_MAX * 0.6

N = 300  # nombre de particule
rho = N / L**2  # densité
T = 1000  # nombre d'itération

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


def evolution_boids(position, vel, N, obstacles=None,
                   r_interaction=R_INTERACTION, r_sep=R_SEPARATION, r_cohesion=2.0,
                   obs_infl=1.0, w_align=1.0, w_sep=1.5, w_cohesion=0.5, w_obs=2.0,
                   obs_speedup=1.0, obs_power=2.0, f_max=F_MAX, v_max=V_MAX, noise=eta):
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
    new_position = np.zeros((N, 2))
    new_vel = vel.copy()

    vx = vel[:, 0].copy()
    vy = vel[:, 1].copy()

    for i in range(N):
        align_fx = 0.0
        align_fy = 0.0
        sep_fx = 0.0
        sep_fy = 0.0
        coh_fx = 0.0
        coh_fy = 0.0
        obs_fx = 0.0
        obs_fy = 0.0
        w_align_sum = 0.0
        coh_count = 0

        xi, yi = position[i, 0], position[i, 1]

        for j in range(N):
            if i == j:
                continue
            xj, yj = position[j, 0], position[j, 1]
            dx = xj - xi
            dy = yj - yi
            # minimum image
            if dx > L/2:
                dx -= L
            elif dx < -L/2:
                dx += L
            if dy > L/2:
                dy -= L
            elif dy < -L/2:
                dy += L
            dist = np.hypot(dx, dy)
            if dist < 1e-8:
                continue

            # alignment/cohesion interaction within r_interaction
            if dist < r_interaction:
                # weight by 1/dist^2 but cap very large weights
                w = 1.0 / (dist*dist + 1e-6)
                align_fx += vx[j] * w
                align_fy += vy[j] * w
                w_align_sum += w
                # cohesion: towards neighbor positions (relative vector)
                coh_fx += dx
                coh_fy += dy
                coh_count += 1

            # separation force for near neighbors
            if dist < r_sep:
                # repulsive ~ 1/dist^2 directed away from neighbor
                sep_fx -= dx / (dist*dist + 1e-6)
                sep_fy -= dy / (dist*dist + 1e-6)

        # process alignment desired velocity
        if w_align_sum > 0:
            align_fx /= w_align_sum
            align_fy /= w_align_sum
            # desired alignment velocity scaled to v_max
            a_norm = np.hypot(align_fx, align_fy)
            if a_norm > 1e-12:
                align_fx = (align_fx / a_norm) * v_max - vx[i]
                align_fy = (align_fy / a_norm) * v_max - vy[i]
            else:
                align_fx = 0.0
                align_fy = 0.0
        else:
            align_fx = 0.0
            align_fy = 0.0

        # cohesion: move towards average neighbor direction (as acceleration)
        if coh_count > 0:
            coh_fx /= coh_count
            coh_fy /= coh_count
            coh_norm = np.hypot(coh_fx, coh_fy)
            if coh_norm > 1e-12:
                coh_fx = (coh_fx / coh_norm) * v_max - vx[i]
                coh_fy = (coh_fy / coh_norm) * v_max - vy[i]
            else:
                coh_fx = 0.0
                coh_fy = 0.0
        else:
            coh_fx = 0.0
            coh_fy = 0.0

        # obstacle avoidance
        max_obs_proximity = 0.0
        if obstacles is not None:
            for (ox, oy, orad) in obstacles:
                dxo = xi - ox
                dyo = yi - oy
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
                if disto < infl_radius and disto > 1e-8:
                    prox = (infl_radius - disto) / infl_radius
                    strength = prox ** obs_power
                    obs_fx += (dxo / disto) * strength * v_max
                    obs_fy += (dyo / disto) * strength * v_max
                    if prox > max_obs_proximity:
                        max_obs_proximity = prox

        # combine behavior forces (these are desired velocity changes)
        ax = w_align * align_fx + w_sep * sep_fx + w_cohesion * coh_fx + w_obs * obs_fx
        ay = w_align * align_fy + w_sep * sep_fy + w_cohesion * coh_fy + w_obs * obs_fy

        # small random perturbation to avoid perfect symmetry
        ax += np.random.uniform(-noise/10.0, noise/10.0)
        ay += np.random.uniform(-noise/10.0, noise/10.0)

        # limit acceleration magnitude (F_max)
        a_norm = np.hypot(ax, ay)
        if a_norm > f_max:
            ax = ax / a_norm * f_max
            ay = ay / a_norm * f_max

        # update velocity and limit speed to v_max
        new_vx = vx[i] + ax * delta_t
        new_vy = vy[i] + ay * delta_t
        speed = np.hypot(new_vx, new_vy)
        if speed > v_max:
            new_vx = new_vx / speed * v_max
            new_vy = new_vy / speed * v_max

        new_vel[i, 0] = new_vx
        new_vel[i, 1] = new_vy

        # increase speed slightly when close to obstacle (anticipatory)
        local_speed = np.hypot(new_vx, new_vy) * (1.0 + obs_speedup * max_obs_proximity)
        if local_speed > v_max:
            local_speed = v_max
        # maintain direction of new velocity but with local_speed
        dir_norm = np.hypot(new_vx, new_vy)
        if dir_norm > 1e-8:
            ux = new_vx / dir_norm
            uy = new_vy / dir_norm
            new_vel[i, 0] = ux * local_speed
            new_vel[i, 1] = uy * local_speed

        # update position with periodic boundaries
        new_position[i, 0] = (xi + new_vel[i, 0] * delta_t) % L
        new_position[i, 1] = (yi + new_vel[i, 1] * delta_t) % L

    return new_position, new_vel

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
