import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import taichi as ti
import os
from datetime import datetime

ti.init(arch=ti.gpu)

# --- Pilotage ---
BATCH_RUN = True  # True pour exécuter toutes les configurations sans animation
CONFIGS_TO_RUN = ["fragmented"]
SINGLE_CONFIG = "fragmented"

# --- Obstacles ---
OBSTACLES = [
    {"x": 20.0, "y": 20.0, "radius": 1.5, "influence_radius": 6.0}
]
OBSTACLE_AVOIDANCE_MODE = "constant"  # "inverse_square" ou "constant"

# --- Traduction ---
def config_name_fr(config):
    mapping = {
        "cohesive": "Cohésive",
        "semi_cohesive": "Semi-cohésive",
        "fragmented": "Fragmentée",
        "semi_chaotic": "Semi-chaotique",
        "chaotic": "Chaotique"
    }
    return mapping.get(config, config)

# --- Paramètres par config (copiés depuis Reynold Taichi 2D.py) ---
def get_params(config):
    if config == "cohesive":
        # Forte cohésion + alignement → essaim compact et synchronisé
        return dict(N=200, WIDTH=35.0, HEIGHT=35.0, dt=0.1,
                    NEIGHBOR_RADIUS=5.0, DESIRED_SEPARATION=1.5, ALIGNMENT_RADIUS=4.0,
                    MAX_SPEED=1.5, MAX_FORCE=0.08,
                    W_SEPARATION=1.8, W_ALIGNMENT=1.2, W_COHESION=1.0, W_OBSTACLE=1.5, T=4000)
    if config == "semi_cohesive":
        # Équilibre modéré → formation de plusieurs groupes stables
        return dict(N=180, WIDTH=40.0, HEIGHT=40.0, dt=0.1,
                    NEIGHBOR_RADIUS=4.0, DESIRED_SEPARATION=1.5, ALIGNMENT_RADIUS=3.5,
                    MAX_SPEED=1.5, MAX_FORCE=0.08,
                    W_SEPARATION=1.5, W_ALIGNMENT=1.0, W_COHESION=0.8, W_OBSTACLE=1.5, T=4000)
    if config == "fragmented":
        # Séparation forte + cohésion limitée → petits groupes dispersés
        return dict(N=160, WIDTH=45.0, HEIGHT=45.0, dt=0.1,
                    NEIGHBOR_RADIUS=3.5, DESIRED_SEPARATION=2.0, ALIGNMENT_RADIUS=2.8,
                    MAX_SPEED=1.4, MAX_FORCE=0.08,
                    W_SEPARATION=2.0, W_ALIGNMENT=0.8, W_COHESION=0.4, W_OBSTACLE=2, T=2000)
    if config == "semi_chaotic":
        # Forces déséquilibrées → mouvement turbulent avec quelques structures
        return dict(N=200, WIDTH=50.0, HEIGHT=50.0, dt=0.1,
                    NEIGHBOR_RADIUS=3.0, DESIRED_SEPARATION=2.0, ALIGNMENT_RADIUS=2.5,
                    MAX_SPEED=1.3, MAX_FORCE=0.07,
                    W_SEPARATION=2.0, W_ALIGNMENT=0.6, W_COHESION=0.3, W_OBSTACLE=2.2, T=4000)
    # chaotic
    # Séparation dominante + cohésion minimale → dispersion maximale
    return dict(N=220, WIDTH=55.0, HEIGHT=55.0, dt=0.1,
                NEIGHBOR_RADIUS=2.5, DESIRED_SEPARATION=2.5, ALIGNMENT_RADIUS=2.0,
                MAX_SPEED=1.2, MAX_FORCE=0.06,
                W_SEPARATION=2.0, W_ALIGNMENT=0.4, W_COHESION=0.1, W_OBSTACLE=2.5, T=4000)

# --- Simulation ---
def run_config(config, show_anim=False, obstacles=None, obs_mode="inverse_square"):
    ti.reset()
    ti.init(arch=ti.gpu)

    p = get_params(config)
    N = p["N"]; WIDTH = p["WIDTH"]; HEIGHT = p["HEIGHT"]; dt = p["dt"]
    NEIGHBOR_RADIUS = p["NEIGHBOR_RADIUS"]; DESIRED_SEPARATION = p["DESIRED_SEPARATION"]
    ALIGNMENT_RADIUS = p["ALIGNMENT_RADIUS"]; MAX_SPEED = p["MAX_SPEED"]; MAX_FORCE = p["MAX_FORCE"]
    W_SEPARATION = p["W_SEPARATION"]; W_ALIGNMENT = p["W_ALIGNMENT"]
    W_COHESION = p["W_COHESION"]; W_OBSTACLE = p["W_OBSTACLE"]; T = p["T"]

    # Obstacles en Taichi field (avec influence_radius personnalisé)
    num_obs = len(obstacles) if obstacles else 0
    obs_x = ti.field(ti.f32, num_obs)
    obs_y = ti.field(ti.f32, num_obs)
    obs_rad = ti.field(ti.f32, num_obs)
    obs_influence = ti.field(ti.f32, num_obs)

    if obstacles:
        for i, obs in enumerate(obstacles):
            obs_x[i] = obs["x"]
            obs_y[i] = obs["y"]
            obs_rad[i] = obs["radius"]
            obs_influence[i] = obs.get("influence_radius", 5.0)  # défaut 5.0

    @ti.data_oriented
    class BoidSimulation:
        def __init__(self, n, width, height):
            self.n = n; self.width = width; self.height = height
            self.pos = ti.Vector.field(2, dtype=ti.f32, shape=n)
            self.vel = ti.Vector.field(2, dtype=ti.f32, shape=n)
            self.sep_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
            self.ali_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
            self.coh_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
            self.obs_force = ti.Vector.field(2, dtype=ti.f32, shape=n)

        def initialize(self):
            self._init_kernel()

        @ti.kernel
        def _init_kernel(self):
            center = ti.Vector([self.width * 0.5, self.height * 0.5])
            R = ti.min(self.width, self.height) * 0.5 * 0.98
            for i in range(self.n):
                r = ti.sqrt(ti.random()) * R
                theta = ti.random() * 2.0 * ti.math.pi
                offset = ti.Vector([ti.cos(theta), ti.sin(theta)]) * r
                self.pos[i] = center + offset
                ang = ti.random() * 2.0 * ti.math.pi
                self.vel[i] = ti.Vector([ti.cos(ang), ti.sin(ang)])

        def compute_separation(self): self._sep_kernel()
        @ti.kernel
        def _sep_kernel(self):
            for i in range(self.n):
                steer = ti.Vector([0.0, 0.0]); c = 0
                for j in range(self.n):
                    if i != j:
                        dvec = self.pos[i] - self.pos[j]
                        d = dvec.norm()
                        if 0.0 < d < DESIRED_SEPARATION:
                            steer += dvec / (d**3); c += 1
                if c > 0: steer *= 1.0 / c
                self.sep_force[i] = self._limit_vec(steer, MAX_FORCE)

        def compute_alignment(self): self._ali_kernel()
        @ti.kernel
        def _ali_kernel(self):
            for i in range(self.n):
                avg = ti.Vector([0.0, 0.0]); c = 0
                for j in range(self.n):
                    if i != j:
                        d = (self.pos[i] - self.pos[j]).norm()
                        if 0.0 < d < ALIGNMENT_RADIUS:
                            avg += self.vel[j]; c += 1
                steer = ti.Vector([0.0, 0.0])
                if c > 0:
                    avg *= 1.0 / c; n = avg.norm()
                    if n > 1e-6:
                        desired = avg / n * MAX_SPEED
                        steer = desired - self.vel[i]
                self.ali_force[i] = self._limit_vec(steer, MAX_FORCE)

        def compute_cohesion(self): self._coh_kernel()
        @ti.kernel
        def _coh_kernel(self):
            for i in range(self.n):
                center = ti.Vector([0.0, 0.0]); c = 0
                for j in range(self.n):
                    if i != j:
                        d = (self.pos[i] - self.pos[j]).norm()
                        if 0.0 < d < NEIGHBOR_RADIUS:
                            center += self.pos[j]; c += 1
                steer = ti.Vector([0.0, 0.0])
                if c > 0:
                    center *= 1.0 / c
                    desired = center - self.pos[i]
                    n = desired.norm()
                    if n > 1e-6:
                        desired = desired / n * MAX_SPEED
                        steer = desired - self.vel[i]
                self.coh_force[i] = self._limit_vec(steer, MAX_FORCE)

        def compute_obstacles(self): self._obs_kernel()
        @ti.kernel
        def _obs_kernel(self):
            for i in range(self.n):
                force = ti.Vector([0.0, 0.0])
                for k in range(num_obs):
                    dx = self.pos[i][0] - obs_x[k]
                    dy = self.pos[i][1] - obs_y[k]
                    d = ti.sqrt(dx*dx + dy*dy)
                    
                    if obs_mode == "inverse_square":
                        # Force = MAX_SPEED / r^2 si dans la zone d'influence
                        if d < obs_influence[k] and d > obs_rad[k]:
                            force_mag = MAX_SPEED / (d * d)
                            force += ti.Vector([dx / d, dy / d]) * force_mag
                    else:  # constant
                        if d < obs_influence[k] and d > obs_rad[k]:
                            force += ti.Vector([dx / d, dy / d]) * MAX_SPEED
                
                self.obs_force[i] = self._limit_vec(force, MAX_FORCE)

        def update(self): self._update_kernel()
        @ti.kernel
        def _update_kernel(self):
            cx = self.width * 0.5; cy = self.height * 0.5
            R = ti.min(self.width, self.height) * 0.5 * 0.98
            for i in range(self.n):
                # --- PRIORITÉ 1: Évitement obstacles (1/r^2) ---
                obs_repulsion = self.obs_force[i]               
                
                
                # --- PRIORITÉ 2: Forces flocking ---
                force = (W_SEPARATION * self.sep_force[i] +
                         W_ALIGNMENT * self.ali_force[i] +
                         W_COHESION * self.coh_force[i])
                
                # Ajoute évitement obstacle avec poids très fort
                force += obs_repulsion * W_OBSTACLE
                
                self.vel[i] = self._limit_vec(self.vel[i] + force*dt, MAX_SPEED)
                
                # --- Déplacement APRÈS évitement calculé ---
                self.pos[i] += self.vel[i] * dt
                
                # --- Post-collision: sécurité si quand même dedans ---
                for k in range(num_obs):
                    dx = self.pos[i][0] - obs_x[k]
                    dy = self.pos[i][1] - obs_y[k]
                    d = ti.sqrt(dx*dx + dy*dy)
                    if d < obs_rad[k] + 0.1:
                        if d > 1e-6:
                            self.pos[i][0] = obs_x[k] + (dx / d) * (obs_rad[k] + 0.1)
                            self.pos[i][1] = obs_y[k] + (dy / d) * (obs_rad[k] + 0.1)
                        self.vel[i] *= 0.5
                
                # --- Wrap circulaire ---
                dx = self.pos[i][0] - cx; dy = self.pos[i][1] - cy
                d = ti.sqrt(dx*dx + dy*dy)
                if d > R:
                    k = (R - 1e-3) / d
                    self.pos[i][0] = cx - dx * k
                    self.pos[i][1] = cy - dy * k

        @ti.func
        def _limit_vec(self, v, max_val):
            n = v.norm(); res = v
            if n > max_val: res = v / n * max_val
            return res

        def get_positions(self):
            return self.pos.to_numpy().copy()

    # --- Simulation ---
    print(f"Init 2D [{config.upper()}]...")
    sim = BoidSimulation(N, WIDTH, HEIGHT)
    sim.initialize()

    plt.switch_backend('Agg')
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap_dir = os.path.join(base_dir, f"sim_snaps_2d_taichi_{config}_{run_stamp}")
    os.makedirs(snap_dir, exist_ok=True)
    print(f"Snapshots dir: {snap_dir}")

    with open(os.path.join(snap_dir, "params.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"Boids 2D Simulation (Taichi GPU) [{config.upper()}]\n"
            f"DateTime: {run_stamp}\n"
            f"N={N}\nWIDTH={WIDTH}, HEIGHT={HEIGHT}\ndt={dt}\nT={T}\n"
            f"NEIGHBOR_RADIUS={NEIGHBOR_RADIUS}\nDESIRED_SEPARATION={DESIRED_SEPARATION}\n"
            f"ALIGNMENT_RADIUS={ALIGNMENT_RADIUS}\nMAX_SPEED={MAX_SPEED}\nMAX_FORCE={MAX_FORCE}\n"
            f"W_SEPARATION={W_SEPARATION}\nW_ALIGNMENT={W_ALIGNMENT}\nW_COHESION={W_COHESION}\n"
            f"W_OBSTACLE={W_OBSTACLE}\n"
            f"Wrap: circular boundary (radius={min(WIDTH, HEIGHT) * 0.5 * 0.98:.2f})\n"
            f"Obstacle avoidance mode: {obs_mode}\n"
            f"Num obstacles: {num_obs}\n"
        )
        if obstacles:
            for i, obs in enumerate(obstacles):
                f.write(f"Obstacle {i+1}: x={obs['x']}, y={obs['y']}, radius={obs['radius']}, influence_radius={obs.get('influence_radius', 5.0)}\n")

    positions_history = []
    arrow_len = 1.2

    print(f"Running 2D [{config}]...")
    for t in range(T):
        sim.compute_separation()
        sim.compute_alignment()
        sim.compute_cohesion()
        sim.compute_obstacles()
        sim.update()
        P = sim.get_positions()
        positions_history.append(P)

        if t % 50 == 0:
            fig_snap = plt.figure(figsize=(8, 8))
            ax_snap = fig_snap.add_subplot(111)
            ax_snap.set_xlim(0, WIDTH); ax_snap.set_ylim(0, HEIGHT)
            ax_snap.set_aspect('equal'); ax_snap.set_facecolor('white')

            R_boundary = min(WIDTH, HEIGHT) * 0.5 * 0.98
            ax_snap.add_patch(plt.Circle((WIDTH*0.5, HEIGHT*0.5), R_boundary, fill=False, color='black', lw=1.2, alpha=0.4))

            if obstacles:
                for obs in obstacles:
                    ax_snap.add_patch(plt.Circle((obs["x"], obs["y"]), obs["radius"],
                                                  fill=True, color='gray', alpha=0.5))
                    influence_r = obs.get("influence_radius", 5.0)
                    ax_snap.add_patch(plt.Circle((obs["x"], obs["y"]), influence_r,
                                                  fill=False, color='gray', linestyle='--', alpha=0.3))

            speeds = np.linalg.norm(P - positions_history[max(0, t-1)], axis=1) if t > 0 else np.zeros(N)
            colors = plt.cm.inferno(np.clip(speeds / MAX_SPEED, 0.0, 1.0))
            dP = P - positions_history[max(0, t-1)] if t > 0 else np.zeros_like(P)
            # Calcul des angles de rotation pour orienter les triangles
            angles = np.arctan2(dP[:, 1], dP[:, 0]) * 180 / np.pi
            # Dessiner chaque boid comme un triangle orienté
            for i in range(N):
                ax_snap.scatter(P[i, 0], P[i, 1], marker=(3, 0, angles[i]-90), 
                               s=80, c=[colors[i]], edgecolors='none', alpha=0.85)

            ax_snap.set_title(f"Iter {t} — Boids 2D [{config_name_fr(config)}] — Obstacles 1/r²", fontsize=13, fontweight='bold')
            ax_snap.grid(True, alpha=0.2)
            txt = (
                f"N={N}, dt={dt}, vmax={MAX_SPEED}\n"
                f"R_neighbor={NEIGHBOR_RADIUS}, R_align={ALIGNMENT_RADIUS}, d_sep={DESIRED_SEPARATION}\n"
                f"W_sep={W_SEPARATION}, W_align={W_ALIGNMENT}, W_coh={W_COHESION}, W_obs={W_OBSTACLE}"
            )
            fig_snap.text(0.02, 0.02, txt, fontsize=8, family="monospace")
            out_path = os.path.join(snap_dir, f"iter_{t:04d}_{config_name_fr(config)}_{run_stamp}.png")
            fig_snap.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close(fig_snap)
            print(f"Saved snapshot: {out_path}")

        if t % 50 == 0:
            print(f"  {t}/{T}")

    if not show_anim:
        return

    import matplotlib
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    R_boundary = min(WIDTH, HEIGHT) * 0.5 * 0.98
    ax.add_patch(plt.Circle((WIDTH*0.5, HEIGHT*0.5), R_boundary, fill=False, color='black', lw=1.2, alpha=0.4))

    if obstacles:
        for obs in obstacles:
            ax.add_patch(plt.Circle((obs["x"], obs["y"]), obs["radius"],
                                    fill=True, color='gray', alpha=0.5))

    pos0 = positions_history[0]
    dP0 = pos0 - positions_history[0]
    dnorm0 = np.linalg.norm(dP0, axis=1, keepdims=True)
    dirv0 = dP0 / (dnorm0 + 1e-6)
    U0 = dirv0[:, 0] * arrow_len; V0 = dirv0[:, 1] * arrow_len
    colors0 = plt.cm.inferno(np.clip(np.linalg.norm(dP0, axis=1) / MAX_SPEED, 0.0, 1.0))
    quiver_plot = ax.quiver(pos0[:, 0], pos0[:, 1], U0, V0,
                            color=colors0, angles='xy', scale_units='xy',
                            scale=1.0, width=0.018, alpha=0.9, pivot='mid')
    fig.suptitle(f"Simulation des boids par le modèle de Reynolds 2D - Configuration {config_name_fr(config)} (Avec Taichi)",
                 fontsize=11, fontweight='bold')

    def update_frame(frame):
        P = positions_history[frame]
        prev = positions_history[frame-1] if frame > 0 else P
        dP = P - prev
        dnorm = np.linalg.norm(dP, axis=1, keepdims=True)
        dirv = dP / (dnorm + 1e-6)
        U = dirv[:, 0] * arrow_len; V = dirv[:, 1] * arrow_len
        colors = plt.cm.inferno(np.clip(np.linalg.norm(dP, axis=1) / MAX_SPEED, 0.0, 1.0))
        quiver_plot.set_offsets(P)
        quiver_plot.set_UVC(U, V)
        quiver_plot.set_color(colors)
        return quiver_plot,

    ani = animation.FuncAnimation(fig, update_frame, frames=len(positions_history), interval=50, blit=True)
    plt.show()

# Main
if __name__ == "__main__":
    if BATCH_RUN:
        for cfg in CONFIGS_TO_RUN:
            run_config(cfg, show_anim=False, obstacles=OBSTACLES, obs_mode=OBSTACLE_AVOIDANCE_MODE)
    else:
        run_config(SINGLE_CONFIG, show_anim=True, obstacles=OBSTACLES, obs_mode=OBSTACLE_AVOIDANCE_MODE)