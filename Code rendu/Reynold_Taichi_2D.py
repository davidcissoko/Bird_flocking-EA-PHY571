import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import taichi as ti
import os
from datetime import datetime

# --- Pilotage ---
BATCH_RUN = True
CONFIGS_TO_RUN = ["cohesive", "fragmented", "chaotic"]
# SINGLE_CONFIG n’est pas utilisé quand BATCH_RUN=True

# ------------------------------------------------------------
# Sélection des paramètres par config
# ------------------------------------------------------------
def get_params(config):
    if config == "cohesive":
        # Forte cohésion + alignement → essaim compact et synchronisé
        return dict(N=200, WIDTH=35.0, HEIGHT=35.0, dt=0.1,
                    NEIGHBOR_RADIUS=5.0, DESIRED_SEPARATION=1.5, ALIGNMENT_RADIUS=4.0,
                    MAX_SPEED=1.5, MAX_FORCE=0.08,
                    W_SEPARATION=1.8, W_ALIGNMENT=1.2, W_COHESION=1.0, T=1000)
    if config == "semi_cohesive":
        # Équilibre modéré → formation de plusieurs groupes stables
        return dict(N=180, WIDTH=40.0, HEIGHT=40.0, dt=0.1,
                    NEIGHBOR_RADIUS=4.0, DESIRED_SEPARATION=1.5, ALIGNMENT_RADIUS=3.5,
                    MAX_SPEED=1.5, MAX_FORCE=0.08,
                    W_SEPARATION=1.5, W_ALIGNMENT=1.0, W_COHESION=0.8, T=4000)
    if config == "fragmented":
        # Séparation forte + cohésion limitée → petits groupes dispersés
        return dict(N=160, WIDTH=45.0, HEIGHT=45.0, dt=0.1,
                    NEIGHBOR_RADIUS=3.5, DESIRED_SEPARATION=2.0, ALIGNMENT_RADIUS=2.8,
                    MAX_SPEED=1.4, MAX_FORCE=0.08,
                    W_SEPARATION=2.0, W_ALIGNMENT=0.8, W_COHESION=0.4, T=1000)
    if config == "semi_chaotic":
        # Forces déséquilibrées → mouvement turbulent avec quelques structures
        return dict(N=200, WIDTH=50.0, HEIGHT=50.0, dt=0.1,
                    NEIGHBOR_RADIUS=3.0, DESIRED_SEPARATION=2.0, ALIGNMENT_RADIUS=2.5,
                    MAX_SPEED=1.3, MAX_FORCE=0.07,
                    W_SEPARATION=2.0, W_ALIGNMENT=0.6, W_COHESION=0.3, T=4000)
    # chaotic
    # Séparation dominante + cohésion minimale → dispersion maximale
    return dict(N=220, WIDTH=55.0, HEIGHT=55.0, dt=0.1,
                NEIGHBOR_RADIUS=2.5, DESIRED_SEPARATION=2.5, ALIGNMENT_RADIUS=2.0,
                MAX_SPEED=1.2, MAX_FORCE=0.06,
                W_SEPARATION=2.0, W_ALIGNMENT=0.4, W_COHESION=0.1, T=4000)

# ------------------------------------------------------------
# Traduction des noms de config
def config_name_fr(config):
    mapping = {
        "cohesive": "Cohésive",
        "semi_cohesive": "Semi-cohésive",
        "fragmented": "Fragmentée",
        "semi_chaotic": "Semi-chaotique",
        "chaotic": "Chaotique"
    }
    return mapping.get(config, config)

# ------------------------------------------------------------
# Classe BoidSimulation (au niveau du module pour être importable)
# ------------------------------------------------------------
@ti.data_oriented
class BoidSimulation:
    def __init__(self, n, width, height, dt, max_speed, max_force,
                 neighbor_radius, desired_separation, alignment_radius,
                 w_separation, w_alignment, w_cohesion,
                 obstacles=None, interaction_mode="quadratic"):
        self.n = n
        self.width = width
        self.height = height
        self.interaction_mode = interaction_mode
        self.is_quadratic = (interaction_mode == "quadratic")
        
        # Champs de position/vitesse/accélération
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.acc = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.sep_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.ali_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.coh_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.obs_force = ti.Vector.field(2, dtype=ti.f32, shape=n)
        
        # Paramètres comme ti.field pour utilisation dans kernels
        self.dt = ti.field(dtype=ti.f32, shape=())
        self.max_speed = ti.field(dtype=ti.f32, shape=())
        self.max_force = ti.field(dtype=ti.f32, shape=())
        self.neighbor_radius = ti.field(dtype=ti.f32, shape=())
        self.desired_separation = ti.field(dtype=ti.f32, shape=())
        self.alignment_radius = ti.field(dtype=ti.f32, shape=())
        self.w_separation = ti.field(dtype=ti.f32, shape=())
        self.w_alignment = ti.field(dtype=ti.f32, shape=())
        self.w_cohesion = ti.field(dtype=ti.f32, shape=())
        # Obstacle-specific scaling
        self.w_obstacle = ti.field(dtype=ti.f32, shape=())
        self.max_force_obs = ti.field(dtype=ti.f32, shape=())
        self.use_quadratic = ti.field(dtype=ti.i32, shape=())
        self.num_obs_field = ti.field(dtype=ti.i32, shape=())
        
        # Initialiser les paramètres
        self.dt[None] = dt
        self.max_speed[None] = max_speed
        self.max_force[None] = max_force
        self.neighbor_radius[None] = neighbor_radius
        self.desired_separation[None] = desired_separation
        self.alignment_radius[None] = alignment_radius
        self.w_separation[None] = w_separation
        self.w_alignment[None] = w_alignment
        self.w_cohesion[None] = w_cohesion
        # Stronger obstacle push by default
        self.w_obstacle[None] = 3.0
        self.max_force_obs[None] = max_force * 3.0
        self.use_quadratic[None] = 1 if self.is_quadratic else 0
        
        # Obstacles (optionnels)
        self.obstacles = obstacles if obstacles else []
        self.num_obs = len(self.obstacles)
        self.num_obs_field[None] = self.num_obs
        if self.num_obs > 0:
            self.obs_x = ti.field(ti.f32, self.num_obs)
            self.obs_y = ti.field(ti.f32, self.num_obs)
            self.obs_rad = ti.field(ti.f32, self.num_obs)
            self.obs_influence = ti.field(ti.f32, self.num_obs)
            for i, obs in enumerate(self.obstacles):
                self.obs_x[i] = obs["x"]
                self.obs_y[i] = obs["y"]
                self.obs_rad[i] = obs["radius"]
                self.obs_influence[i] = obs.get("influence_radius", 5.0)
        else:
            # Créer des fields vides si pas d'obstacles
            self.obs_x = ti.field(ti.f32, 1)
            self.obs_y = ti.field(ti.f32, 1)
            self.obs_rad = ti.field(ti.f32, 1)
            self.obs_influence = ti.field(ti.f32, 1)

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
            self.acc[i] = ti.Vector([0.0, 0.0])
    def compute_obstacles(self):
        if self.num_obs > 0:
            self._obs_kernel()

    @ti.kernel
    def _obs_kernel(self):
        for i in range(self.n):
            force = ti.Vector([0.0, 0.0])
            for k in range(self.num_obs_field[None]):
                dx = self.pos[i][0] - self.obs_x[k]
                dy = self.pos[i][1] - self.obs_y[k]
                d = ti.sqrt(dx*dx + dy*dy)
                
                if d < self.obs_influence[k]:
                    force_mag = 0.0
                    if self.use_quadratic[None] == 1:
                        # Mode quadratic: 1/r²
                        force_mag = self.max_force[None] / (d - self.obs_rad[k])**2
                    else:
                        # Mode constant
                        force_mag = self.max_force[None]
                    
                    force += ti.Vector([dx / d, dy / d]) * force_mag
            
            self.obs_force[i] = self._limit_vec(force, self.max_force[None])


    def compute_separation(self): self._sep_kernel()
    @ti.kernel
    def _sep_kernel(self):
        for i in range(self.n):
            steer = ti.Vector([0.0, 0.0]); c = 0
            for j in range(self.n):
                if i != j:
                    dvec = self.pos[i] - self.pos[j]
                    d = dvec.norm()
                    if 0.0 < d < self.desired_separation[None]:
                        # Interaction quadratique: 1/r²
                        steer += dvec / (d**3); c += 1
            if c > 0: steer *= 1.0 / c
            self.sep_force[i] = self._limit_vec(steer, self.max_force[None])

    def compute_alignment(self): self._ali_kernel()
    @ti.kernel
    def _ali_kernel(self):
        for i in range(self.n):
            avg = ti.Vector([0.0, 0.0]); c = 0
            for j in range(self.n):
                if i != j:
                    d = (self.pos[i] - self.pos[j]).norm()
                    if 0.0 < d < self.alignment_radius[None]:
                        avg += self.vel[j]; c += 1
            steer = ti.Vector([0.0, 0.0])
            if c > 0:
                avg *= 1.0 / c; n = avg.norm()
                if n > 1e-6:
                    desired = avg / n * self.max_speed[None]
                    steer = desired - self.vel[i]
            self.ali_force[i] = self._limit_vec(steer, self.max_force[None])

    def compute_cohesion(self): self._coh_kernel()
    @ti.kernel
    def _coh_kernel(self):
        for i in range(self.n):
            center = ti.Vector([0.0, 0.0]); c = 0
            for j in range(self.n):
                if i != j:
                    d = (self.pos[i] - self.pos[j]).norm()
                    if 0.0 < d < self.neighbor_radius[None]:
                        center += self.pos[j]; c += 1
            steer = ti.Vector([0.0, 0.0])
            if c > 0:
                center *= 1.0 / c
                desired = center - self.pos[i]
                n = desired.norm()
                if n > 1e-6:
                    desired = desired / n * self.max_speed[None]
                    steer = desired - self.vel[i]
            self.coh_force[i] = self._limit_vec(steer, self.max_force[None])

    

    def update(self): self._update_kernel()
    @ti.kernel
    def _update_kernel(self):
        for i in range(self.n):
            force = (self.w_separation[None] * self.sep_force[i] + 
                     self.w_alignment[None] * self.ali_force[i] +
                     self.w_cohesion[None] * self.coh_force[i] +
                     self.obs_force[i])
            self.acc[i] = force
            self.vel[i] = self._limit_vec(self.vel[i] + force*self.dt[None], self.max_speed[None])
            self.pos[i] += self.vel[i] * self.dt[None]
            # Wrap circulaire
            cx = self.width * 0.5; cy = self.height * 0.5
            R = min(self.width, self.height) * 0.5 * 0.98
            dx = self.pos[i][0] - cx; dy = self.pos[i][1] - cy
            r = ti.sqrt(dx*dx + dy*dy)
            if r > R:
                k = (R - 1e-3) / r
                self.pos[i][0] = cx - dx * k
                self.pos[i][1] = cy - dy * k

    @ti.func
    def _limit_vec(self, v, max_val):
        n = v.norm(); res = v
        if n > max_val: res = v / n * max_val
        return res

    def get_positions(self):
        return self.pos.to_numpy().copy()
    
    def get_speed(self):
        return self.vel.to_numpy().copy()

# -------
# Fonction run_config pour les simulations
# -------
def run_config(config, show_anim=False, obstacles=None, interaction_mode="quadratic"):
    # Re-init Taichi pour chaque run
    ti.reset()
    ti.init(arch=ti.gpu)

    p = get_params(config)
    
    # Raccourcis pour accéder à p directement
    N = p["N"]
    WIDTH = p["WIDTH"]
    HEIGHT = p["HEIGHT"]
    dt = p["dt"]
    NEIGHBOR_RADIUS = p["NEIGHBOR_RADIUS"]
    DESIRED_SEPARATION = p["DESIRED_SEPARATION"]
    ALIGNMENT_RADIUS = p["ALIGNMENT_RADIUS"]
    MAX_SPEED = p["MAX_SPEED"]
    MAX_FORCE = p["MAX_FORCE"]
    W_SEPARATION = p["W_SEPARATION"]
    W_ALIGNMENT = p["W_ALIGNMENT"]
    W_COHESION = p["W_COHESION"]
    T = p["T"]

    # --- Simulation ---
    print(f"Init 2D [{config.upper()}]...")
    sim = BoidSimulation(N, WIDTH, HEIGHT, dt, MAX_SPEED, MAX_FORCE,
                         NEIGHBOR_RADIUS, DESIRED_SEPARATION, ALIGNMENT_RADIUS,
                         W_SEPARATION, W_ALIGNMENT, W_COHESION,
                         obstacles=obstacles, interaction_mode=interaction_mode)
    sim.initialize()

    plt.switch_backend('Agg')
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap_dir = os.path.join(base_dir, f"sim_snaps_2d_{interaction_mode}_{config}_{run_stamp}")
    os.makedirs(snap_dir, exist_ok=True)
    print(f"Snapshots dir: {snap_dir}")

    with open(os.path.join(snap_dir, "params.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"Boids 2D Simulation Parameters [{config.upper()}]\n"
            f"DateTime: {run_stamp}\n"
            f"N={N}\nWIDTH={WIDTH}, HEIGHT={HEIGHT}\ndt={dt}\nT={T}\n"
            f"NEIGHBOR_RADIUS={NEIGHBOR_RADIUS}\nDESIRED_SEPARATION={DESIRED_SEPARATION}\n"
            f"ALIGNMENT_RADIUS={ALIGNMENT_RADIUS}\nMAX_SPEED={MAX_SPEED}\nMAX_FORCE={MAX_FORCE}\n"
            f"W_SEPARATION={W_SEPARATION}\nW_ALIGNMENT={W_ALIGNMENT}\nW_COHESION={W_COHESION}\n"
            f"Wrap: circular boundary\n"
            f"Interaction: 1/r² (quadratic)\n"
        )
        if obstacles:
            f.write(f"Obstacles: {len(obstacles)}\n")
            for i, obs in enumerate(obstacles):
                f.write(f"  Obstacle {i+1}: x={obs['x']}, y={obs['y']}, radius={obs['radius']}, influence_radius={obs.get('influence_radius', 5.0)}\n")
            f.write(f"Obstacle avoidance mode: {interaction_mode}\n")

    positions_history = []
    arrow_len = 0.8
    print(f"Running 2D [{config}]...")
    for t in range(T):
        sim.compute_separation()
        sim.compute_alignment()
        sim.compute_cohesion()
        if obstacles:
            sim.compute_obstacles()
        sim.update()
        P = sim.get_positions()
        positions_history.append(P)

        if t % 50 == 0:
            fig_snap = plt.figure(figsize=(8, 8))
            ax_snap = fig_snap.add_subplot(111)
            ax_snap.set_xlim(0, WIDTH); ax_snap.set_ylim(0, HEIGHT)
            ax_snap.set_aspect('equal'); ax_snap.set_facecolor('white')
            R = min(WIDTH, HEIGHT) * 0.5 * 0.98
            ax_snap.add_patch(plt.Circle((WIDTH*0.5, HEIGHT*0.5), R, fill=False, color='black', lw=1.2, alpha=0.4))
            
            # Ajouter les obstacles s'il y en a
            if obstacles:
                for obs in obstacles:
                    ax_snap.add_patch(plt.Circle((obs["x"], obs["y"]), obs["radius"],
                                                  fill=True, color='red', alpha=0.6, label='Obstacle'))
                    influence_r = obs.get("influence_radius", 5.0)
                    ax_snap.add_patch(plt.Circle((obs["x"], obs["y"]), influence_r,
                                                  fill=False, color='red', linestyle='--', lw=2, alpha=0.4))
            
            speeds = np.linalg.norm(P - positions_history[max(0, t-1)], axis=1) if t > 0 else np.zeros(N)
            colors = plt.cm.inferno(np.clip(speeds / MAX_SPEED, 0.0, 1.0))
            dP = P - positions_history[max(0, t-1)] if t > 0 else np.zeros_like(P)
            # Calcul des angles de rotation pour orienter les triangles
            angles = np.arctan2(dP[:, 1], dP[:, 0]) * 180 / np.pi
            # Dessiner chaque boid comme un triangle orienté
            for i in range(N):
                ax_snap.scatter(P[i, 0], P[i, 1], marker=(3, 0, angles[i]-90), 
                               s=80, c=[colors[i]], edgecolors='none', alpha=0.85)
            
            obs_txt = f" — Obstacles {interaction_mode}" if obstacles else f" — {interaction_mode}"
            ax_snap.set_title(f"Iter {t} — Boids 2D [{config_name_fr(config)}]{obs_txt}", fontsize=13, fontweight='bold')
            ax_snap.grid(True, alpha=0.2)
            txt = (
                f"N={N}, dt={dt}, vmax={MAX_SPEED}\n"
                f"R_neighbor={NEIGHBOR_RADIUS}, R_align={ALIGNMENT_RADIUS}, d_sep={DESIRED_SEPARATION}\n"
                f"W_sep={W_SEPARATION}, W_align={W_ALIGNMENT}, W_coh={W_COHESION}"
            )
            fig_snap.text(0.02, 0.02, txt, fontsize=8, family="monospace")
            out_path = os.path.join(snap_dir, f"iter_{t:04d}.png")
            fig_snap.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close(fig_snap)
            print(f"Saved snapshot: {out_path}")
        if t % 50 == 0:
            print(f"  {t}/{T}")

    # Animation facultative
    if not show_anim:
        return

    import matplotlib
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    R = min(WIDTH, HEIGHT) * 0.5 * 0.98
    ax.add_patch(plt.Circle((WIDTH*0.5, HEIGHT*0.5), R, fill=False, color='black', lw=1.2, alpha=0.4))

    pos0 = positions_history[0]
    scatter_plot = ax.scatter(pos0[:, 0], pos0[:, 1], s=50, c='blue', marker='o', alpha=0.7)
    fig.suptitle(f"Simulation des boids par le modèle de Reynolds 2D - Configuration {config_name_fr(config)} (Avec Taichi)",
                 fontsize=11, fontweight='bold')

    def update_frame(frame):
        P = positions_history[frame]
        prev = positions_history[frame-1] if frame > 0 else P
        dP = P - prev
        speeds = np.linalg.norm(dP, axis=1)
        colors = plt.cm.inferno(np.clip(speeds / MAX_SPEED, 0.0, 1.0))
        scatter_plot.set_offsets(P)
        scatter_plot.set_color(colors)
        return scatter_plot,

    ani = animation.FuncAnimation(fig, update_frame, frames=len(positions_history), interval=50, blit=True)
    plt.show()

# ------------------------------------------------------------
if __name__ == "__main__":
    if BATCH_RUN:
        for cfg in CONFIGS_TO_RUN:
            run_config(cfg, show_anim=False)
    else:
        run_config(SINGLE_CONFIG, show_anim=True)