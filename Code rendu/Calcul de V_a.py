import os
import csv
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
from Reynold_Taichi_2D import BoidSimulation

# Initialiser Taichi avant d'utiliser les classes
ti.init(arch=ti.gpu)












def compute_Va(V):
    V_a=0
    Norm=0
    for v in V:
        V_a+=v
        Norm+=np.linalg.norm(v)
    return np.linalg.norm(V_a)/Norm


# Paramètres constants
N = 200
WIDTH = 35.0
HEIGHT = 35.0
dt = 0.1
NEIGHBOR_RADIUS = 5.0
DESIRED_SEPARATION = 1.5
ALIGNMENT_RADIUS = 4.0
MAX_SPEED = 1.5
MAX_FORCE = 0.08
T = 2000

# Configuration de base
W_SEP_BASE = 1.8
W_ALI_BASE = 1.2
W_COH_BASE = 1.0

# Créer figure avec 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ===== GRAPHIQUE 1 : Varier W_SEPARATION =====
print("\n=== Variation de W_SEPARATION ===")
W_sep_values = [0.5, 1.0, 1.5, 1.8, 2.2, 2.8,3.5, 4.0, 5.0]
for W_sep in W_sep_values:
    print(f"Simulation avec W_SEPARATION={W_sep}")
    
    sim = BoidSimulation(N, WIDTH, HEIGHT, dt, MAX_SPEED, MAX_FORCE,
                         NEIGHBOR_RADIUS, DESIRED_SEPARATION, ALIGNMENT_RADIUS,
                         W_sep, W_ALI_BASE, W_COH_BASE)
    
    V_a = []
    Temps = []
    sim.initialize()
    
    for t in range(T):
        sim.compute_separation()
        sim.compute_alignment()
        sim.compute_cohesion()
        sim.update()
        V = sim.get_speed()
        V_a.append(compute_Va(V))
        Temps.append(t*dt)
    
    axes[0].plot(Temps, V_a, linewidth=2, label=f"W_sep = {W_sep}")

axes[0].set_xlabel("Temps (s)", fontsize=11)
axes[0].set_ylabel("V_a (ordre d'alignement)", fontsize=11)
axes[0].set_title(f"Variation de W_SEPARATION (N={N})", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3)

# ===== GRAPHIQUE 2 : Varier W_ALIGNMENT =====
print("\n=== Variation de W_ALIGNMENT ===")
W_ali_values = [0.2, 0.5, 1.0, 1.2, 1.6, 2.0]
for W_ali in W_ali_values:
    print(f"Simulation avec W_ALIGNMENT={W_ali}")
    
    sim = BoidSimulation(N, WIDTH, HEIGHT, dt, MAX_SPEED, MAX_FORCE,
                         NEIGHBOR_RADIUS, DESIRED_SEPARATION, ALIGNMENT_RADIUS,
                         W_SEP_BASE, W_ali, W_COH_BASE)
    
    V_a = []
    Temps = []
    sim.initialize()
    
    for t in range(T):
        sim.compute_separation()
        sim.compute_alignment()
        sim.compute_cohesion()
        sim.update()
        V = sim.get_speed()
        V_a.append(compute_Va(V))
        Temps.append(t*dt)
    
    axes[1].plot(Temps, V_a, linewidth=2, label=f"W_ali = {W_ali}")

axes[1].set_xlabel("Temps (s)", fontsize=11)
axes[1].set_ylabel("V_a (ordre d'alignement)", fontsize=11)
axes[1].set_title(f"Variation de W_ALIGNMENT (N={N})", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3)

# ===== GRAPHIQUE 3 : Varier W_COHESION =====
print("\n=== Variation de W_COHESION ===")
W_coh_values = [0.1, 0.3, 0.7, 1.0, 1.5, 2.0]
for W_coh in W_coh_values:
    print(f"Simulation avec W_COHESION={W_coh}")
    
    sim = BoidSimulation(N, WIDTH, HEIGHT, dt, MAX_SPEED, MAX_FORCE,
                         NEIGHBOR_RADIUS, DESIRED_SEPARATION, ALIGNMENT_RADIUS,
                         W_SEP_BASE, W_ALI_BASE, W_coh)
    
    V_a = []
    Temps = []
    sim.initialize()
    
    for t in range(T):
        sim.compute_separation()
        sim.compute_alignment()
        sim.compute_cohesion()
        sim.update()
        V = sim.get_speed()
        V_a.append(compute_Va(V))
        Temps.append(t*dt)
    
    axes[2].plot(Temps, V_a, linewidth=2, label=f"W_coh = {W_coh}")

axes[2].set_xlabel("Temps (s)", fontsize=11)
axes[2].set_ylabel("V_a (ordre d'alignement)", fontsize=11)
axes[2].set_title(f"Variation de W_COHESION (N={N})", fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9, loc='best')
axes[2].grid(True, alpha=0.3)

# Finaliser
fig.suptitle("Effet des poids des forces sur l'ordre d'alignement (Modèle de Reynolds)", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Va_poids_forces.png", dpi=120, bbox_inches='tight')
print("\nGraphique sauvegardé: Va_poids_forces.png")
plt.show()