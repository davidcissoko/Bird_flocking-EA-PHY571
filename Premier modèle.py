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

def initialisation(N):
    position=np.zeros((N,2))
    angle=np.zeros(N)
    for i in range(N):
        position[i,0]=random.uniform(0,L)
        position[i,1]=random.uniform(0,L)
        angle[i]=random.uniform(0,2*np.pi)
    return position,angle
position,angle=initialisation(N)
positions_history=[position.copy()]
for t in range(T):
    position,angle=evolution(position,angle,N)
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
plt.xlim(0, L)
plt.ylim(0, L)
plt.title('Particle Directions and Last {} Steps Trajectories'.format(num_steps))
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.savefig('cas a.png')