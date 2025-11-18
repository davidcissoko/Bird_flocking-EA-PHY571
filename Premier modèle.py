import math as m
import numpy as np
import random
import matplotlib.pyplot as plt

L=5 #size
r=1 #interaction radius
delta_t=1 #time interval

eta=0.1 #dispersion
v=0.03 #distance entre deux collision
N=500 #nombre de particule
rho=N/L**2 #densité
T=20 #nombre d'itération
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
for t in range(T):
    position,angle=evolution(position,angle,N)  


plt.figure(figsize=(8,8))
plt.quiver(position[:,0], position[:,1], np.cos(angle), np.sin(angle), angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(0, L)
plt.ylim(0, L)
plt.title('Particle Directions after Evolution')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.savefig('small noise and higher density.png')