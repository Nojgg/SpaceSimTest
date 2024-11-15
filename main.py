import os
import shutil
import random
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Code original pris sur https://github.com/pmocz/nbody-python (Philip Mocz (2020) Princeton Univeristy)
J'ai repris et modifié ce dernier pour :
- Utiliser une version non véctorisé sur le calcul des accélérations : pour mettre mieux en évidence la seconde loi de Newton
- Retirer l'évolution de la force dans le système dans le temps (pas utile dans ma démarche)
- Créer des scénarios prédéfinies et aléatoires avec : la constante gravitationnelle G n'est plus fixée à 1, mais varie entre 1 et 5, de même pour la masse des corps et de leur nombre : ils sont générés aléatoirement
- Et enfn, sauvegarder les images de la simulation pour enfaire ensuite des vidéos qui composent les scénarios.
"""

def getAcc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))

	return a


def main(repeat=1):
    
    """ N-body simulation """

    for simuNumber in range(repeat):            
        # Simulation parameters
        N         = random.randint(2, 20)    # Number of particles
        t         = 0      # current time of the simulation
        tEnd      = 10.0   # time at which simulation ends
        dt        = 0.01   # timestep
        softening = 0.1    # softening length        
        G = random.randint(1, 5) # Newton's Gravitational Constant
        plotRealTime = True

        # Generate Initial Conditions
        np.random.seed(42)

        #mass = 20.0*np.ones((N,1))/N  #static and uniform mass
        mass = np.random.randint(1, 20, size=(N,1)) #random and unequal mass
        pos  = np.random.randn(N,3)   
        vel  = np.random.randn(N,3)

        # calculate initial gravitational accelerations
        acc = getAcc( pos, mass, G, softening )

        
        Nt = 300  # number of steps
        #Nt = int(np.ceil(tEnd/dt)) # number of timesteps

        # particle orbits for plotting trails
        
        pos_save = np.zeros((N,3,Nt+1))
        pos_save[:,:,0] = pos

        # prep figure
        fig = plt.figure(figsize=(4,5), dpi=80)
        grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        ax1 = plt.subplot(grid[0:3,0])

        image_files = []
        os.makedirs('temp')
        print("nbiter : ", Nt)
            
        
        # Simulation Main Loop
        for i in range(Nt):

            print(i)
            # (1/2) kick
            vel += acc * dt/2.0 
            # drift
            pos += vel * dt

            # update accelerations
            acc = getAcc( pos, mass, G, softening )

            # (1/2) kick
            vel += acc * dt/2.0

            # update time
            t += dt
            # save energies, positions for plotting trail
            pos_save[:,:,i+1] = pos


            # plot in real time
            if plotRealTime or (i == Nt-1):
                plt.sca(ax1)            
                plt.cla()
                xx = pos_save[:,0,max(i-50,0):i+1]
                yy = pos_save[:,1,max(i-50,0):i+1]
                plt.scatter(xx,yy,s=1,color=[.7,.7,1])
                plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
                ax1.set(xlim=(-2, 2), ylim=(-2, 2))
                ax1.set_aspect('equal', 'box')                
                imgpath = 'temp/nbody' + str(i) + '.png'
                plt.savefig(imgpath, dpi=240)
                image_files.append(imgpath)                
                

        plt.xlabel('time')
        plt.ylabel('energy')                

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=30)
        clip.write_videofile('videos/simulation_' + str(N) + '_gravity_' + str(G) + 'particles_' + str(simuNumber) + '.mp4')
        shutil.rmtree('temp/') #delet all previous images, run again simulation

    return 0


  
if __name__== "__main__":
  main(100)