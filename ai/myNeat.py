import neat
import sys
import os
import numpy as np
import pickle
import time
import csv
import multiprocessing


sys.path.append(r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0-new\ONCS2024")
from simulation.virtualSim import *
import secrets
MAX_FITNESS = 0



def my_random(low, high):
    return secrets.randbelow(high + 1 - low) + low


def set_random_initial_conditions():
    global collector_initial_position, earth_initial_angle
    earth_initial_angle = (my_random(0, 9999999999)/9999999999)*2*np.pi
    theta = (my_random(0, 9999999999)/9999999999)*2*np.pi
    phi = (my_random(0, 9999999999)/9999999999)*np.pi
    collector_initial_position = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]) * MOON_RADIUS
    # print(earth_initial_angle, collector_initial_position)
    
    
def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    
    global sat_initial_position, sat_initial_velocity, apoapsis, periapsis, MAX_FITNESS
    for i, (genome_id1, genome1) in enumerate(genomes):
        # print("GENOME " + str(i) + "   ID: " + str(genome_id1))
        set_random_initial_conditions()
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        
        output = net1.activate((collector_initial_position[0], collector_initial_position[1], collector_initial_position[2], 
                                earth_initial_angle, ))
        # print("GENOME OUTPUT: ", output)
        sat_initial_position = np.array([output[0], output[1], output[2]])

        
        if mymath.magnitude(sat_initial_position) < MOON_RADIUS + 100e3 or mymath.magnitude(sat_initial_position) > MOON_RADIUS + 500e3:
            genome1.fitness = -100
            # print("PREV INIT ALTITUDE TEST FAILED \n")
            continue
        
        sat_initial_velocity = np.array([output[3], output[4], output[5]])
        altitude_test, chunk_fail_num = start_simulation(True)
        
        
        with open(os.path.join(r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0-new\ONCS2024\DATA", 'training.csv')) as data_file:
            reader = csv.reader(data_file, delimiter=',')
            data = list(reader)
            data_file.seek(0)
            rows_number = sum(1 for row in reader)
            rows_number -= 1
        
        # period = 2 * np.pi * np.sqrt((((apoapsis + periapsis)/2)**3) / (GRAVITATIONAL_CONSTANT * MOON_MASS))
        
        if altitude_test:
            genome1.fitness = chunk_fail_num *100
        else:
            genome1.fitness = float(data[rows_number][24])/3600
            # print("FITNESS: " + str(genome1.fitness) + '\n\n')
        
        if genome1.fitness > MAX_FITNESS:
            MAX_FITNESS = genome1.fitness
            print('MAX_FITNESS ' + str(MAX_FITNESS) + '\n')
            
        apoapsis = 0
        periapsis = np.inf
        
        



def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-25')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))

    
    
    winner = p.run(eval_genomes, 3000000)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    # # Define the background colour 
    # # using RGB color coding. 
    # background_colour = (234, 212, 252) 
    
    # # Define the dimensions of 
    # # screen object(width,height) 
    # screen = pygame.display.set_mode((300, 300)) 
    
    # # Set the caption of the screen 
    # pygame.display.set_caption('Geeksforgeeks') 
    
    # # Fill the background colour to the screen 
    # screen.fill(background_colour) 
    
    # # Update the display using flip 
    # pygame.display.flip() 
    
    
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    run_neat(config)
    
