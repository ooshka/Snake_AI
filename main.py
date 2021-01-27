#####################################################################################################################
## Title: Evolutionary Neural Network for use in Snake Game
#####################################################################################################################
## Description: main function loop for evolutionary neural net used in playing Snake.
##              Displays live gameplay in addition to generational score tracking to track neural network progression
#####################################################################################################################
## Author: Alex Wadey
## Version: 1.0
## Email: wadeyalex@gmail.com
## Status: active
## File Dependencies: neural.py, snake.py, game_logic.py, display.py
#####################################################################################################################

import neural as nw
import game_logic as gm
import display as dp
import copy
import sys


#Start of main function code
def main():

    #Neural Network Details
    ######################################################################################################

    #On first iteration read model details from file
    first = True
    #Set score defaults
    model_high_score = 0
    high_score = 0
    #Number of models per generation
    pop_size = 50
    #Number of trials allowed for each model
    trials = 5
    #Number of generations
    generations = 1000
    #Number of inputs to the neural network
    n_inputs = 6
    #Number of nearons in each hidden layer
    n_neurons = 64
    #Number of outputs from the neural network
    n_outputs = 4

    #Create new generation set
    gen = nw.Generation(n_inputs = n_inputs, n_neurons = n_neurons, n_outputs = n_outputs, population_size = pop_size)


    #Display Details
    ######################################################################################################
    
    #Generate new pygame display to show to end users
    disp = dp.Display()

    #Main Game/Neural Network loop
    ######################################################################################################

    #Iterate over the number of generations set in the neural network details
    for generation in range(generations):

        #Iterate over the number of models in each generation set by pop_size in neural network details
        for model in gen.population:

            #If this is the first time we're running through the loop read previously stored model data if available
            if first == True:
                model.Model_Read()
                first = False

            #For each model's playthrough give the model trials number of attempts
            for trial in range(trials):
                
                #Play one round of snake with the model at hand
                gm.Play_Game(model, disp, generation, n_inputs)


            #Average the model's score across the number of trials
            model.score = model.score/trials

            #If the model score exceeds the saved high score write the model to file
            if model.score > model_high_score:
                model_high_score = model.score
                model.Model_Write()

        #Sort the generation's models based on score
        temp_score = gen.score_sort()
        #If the top score of the generation is above the saved high score run the game once with the model that achieved it to show users
        if temp_score > high_score:
            high_score = temp_score
            disp.game_view = True
            #Play one round of snake with the best model from that generation
            gm.Play_Game(copy.copy(gen.population[0]), disp, generation, n_inputs)


        #Update the generation graph with the new generational scores
        disp.Gen_Update(generation, gen)


        #Create the next generation of models using the evolutionary algorithm
        gen.generation_mate()

    #Exit Pygame and quit the program
    disp.Quit()

#Check to make sure this program is running the file and is not simply being imported by another .py file
if __name__ == '__main__':
    main()







