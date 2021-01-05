"""
Snake V0
2020-07-22
"""
import sys
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import copy
import numpy
import random
import Neural_V1 as nw
import Snake_Class as sn

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

#Function Definitions
######################################################################################################

#Draw a pygame rectangle (used to slightly simplify rectangle code in main function)
def draw_rect(x, y, color, size, border, top_bar):

    rect = pygame.Rect(x*size, y*size + top_bar, size-border, size-border)
    pygame.draw.rect(SCREEN, color, rect)

#Return a random array index that has not already been filled by a number
def random_insert(Array, size):
    
    x = random.randint(0,size-1)
    y = random.randint(0,size-1)
    while Array[x,y] != 0:
        x = random.randint(0,size-1)
        y = random.randint(0,size-1)
    return x,y

######################################################################################################

#Start of main function code
def main():


    #Pygame details
    ######################################################################################################
    #Set Pygame global variables
    global SCREEN, CLOCK

    #Set boolean defualts
    game = False
    program_over = False
    first = True

    #Set score defaults
    high_score = 0
    model_high_score = 0

    #Neural Network Details
    ######################################################################################################

    #Number of models per generation
    pop_size = 75
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
    
    #Overall display size
    disp_size = 700
    #Depth of the top text bar above snake and graph
    top_bar = 40
    #grid size along one axis (vertical and horizontal)
    n_boxes = 10
    #Frame rate of th
    fps = 20
    #Border between grid boxes for the snake
    border = 1
    #Gap between edge of display and axis lines
    axis_gap = 80
    #Gap between edge of axis and axis text
    axis_text_gap = 35
    #Gap between axis data labels and axis label text
    text_gap = 25
    #Size of data points on graph
    circle_size = 5
    #Size of each axis
    axis_size = disp_size - 2 * axis_gap
    #Zero point of axis
    axis_zero = (disp_size + axis_gap, top_bar + disp_size - axis_gap)

    #Initialize Pygame
    pygame.init()

    #Set initial display window size and variable
    SCREEN = pygame.display.set_mode((disp_size * 2,disp_size + top_bar))

    #Set display caption
    pygame.display.set_caption("Snake Genetic AI")

    #Initialize pygame clock to set framerate
    CLOCK = pygame.time.Clock()

    #Initial Snake Grid Setup
    SCREEN.fill(WHITE)
    box_size = int(disp_size/n_boxes)

    #Initial Graph Setup
    graph_background = pygame.Rect(disp_size, top_bar, disp_size, disp_size)
    pygame.draw.rect(SCREEN, BLACK, graph_background)

    #Create null lists to house graph data later on
    x_data = []
    y_data = []

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

            #iterate over the number of trials set in neural network details
            for trial in range(trials):

                #Reset array for update
                A = numpy.zeros([n_boxes,n_boxes])

                #Default to not game over
                game_over = False
                
                #Place head of snake
                Sx,Sy = random_insert(A, n_boxes)
                #Create new snake and pass in head coordinates
                snake = sn.Snake((Sx,Sy))

                #Set the head array value to one to indicate the snake is there
                A[Sx,Sy] = 1
                #Direction defaults to nothign
                dir = [0,0]
                #Snake is not growing unless conditions are met
                grow = False
                #Tick counter for how many iterations a snake has been alive
                tick = 0
                #Score tracker
                score = 0
                #The number of ticks that a snake can move between each food piece. Adds up if food is taken early
                food_bonus = 25
                #Set the initial number of moves available to the food bonus
                num_moves = food_bonus

                #While game is not over
                while not game_over and num_moves >= 0:

                    #Remove the addition of food so that we can grow the snake and train it in absence of a food source
                    #Check to see if Snek food is on the grid
                    Two_Test = False
                    for x in range(len(A)):
                        for y in range(len(A)):
                            if A[x,y]==2:
                                Two_Test = True

                    #If there is no food on the grid insert a piece of food at a random location not already filled by snek
                    if Two_Test!=True:
                        Ax,Ay = random_insert(A, n_boxes)
                        A[Ax,Ay]=2


                    #Create input array for neural network
                    #We actually likely don't need any inputs aside from the distance to the foods and the fill information
                    #B/C the fill info gives some knowledge about wall distance and self distance which are essentially the same thing
                    
                    X = numpy.zeros(n_inputs)

                    #Distance to food

                    #Distance in the vertical direction
                    X[0] = Ax - snake.body[0][0]

                    #Distance in the horizontal direction
                    X[1] = Ay - snake.body[0][1]

                    #Fill function to decide which direction is safest to turn

                    #left fill
                    L = copy.copy(A)
                    X[2] = snake.fill(L, (snake.body[0][0], snake.body[0][1] - 1), 0, len(snake.body))
                    #right fill
                    R = copy.copy(A)
                    X[3] = snake.fill(R, (snake.body[0][0], snake.body[0][1] + 1), 0, len(snake.body))
                    #up fill
                    U = copy.copy(A)
                    X[4] = snake.fill(U, (snake.body[0][0] - 1, snake.body[0][1]), 0, len(snake.body))
                    #down fill
                    D = copy.copy(A)
                    X[5] = snake.fill(D, (snake.body[0][0] + 1, snake.body[0][1]), 0, len(snake.body))

                    #Get prediction from the output of the model based on the current layout of the board
                    output = model.forward(X)

                    #Take the direction with the highest level of confidence
                    prediction = numpy.argmax(output)

                    #If prediction == Left
                    if prediction == 0:
                        #if going right do nothing
                        if dir == [0,1]:
                            pass
                        else:
                            dir = [0,-1]

                    #If prediction == Right
                    elif prediction == 1:
                        #if going left do nothing
                        if dir == [0,-1]:
                            pass
                        else:
                            dir = [0,1]

                    #If prediction == Up
                    elif prediction == 2:
                        #if going down do nothing
                        if dir == [1,0]:
                            pass
                        else:
                            dir = [-1,0]

                    #If prediction == Down
                    elif prediction == 3:
                        #if going up do nothing
                        if dir == [-1,0]:
                            pass
                        else:
                            dir = [1,0]


                    #Do Snake manipulation here 

                    #Check for out of bounds
                    if snake.body[0][0]+dir[0] > (n_boxes-1) or snake.body[0][0]+dir[0] < 0 or snake.body[0][1]+dir[1] > (n_boxes-1) or snake.body[0][1]+dir[1] < 0:
                        game_over = True
                    
                    else:

                        #Check for self hit
                        try:
                            ind = snake.body.index((snake.body[0][0],snake.body[0][1]),2)
                            if ind != 0:
                                game_over = True
                        except ValueError:
                            pass
                    
                    #If we haven't hit a game over state
                    if game_over != True:

                        #Place the new head of the snake in the direction of travel
                        snake.body.insert(0,(snake.body[0][0]+dir[0], snake.body[0][1]+dir[1]))

                        #If the new head intersects with food set grow to True and increase the number of moves available to the snake
                        if A[snake.body[0][0],snake.body[0][1]] == 2:
                             grow = True
                             num_moves += food_bonus

                        #If grow isn't True then take off the end piece of the snake
                        if grow != True:
                            A[snake.body[-1][0],snake.body[-1][1]]=0
                            snake.body.pop(-1)
                        #If grow was True we don't have to destroy the last piece and can continue playing with it in the game
                        else:
                            grow = False

                        #Fill the array with 1's corresponding to the snake's body
                        for block in snake.body:
                            A[block[0],block[1]]=1

                        #If we are wanting to visualize the snake running
                        if game == True:

                            #Check for game over or user quit
                            for event in pygame.event.get():

                                if event.type == pygame.QUIT:
                                    game_over = True
                                    program_over = True

                            #Refresh the game background behind the snake
                            snake_background = pygame.Rect(0,0,disp_size, disp_size + top_bar)
                            pygame.draw.rect(SCREEN, WHITE, snake_background)
                            
                            #Place generation text at the top of the screen
                            font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 24)
                            gen_text = font.render(f"Generation: {generation+1}", True, BLACK)
                            gen_text_rect = gen_text.get_rect()
                            gen_text_rect.center = (int(disp_size*0.25), int(top_bar/2))
                            SCREEN.blit(gen_text, gen_text_rect)

                            #Place score text at the top of the screen
                            score_text = font.render(f"Score: {score}", True, BLACK)
                            score_text_rect = gen_text.get_rect()
                            score_text_rect.center = (int(disp_size*0.75), int(top_bar/2))
                            SCREEN.blit(score_text, score_text_rect)

                            #Draw rectangles in accordance with the Array values for snake body, blanks, and food
                            for x in range(len(A)):
                                for y in range(len(A)):
                                    #If there is no snake
                                    if A[x,y] == 0:
                                        draw_rect(y,x,BLACK,box_size,border, top_bar)
                                    #If there is snake body
                                    if A[x,y] == 1:
                                        draw_rect(y,x,BLUE,box_size,border, top_bar)
                                    #If there is food
                                    if A[x,y] == 2:
                                        draw_rect(y,x,GREEN,box_size,border, top_bar)


                            #Set the pygame refresh rate to "fps"
                            CLOCK.tick(fps)

                            #Pygame likes to freeze up the program if you don't take its events so pumping the events prevents this
                            pygame.event.pump()

                            #Update the display
                            pygame.display.update()

                        #Increment the number of steps that the snake has been alive (used for scoring)
                        tick += 1
                        #Reduce the number of moves available to the snake (increased when it hits food)
                        num_moves -= 1
                        #Calculate snake's score
                        score = tick*(len(snake.body))

                game = False

                #Each time a trial hits a high score show that model's next run if available
                if score > high_score and trial < trials:
                    high_score = score
                    game = True

                #Each trial add to that model's score
                model.score += score
 
            #Average the model's score across the number of trials
            model.score = model.score/trials

            #If the model score exceeds the saved high score write the model to file
            if model.score > model_high_score:
                model_high_score = model.score
                model.Model_Write()

        #At the end of each generation reset the snake display so the previous high score run isn't locked to screen
        for x in range(len(A)):
            for y in range(len(A)):
                draw_rect(y,x,BLACK,box_size,border, top_bar)

        #Sort the generation's models based on score
        gen.score_sort()

        #Graph Details for tracking generation scores
        ######################################################################################################

        #Append the generation number to the x-axis data
        x_data.append(generation+1)

        #Append the top score from each generation to the y-axis data
        y_data.append(gen.population[0].score)

        #Reset the graphical display background
        graph_background = pygame.Rect(disp_size, top_bar, disp_size, disp_size)
        pygame.draw.rect(SCREEN, BLACK, graph_background)

        #Draw x and y axis
        pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + axis_gap), (disp_size + axis_gap, top_bar + disp_size - axis_gap))
        pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + disp_size - axis_gap), (disp_size*2 - axis_gap, top_bar + disp_size - axis_gap))

        #Round to the nearest 5 on both the score and generation
        x_max = (int(max(x_data)/5)+1)*5
        y_max = (int(max(y_data)/5)+1)*5

        #Set x-axis font
        axis_font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 15)
        x_axis = axis_font.render("Generation", True, WHITE)
        x_axis_rect = x_axis.get_rect()
        x_axis_rect.center = (int(axis_size/2 + axis_zero[0]), int(axis_zero[1] + text_gap + axis_text_gap))
        SCREEN.blit(x_axis, x_axis_rect)

        #Set y-axis font
        y_axis = axis_font.render("Average Score", True, WHITE)
        y_axis = pygame.transform.rotate(y_axis, 90)
        y_axis_rect = y_axis.get_rect()
        y_axis_rect.center = (int(axis_zero[0] - text_gap - axis_text_gap), int(axis_zero[1] - axis_size/2))
        SCREEN.blit(y_axis, y_axis_rect)

        #Set the spacings for the axis label points
        spacing = axis_size/10
        x_ratio = x_max/axis_size
        y_ratio = y_max/axis_size

        #Iteratively set the label points for each axis
        for i in range(11):

            #Set x-axis label points
            x_text = axis_font.render(str(int(spacing*i*x_ratio)), True, WHITE)
            x_text_rect = x_text.get_rect()
            x_text_rect.center = (int(axis_zero[0] + spacing*i), int(axis_zero[1] + text_gap))
            SCREEN.blit(x_text, x_text_rect)

            #Set y-axis label points
            y_text = axis_font.render(str(int(spacing*i*y_ratio)), True, WHITE)
            y_text_rect = y_text.get_rect()
            y_text_rect.center = (int(axis_zero[0] - text_gap), int(axis_zero[1] - spacing*i))
            SCREEN.blit(y_text, y_text_rect)

        #Draw data points for generational scores as circles on the graph
        for i in range(len(x_data)):
            pygame.draw.circle(SCREEN, RED,(int(axis_zero[0] + 1/x_ratio * x_data[i]), int(axis_zero[1] - 1/y_ratio * y_data[i])), int(circle_size))

        #Pygame likes to freeze up the program if you don't take its events so pumping the events prevents this
        pygame.event.pump()

        #Update pygame display
        pygame.display.update()

        #Create the next generation of sneks
        gen.generation_mate()

    #Quit pygame
    pygame.quit()
    #Quit the program
    quit()

#Check to make sure this program is running the file and is not simply being imported by another .py file
if __name__ == '__main__':
    main()







