"""
 Snake V0
 2020-07-22
"""
 
import sys
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

def draw_rect(x,y,color,size,b):

    rect = pygame.Rect(x*size, y*size, size-b, size-b)
    pygame.draw.rect(SCREEN, color, rect)

def random_insert(Array, size):
    
    x = random.randint(0,size-1)
    y = random.randint(0,size-1)
    while Array[x,y] != 0:
        x = random.randint(0,size-1)
        y = random.randint(0,size-1)
    return x,y

############################ Start of main function code #############################################

global SCREEN, CLOCK

#Set the random seed so that each playthrough is identical
#random.seed(10)

game = False
program_over = False

disp_size = 600
n_boxes = 10
fps = 8
border = 1

#Neural Network Details
#######################################################################################

pop_size = 100
trials = 5
generations = 100
n_inputs = 6

gen = nw.Generation(n_inputs = n_inputs, n_neurons = 512, n_outputs = 4, population_size = pop_size)

#######################################################################################

for generation in range(generations):

    if generation == generations-1:
        pass
        #game = True

    if game == True:

        pygame.init()

        SCREEN = pygame.display.set_mode((disp_size,disp_size))

        pygame.display.set_caption("Snake V0")

        CLOCK = pygame.time.Clock()

        #Initial Grid Setup
        SCREEN.fill(WHITE)
        box_size = int(disp_size/n_boxes)

    for model in gen.population:


        if program_over == False:

            for trial in range(trials):

                #Reset array for update
                A = numpy.zeros([n_boxes,n_boxes])

                game_over = False
                #Place head of snake
                Sx,Sy = random_insert(A, n_boxes)
                snake = sn.Snake((Sx,Sy))
                A[Sx,Sy] = 1
                dir = [0,0]
                grow = False
                tick = 0
                food_bonus = 20
                num_moves = food_bonus


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
                    B = copy.copy(A)
                    X[2] = snake.fill(B,(snake.body[0][0], snake.body[0][1] - 1), 0, len(snake.body))

                    #right fill
                    B = copy.copy(A)
                    X[3] = snake.fill(B,(snake.body[0][0], snake.body[0][1] + 1), 0, len(snake.body))

                    #up fill
                    B = copy.copy(A)
                    X[4] = snake.fill(B,(snake.body[0][0] - 1, snake.body[0][1]), 0, len(snake.body))

                    #down fill
                    B = copy.copy(A)
                    X[5] = snake.fill(B,(snake.body[0][0] + 1, snake.body[0][1]), 0, len(snake.body))

                    #print(X)

                    #Get prediction from the output of the model being iterated through
                    output = model.forward(X)
                    prediction = numpy.argmax(output)

                    if game == True:

                        #Check for game over or user quit
                        for event in pygame.event.get():

                            if event.type == pygame.QUIT:
                                game_over = True
                                program_over = True

                    if prediction == 0:
                        if dir == [0,1]:
                            pass
                        else:
                            dir = [0,-1]

                    elif prediction == 1:
                        if dir == [0,-1]:
                            pass
                        else:
                            dir = [0,1]

                    elif prediction == 2:
                        if dir == [1,0]:
                            pass
                        else:
                            dir = [-1,0]

                    elif prediction == 3:
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
                    
                    if game_over != True:

                        snake.body.insert(0,(snake.body[0][0]+dir[0], snake.body[0][1]+dir[1]))

                        
                        if A[snake.body[0][0],snake.body[0][1]] == 2:
                             grow = True
                             num_moves += food_bonus

                        if grow != True:
                            A[snake.body[-1][0],snake.body[-1][1]]=0
                            snake.body.pop(-1)
                        else:
                            grow = False

                        for block in snake.body:
                            A[block[0],block[1]]=1

                        if game == True:

                            for x in range(len(A)):
                                for y in range(len(A)):
                                    if A[x,y] == 0:
                                        draw_rect(y,x,BLACK,box_size,border)
                                    if A[x,y] == 1:
                                        draw_rect(y,x,BLUE,box_size,border)
                                    if A[x,y] == 2:
                                        draw_rect(y,x,GREEN,box_size,border)
                            CLOCK.tick(fps)
                            pygame.display.update()

                        tick += 1
                        num_moves -= 1

                #Once we've broken out of the loop quit the game
                score = tick*(len(snake.body))

                model.score += score

            model.score = model.score/trials

    gen.score_sort()
    
    print("Gen ", generation, " Score is: ", gen.population[0].score)

    gen.generation_mate()


pygame.quit()
quit()









