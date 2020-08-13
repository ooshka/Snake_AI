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
user_input = False
program_over = False

disp_size = 600
n_boxes = 10
fps = 2
border = 1

#Neural Network Details
#######################################################################################

pop_size = 1
trials = 1
generations = 1
n_inputs = 6

gen = nw.Generation(n_inputs = n_inputs, n_neurons = 512, n_outputs = 4, population_size = pop_size)

#######################################################################################

for generation in range(generations):

    if generation == generations-1:
        game = True

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
                # #Place head of snake
                Sx,Sy = random_insert(A, n_boxes)
                snake = sn.Snake((Sx,Sy))
                A[Sx,Sy] = 1
                dir = [0,0]
                grow = False
                tick = 0
                food_bonus = 50
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

                    '''
                    #Distance to Walls

                    #Distance to left wall
                    X[0] = snake.body[0][1]
                    #Distance to right wall
                    X[1] = (n_boxes - 1) - X[0]
                    #Distance to top wall
                    X[2] = snake.body[0][0]
                    #Distance to bottom wall
                    X[3] = (n_boxes - 1) - X[2]
                    '''

                    #Distance to food

                    #Distance in the vertical direction
                    X[0] = Ax - snake.body[0][0]

                    #Distance in the horizontal direction
                    X[1] = Ay - snake.body[0][1]


                    '''
                    #Distance to Self
                    
                    #Distance to self left
                    i = snake.body[0][1] - 1
                    self_found = False

                    while i >= 0:
                        if A[snake.body[0][0], i] == 1:
                            X[6] = snake.body[0][1] - i
                            self_found = True
                        i-=1
                    if self_found == False:
                        X[6] = snake.body[0][1]

                    #Distance to self right
                    i = snake.body[0][1] + 1
                    self_found = False

                    while i < n_boxes:
                        if A[snake.body[0][0], i] == 1:
                            X[7] = i - snake.body[0][1]
                            self_found = True
                        i+=1
                    if self_found == False:
                        X[7] = (n_boxes - 1) - snake.body[0][1]

                    #Distance to self up
                    i = snake.body[0][0] - 1
                    self_found = False

                    while i >= 0:
                        if A[i, snake.body[0][1]] == 1:
                            X[8] = snake.body[0][0] - i
                            self_found = True
                        i-=1
                    if self_found == False:
                        X[8] = snake.body[0][0]

                    #Distance to self down
                    i = snake.body[0][0] + 1
                    self_found = False

                    while i < n_boxes:
                        if A[i, snake.body[0][1]] == 1:
                            X[9] = i - snake.body[0][0]
                            self_found = True
                        i+=1
                    if self_found == False:
                        X[9] = (n_boxes - 1) - snake.body[0][0]
                    '''

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


                    print(X)

                    #Get prediction from the output of the model being iterated through
                    output = model.forward(X)
                    prediction = numpy.argmax(output)

                    if game == True:

                        #Check for game over or user quit
                        for event in pygame.event.get():

                            #If we want to use user input we take in keyboard inputs
                            if user_input == True:

                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_LEFT and dir != [0,1]:
                                        dir = [0,-1]
                                    elif event.key == pygame.K_RIGHT and dir != [0,-1]:
                                        dir = [0,1]
                                    elif event.key == pygame.K_UP and dir != [1,0]:
                                        dir = [-1,0]
                                    elif event.key == pygame.K_DOWN and dir != [-1,0]: 
                                        dir = [1,0]
                                    if event.key == pygame.K_ESCAPE:
                                        game_over = True

                            if event.type == pygame.QUIT:
                                game_over = True
                                program_over = True

                    if user_input == False:

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
                                print("Game Over Self Hit ", ind)
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
                score = tick*(len(snake.body)-1)

                model.score += score

            model.score = model.score/trials

    gen.score_sort()


    print("Gen ", generation, " Score is: ", gen.population[0].score)

    gen.generation_mate()


pygame.quit()
quit()









