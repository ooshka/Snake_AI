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

def draw_snake_rect(x, y, color, size, border, top_bar):

    rect = pygame.Rect(x*size, y*size + top_bar, size-border, size-border)
    pygame.draw.rect(SCREEN, color, rect)

def random_insert(Array, size):
    
    x = random.randint(0,size-1)
    y = random.randint(0,size-1)
    while Array[x,y] != 0:
        x = random.randint(0,size-1)
        y = random.randint(0,size-1)
    return x,y

def main():

    print("Snake Start")

    ############################ Start of main function code #############################################


    ################################### Pygame Details ###################################################
    global SCREEN, CLOCK

    game = False
    program_over = False
    first = True

    disp_size = 700
    top_bar = 40
    n_boxes = 10
    fps = 15
    border = 1
    high_score = 0
    model_high_score = 0

    #Neural Network Details
    ######################################################################################################

    pop_size = 75
    trials = 5
    generations = 1000
    n_inputs = 6

    gen = nw.Generation(n_inputs = n_inputs, n_neurons = 64, n_outputs = 4, population_size = pop_size)

    ######################################################################################################
    #Setup Display Details

    pygame.init()

    SCREEN = pygame.display.set_mode((disp_size * 2,disp_size + top_bar))
    pygame.display.set_caption("Snake Genetic AI")

    CLOCK = pygame.time.Clock()

    #Initial Snake Grid Setup
    SCREEN.fill(WHITE)
    box_size = int(disp_size/n_boxes)

    #Initial Graph Setup
    graph_background = pygame.Rect(disp_size, top_bar, disp_size, disp_size)
    pygame.draw.rect(SCREEN, BLACK, graph_background)

    axis_gap = 80
    axis_text_gap = 30
    axis_base = 5
    text_gap = 25
    circle_size = 5
    axis_size = disp_size - 2 * axis_gap
    axis_zero = (disp_size + axis_gap, top_bar + disp_size - axis_gap)

    pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + axis_gap), (disp_size + axis_gap, top_bar + disp_size - axis_gap))
    pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + disp_size - axis_gap), (disp_size*2 - axis_gap, top_bar + disp_size - axis_gap))

    x_data = []
    y_data = []

    for generation in range(generations):

        for model in gen.population:

            if first == True:
                model.Model_Read()
                first = False

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
                    score = 0
                    food_bonus = 25
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

                                snake_background = pygame.Rect(0,0,disp_size, disp_size + top_bar)
                                pygame.draw.rect(SCREEN, WHITE, snake_background)
                                
                                font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 24)
                                gen_text = font.render(f"Generation: {generation+1}", True, BLACK)
                                gen_text_rect = gen_text.get_rect()
                                gen_text_rect.center = (int(disp_size*0.25), int(top_bar/2))

                                SCREEN.blit(gen_text, gen_text_rect)

                                score_text = font.render(f"Score: {score}", True, BLACK)
                                score_text_rect = gen_text.get_rect()
                                score_text_rect.center = (int(disp_size*0.75), int(top_bar/2))

                                SCREEN.blit(score_text, score_text_rect)

                                for x in range(len(A)):
                                    for y in range(len(A)):
                                        if A[x,y] == 0:
                                            draw_snake_rect(y,x,BLACK,box_size,border, top_bar)
                                        if A[x,y] == 1:
                                            draw_snake_rect(y,x,BLUE,box_size,border, top_bar)
                                        if A[x,y] == 2:
                                            draw_snake_rect(y,x,GREEN,box_size,border, top_bar)

                                CLOCK.tick(fps)
                                pygame.display.update()

                            tick += 1
                            num_moves -= 1

                        
                        score = tick*(len(snake.body))

                    game = False

                    if score > high_score and trial < trials:
                        high_score = score
                        game = True

                    #Once we've broken out of the loop quit the game
                    model.score += score
     
                model.score = model.score/trials

                if model.score > model_high_score:
                    model_high_score = model.score
                    model.Model_Write()

        gen.score_sort()

        x_data.append(generation+1)
        y_data.append(gen.population[0].score)

        for x in range(len(A)):
            for y in range(len(A)):
                draw_snake_rect(y,x,BLACK,box_size,border, top_bar)

        graph_background = pygame.Rect(disp_size, top_bar, disp_size, disp_size)
        pygame.draw.rect(SCREEN, BLACK, graph_background)

        pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + axis_gap), (disp_size + axis_gap, top_bar + disp_size - axis_gap))
        pygame.draw.line(SCREEN,WHITE, (disp_size + axis_gap, top_bar + disp_size - axis_gap), (disp_size*2 - axis_gap, top_bar + disp_size - axis_gap))

        x_max = (int(max(x_data)/5)+1)*5
        y_max = (int(max(y_data)/5)+1)*5

        axis_font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 14)

        x_axis = axis_font.render("Generation", True, WHITE)
        x_axis_rect = x_axis.get_rect()
        x_axis_rect.center = (int(axis_size/2 + axis_zero[0]), int(axis_zero[1] + text_gap + axis_text_gap))
        SCREEN.blit(x_axis, x_axis_rect)

        y_axis = axis_font.render("Average Score", True, WHITE)
        y_axis = pygame.transform.rotate(y_axis, 90)
        y_axis_rect = y_axis.get_rect()
        y_axis_rect.center = (int(axis_zero[0] - text_gap - axis_text_gap), int(axis_zero[1] - axis_size/2))
        SCREEN.blit(y_axis, y_axis_rect)

        spacing = axis_size/10
        x_ratio = x_max/axis_size
        y_ratio = y_max/axis_size

        for i in range(11):
            x_text = axis_font.render(str(int(spacing*i*x_ratio)), True, WHITE)
            x_text_rect = x_text.get_rect()
            x_text_rect.center = (int(axis_zero[0] + spacing*i), int(axis_zero[1] + text_gap))
            SCREEN.blit(x_text, x_text_rect)

            y_text = axis_font.render(str(int(spacing*i*y_ratio)), True, WHITE)
            y_text_rect = y_text.get_rect()
            y_text_rect.center = (int(axis_zero[0] - text_gap), int(axis_zero[1] - spacing*i))
            SCREEN.blit(y_text, y_text_rect)

        for i in range(len(x_data)):
            pygame.draw.circle(SCREEN, RED,(int(axis_zero[0] + 1/x_ratio * x_data[i]), int(axis_zero[1] - 1/y_ratio * y_data[i])), int(circle_size))

        pygame.event.pump()
        pygame.display.update()

        gen.generation_mate()


    pygame.quit()
    quit()

if __name__ == '__main__':
    main()







