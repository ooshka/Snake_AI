#####################################################################################################################
## Title: Disply
#####################################################################################################################
## Description: This contains the details necessary to display the game Snake using pygame as the medium.
##              Two main types of display updates are used: one to display an individual game running, and one to
##				track the running score of each neural network generation
#####################################################################################################################
## Author: Alex Wadey
## Version: 1.0
## Email: wadeyalex@gmail.com
## Status: active
#####################################################################################################################

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import sys

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

#Pygame Display Class
class Display:

	def __init__(self):

		#Initialize game viewing to true
		self.game_view = True
		#program over flag for determining program end
		self.program_over = False

	    #Overall display size
		self.disp_size = 700
		#Depth of the top text bar above snake and graph
		self.top_bar = 40
		#grid size along one axis (vertical and horizontal)
		self.n_boxes = 10
		#Frame rate of th
		self.fps = 20
		#Border between grid boxes for the snake
		self.border = 1
		#Gap between edge of display and axis lines
		self.axis_gap = 80
		#Gap between edge of axis and axis text
		self.axis_text_gap = 35
		#Gap between axis data labels and axis label text
		self.text_gap = 25
		#Size of data points on graph
		self.circle_size = 5
		#Size of each axis
		self.axis_size = self.disp_size - 2 * self.axis_gap
		#Zero point of axis
		self.axis_zero = (self.disp_size + self.axis_gap, self.top_bar + self.disp_size - self.axis_gap)

		#The size of individual boxes within the snake grid
		self.box_size = int(self.disp_size/self.n_boxes)

		#Initialize generational graph data to null lists
		self.x_data = []
		self.y_data = []

		#Initialilze pygame
		pygame.init()

		#Set initial display window size and variable
		self.screen = pygame.display.set_mode((self.disp_size * 2,self.disp_size + self.top_bar))

		#Set display caption
		pygame.display.set_caption("Snake Genetic AI")

		#Initialize pygame clock to set framerate
		self.clock = pygame.time.Clock()

		#Initial Snake Grid Setup
		self.screen.fill(WHITE)
		box_size = int(self.disp_size/self.n_boxes)

		#Initial Graph Setup
		graph_background = pygame.Rect(self.disp_size, self.top_bar, self.disp_size, self.disp_size)
		pygame.draw.rect(self.screen, BLACK, graph_background)


	#Draw a pygame rectangle (used to slightly simplify rectangle code in main function)
	def Draw_Rect(self, x, y, color, size, border, top_bar):

	    rect = pygame.Rect(x*size, y*size + top_bar, size-border, size-border)
	    pygame.draw.rect(self.screen, color, rect)

	def Game_Update(self, array, generation, score):

		#Check for user quit
		for event in pygame.event.get(pump=True):
			if event.type == pygame.QUIT:
				game_over = True
				program_over = True
				self.Quit()
				return -1

		#Refresh the game background behind the snake
		snake_background = pygame.Rect(0,0,self.disp_size, self.disp_size + self.top_bar)
		pygame.draw.rect(self.screen, WHITE, snake_background)

		#Place generation text at the top of the screen
		font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 24)
		gen_text = font.render(f"Generation: {generation}", True, BLACK)
		gen_text_rect = gen_text.get_rect()
		gen_text_rect.center = (int(self.disp_size*0.25), int(self.top_bar/2))
		self.screen.blit(gen_text, gen_text_rect)

		#Place score text at the top of the screen
		score_text = font.render(f"Score: {score}", True, BLACK)
		score_text_rect = gen_text.get_rect()
		score_text_rect.center = (int(self.disp_size*0.75), int(self.top_bar/2))
		self.screen.blit(score_text, score_text_rect)

		#Draw rectangles in accordance with the Array values for snake body, blanks, and food
		for x in range(len(array)):
			for y in range(len(array)):
				#If there is no snake
				if array[x,y] == 0:
					self.Draw_Rect(y, x, BLACK, self.box_size, self.border, self.top_bar)
				#If there is snake body
				if array[x,y] == 1:
					self.Draw_Rect(y, x, BLUE, self.box_size , self.border, self.top_bar)
				#If there is food
				if array[x,y] == 2:
					self.Draw_Rect(y, x, GREEN, self.box_size, self.border, self.top_bar)

		#Set the game fps to self.fps
		self.clock.tick(self.fps)

		#Update display
		pygame.display.update()

	#Used for updating the graphical score tracker with new information
	def Gen_Update(self, gen_num, gen):

		#Check for user quit
		for event in pygame.event.get(pump=True):
			if event.type == pygame.QUIT:
				game_over = True
				program_over = True
				self.Quit()
				return -1

		#Append the generation number to the x-axis data
		self.x_data.append(gen_num + 1)

		#Append the top score from each generation to the y-axis data
		self.y_data.append(gen.population[0].score)

		#At the end of each generation reset the snake display so the previous high score run isn't locked to screen
		for x in range(self.n_boxes):
			for y in range(self.n_boxes):
				self.Draw_Rect(y, x, BLACK,self.box_size, self.border, self.top_bar)

		#Reset the graphical display background
		graph_background = pygame.Rect(self.disp_size, self.top_bar, self.disp_size, self.disp_size)
		pygame.draw.rect(self.screen, BLACK, graph_background)

		#Draw x and y axis
		pygame.draw.line(self.screen, WHITE, (self.disp_size + self.axis_gap, self.top_bar + self.axis_gap), (self.disp_size + self.axis_gap, self.top_bar + self.disp_size - self.axis_gap))
		pygame.draw.line(self.screen, WHITE, (self.disp_size + self.axis_gap, self.top_bar + self.disp_size - self.axis_gap), (self.disp_size*2 - self.axis_gap, self.top_bar + self.disp_size - self.axis_gap))

		#Round to the nearest 5 on both the score and generation
		x_max = (int(max(self.x_data)/5)+1)*5
		y_max = (int(max(self.y_data)/5)+1)*5

		#Set x-axis font
		axis_font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 15)
		x_axis = axis_font.render("Generation", True, WHITE)
		x_axis_rect = x_axis.get_rect()
		x_axis_rect.center = (int(self.axis_size/2 + self.axis_zero[0]), int(self.axis_zero[1] + self.text_gap + self.axis_text_gap))
		self.screen.blit(x_axis, x_axis_rect)

		#Set y-axis font
		y_axis = axis_font.render("Top Model Score", True, WHITE)
		y_axis = pygame.transform.rotate(y_axis, 90)
		y_axis_rect = y_axis.get_rect()
		y_axis_rect.center = (int(self.axis_zero[0] - self.text_gap - self.axis_text_gap), int(self.axis_zero[1] - self.axis_size/2))
		self.screen.blit(y_axis, y_axis_rect)

		#Set the spacings for the axis label points
		spacing = self.axis_size/10
		x_ratio = x_max/self.axis_size
		y_ratio = y_max/self.axis_size

		#Iteratively set the label points for each axis
		for i in range(11):

			#Set x-axis label points
			x_text = axis_font.render(str(int(spacing*i*x_ratio)), True, WHITE)
			x_text_rect = x_text.get_rect()
			x_text_rect.center = (int(self.axis_zero[0] + spacing*i), int(self.axis_zero[1] + self.text_gap))
			self.screen.blit(x_text, x_text_rect)

			#Set y-axis label points
			y_text = axis_font.render(str(int(spacing*i*y_ratio)), True, WHITE)
			y_text_rect = y_text.get_rect()
			y_text_rect.center = (int(self.axis_zero[0] - self.text_gap), int(self.axis_zero[1] - spacing*i))
			self.screen.blit(y_text, y_text_rect)

		#Draw data points for generational scores as circles on the graph
		for i in range(len(self.x_data)):
			pygame.draw.circle(self.screen, RED,(int(self.axis_zero[0] + 1/x_ratio * self.x_data[i]), int(self.axis_zero[1] - 1/y_ratio * self.y_data[i])), int(self.circle_size))

		self.clock.tick(self.fps)

		#Update pygame display
		pygame.display.update()

	def Quit(self):

		#Write note to user
		print("User Has Exited Game")
		#Quit pygame
		pygame.quit()
		#Quit the program
		sys.exit(0)


