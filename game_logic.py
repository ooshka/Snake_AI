#####################################################################################################################
## Title: Game Logic
#####################################################################################################################
## Description: This code contains all logic necessary to run the game Snake.
##              The Game_Logic class contains all necessary logic checks and information tracking, while the 
##				Play_Game() function plays one round of the game.  Requires inputs for snake direction				
#####################################################################################################################
## Author: Alex Wadey
## Version: 1.0
## Email: wadeyalex@gmail.com
## Status: active
#####################################################################################################################

import numpy as np
import random as rand
import copy

#Define the cardinal directions of the game board
UP    = [-1,0]
LEFT  = [0,-1]
DOWN  =	[1,0]
RIGHT =	[0,1]

#Class defining all logic required for snake game operation
class Game_Logic:

	#Pass in the number of boxes being used along a side of the board
	def __init__(self, n_boxes, food_bonus = 25):

		#Array to store game information
		self.array = np.zeros([n_boxes,n_boxes])
		#Initialize the snake to be an empty list
		self.snake = []

		#Game over falg
		self.game_over = False
		#Initial direction of the snake head set to null
		self.dir = [0,0]
		#Flag defining if the snake should grow or not this iteration
		self.grow = False
		#A count of how many iterations the snake has been alive
		self.tick = 0
		#Snake score
		self.score = 0
		#The number of cells along one side of the game board (i.e. 8x8 or 10x10 depending how big you want the grid to be)
		self.n_boxes = n_boxes
		#The number of additional moves the snake is awarded for each piece of food it acquires (used to prevent the snake infinitely looping to rack up score)
		self.food_bonus = food_bonus
		#Tracks the number of moves left to the snake
		self.num_moves = food_bonus
		#Flag to note if there is food on the board
		self.food_test = False
		#Place holder for the food location on the board
		self.food_loc = []

		#Place the head of the snake at a random board location
		self.Snake_Head()
		#Place food at a random board location
		self.Food_Test()

	#Find a random clear spot on the board and return the coordinates
	def Rand_Insert(self):

		#Set random coordinates
		Ax = rand.randint(0,self.n_boxes-1)
		Ay = rand.randint(0,self.n_boxes-1)
		
		#If the coordinates were filled continue iterating until we find a clear spot
		while self.array[Ax,Ay] != 0:
			Ax = rand.randint(0,self.n_boxes-1)
			Ay = rand.randint(0,self.n_boxes-1)
		#Return coordinates
		return Ax,Ay

	#Check to see if food is on the board.  If not add food to the board
	def Food_Test(self):

		#Iterate over the board to see if food is present (denoted by the value 2 in the array)
		for x in range(len(self.array)):
			for y in range(len(self.array)):
				if self.array[x,y]==2:
					self.food_test = True

	    #If there is no food on the grid insert a piece of food at a random location not already filled by snek
		if self.food_test != True:

			Ax,Ay = self.Rand_Insert()

			self.array[Ax,Ay] = 2
			self.food_loc = [Ax,Ay]
			#Set the food test to false to refresh the check for next iteration
			self.food_test = False

	#Insert the head of the snake on the board at the beginning of the game
	def Snake_Head(self):

		Ax,Ay = self.Rand_Insert()
		#Add the snake head to the list containing all snake body parts
		self.snake.append(np.array([Ax,Ay]))
		self.array[Ax,Ay] = 1

	#Function to determine how much space the snake has to move in each of the cardinal directions
	#This function is recursive and the "pos" that is passed in is simply a placeholder for where the algorithm is currently centered on the board
	def Snake_Fill(self, B, pos, count):

		#Record the snake's length
		length = len(self.snake)

		#Set x and y to the two coordinates of the algorithm's current positino
		x = int(pos[0])
		y = int(pos[1])

		#If we've found more space than the snake has length simply return
		if count >= length:
			return count

		#If either of the head coordinates is off the board that's the furthest we can go in that direction so return the total count
		if x < 0 or x >= len(B):
			return count
		elif y < 0 or y >= len(B):
			return count
		#If the space we have entered is either part of the snake (cell contains 1) or has already been traversed by the algorithm (cell contains 3) we've hit the end and return the total count
		elif B[x, y] == 1 or B[x, y] == 3:
			return count

		#Mark the array so that we know we have traversed this space
		B[x, y] = 3
		#Iterate the total count to track how many spaces we have traversed
		count+=1

		#Call the function again but with the new positino being one cell to the left
		left_count = self.Snake_Fill(B, pos + LEFT, count)

		#If we find any more space in this direction suplant our current count
		if left_count>count:
			count = left_count

		#Do the same thing with each of the other cardinal directions
		right_count = self.Snake_Fill(B, pos + RIGHT, count)

		if right_count>count:
			count = right_count

		up_count = self.Snake_Fill(B, pos + UP, count)

		if up_count>count:
			count = up_count

		down_count = self.Snake_Fill(B, pos + DOWN, count)

		if down_count>count:
			count = down_count
	    
	    #Return the count to the previous level of the algorithm    
		return count

	#Change the direction of the snake based on the prediction from the neural network
	def Direction_Change(self, pred):

		#We cannot turn completely around so if the snake is going in the opposite direction of the prediction do nothing

		#If pred == Left
		if pred == 0:
			#if going right do nothing
			if self.dir == RIGHT:
				pass
			else:
				self.dir = LEFT

		#If pred == Right
		elif pred == 1:
			#if going left do nothing
			if self.dir == LEFT:
				pass
			else:
				self.dir = RIGHT

		#If pred == Up
		elif pred == 2:
			#if going down do nothing
			if self.dir == DOWN:
				pass
			else:
				self.dir = UP

		#If pred == Down
		elif pred == 3:
			#if going up do nothing
			if self.dir == UP:
				pass
			else:
				self.dir = DOWN

	#Check the game logic for a game over state
	def Game_Over_Check(self):

		#If the next location of the snake head is to be off board then game over is reached
		loc = self.snake[0] + self.dir

		if loc[0] < 0 or loc[0] > self.n_boxes - 1 or loc[1] < 0 or loc[1] > self.n_boxes - 1:

			self.game_over = True

		#Check to see if the head location matches one of the body block locations.  If so we the snake has hit itself
		for block in self.snake[1:]:

			if np.all(block == self.snake[0]):
				self.game_over = True

		#If the number of moves available to the snake reaches zero the game is over
		if self.num_moves == 0:
			self.game_over = True

	#Iterate the snake game logic once
	def Snake_Iterate(self):

		#Insert a new head of the snake in the direction of travel and add it to the front of the snake list
		self.snake.insert(0, self.snake[0] + self.dir)

		#If the new head is now on a piece food (equal to 2) then we need to grow the snake
		if self.array[self.snake[0][0], self.snake[0][1]] == 2:
			#Set grow flag to true
			self.grow = True
			#Increase the number of moves available by the food bonus
			self.num_moves += self.food_bonus

		#If we aren't growing
		if self.grow == False:
			#Make the cell of the snake tail equal to zero
			self.array[self.snake[-1][0], self.snake[-1][1]] = 0
			#Remove the tail from the snake list
			self.snake.pop(-1)
		else:
			#If grow is equal to true we don't need to alter the tail of the snake
			self.grow = False
			#If we have grown we need to set the food test to false
			self.food_test = False

		#Ensure every array index with a snake in it is set to one
		for block in self.snake:
			self.array[block[0], block[1]] = 1

		#Increase the number of iterations the snake has been alive for
		self.tick += 1
		#Reduce the number of moves available to the snake by 1
		self.num_moves -= 1
		#Calculate the score of the snake
		self.score = self.tick*(len(self.snake))

#Play one round of the snake game
def Play_Game(model, display, generation, n_inputs):

	game = Game_Logic(display.n_boxes)

	#While game is not over
	while not game.game_over:

		game.Food_Test()

		#Create input array for neural network
		#We actually likely don't need any inputs aside from the distance to the foods and the fill information
		#B/C the fill info gives some knowledge about wall distance and self distance which are essentially the same thing

		X = np.zeros(n_inputs)

		#Distance to food
		#Distance in the vertical direction
		X[0] = game.food_loc[0] - game.snake[0][0]

		#Distance in the horizontal direction
		X[1] = game.food_loc[1] - game.snake[0][1]

		#Fill function to decide which direction is safest to turn
		B = copy.copy(game.array)
		#left fill
		#If there is room to fit the snake
		if game.Snake_Fill(B, game.snake[0] + LEFT, 0) == len(game.snake):
			X[2] = 1
		#If there is no room
		else:
			X[2] = 0

		#right fill
		if game.Snake_Fill(B, game.snake[0] + RIGHT, 0) == len(game.snake):
			X[3] = 1
		else:
			X[3] = 0
		
		#up fill
		if game.Snake_Fill(B, game.snake[0] + UP, 0) == len(game.snake):
			X[4] = 1
		else:
			X[4] = 0

		#down fill
		if game.Snake_Fill(B, game.snake[0] + DOWN, 0) == len(game.snake):
			X[5] = 1
		else:
			X[5] = 0

		#Get prediction from the output of the model based on the current layout of the board
		output = model.forward(X)

		#Take the direction with the highest level of confidence
		prediction = np.argmax(output)

		#Change the direction of snake based on the network's prediction
		game.Direction_Change(prediction)

		#Check for game over state
		game.Game_Over_Check()


		if not game.game_over:
			#Iterate the game logic once
			game.Snake_Iterate()

			#If we want to see the game
			if display.game_view == True:
				#Display the game logic changing
				display.Game_Update(game.array, generation, game.score)

	#After the game has been played set the display flag to false
	display.game_view = False
	#Add the trial run's score to the model's
	model.score+=game.score