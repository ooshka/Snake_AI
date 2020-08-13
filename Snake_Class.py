#Class containing information about the snake for the snake game
#2020-08-13

class Snake:
	
	def __init__(self, head):
		self.body = [head]


	def fill(self, B, head, count, s_len):

	    if count >= s_len:
	        pass
	        #return count

	    if head[0] < 0 or head[0] >= len(B):
	        return count
	    if head[1] < 0 or head[1] >= len(B):
	        return count
	    if B[head[0], head[1]] == 1:
	        return count

		B[head[0], head[1]] = 3
		count+=1

		left_count = self.fill(B, [head[0], head[1] - 1], count, s_len)

		if left_count>count:
			count = left_count

		right_count = self.fill(B, [head[0], head[1] + 1], count, s_len)

		if right_count>count:
			count = right_count

		up_count = self.fill(B, [head[0] - 1, head[1]], count, s_len)

		if up_count>count:
			count = up_count

		down_count = self.fill(B, [head[0] + 1, head[1]], count, s_len)

		if down_count>count:
			count = down_count
	            
	    return count