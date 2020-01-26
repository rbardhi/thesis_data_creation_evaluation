from utils import less_then
from World import World

class Goal:
  def __init__(self):
    self.condition = self.niteshs_goal
    self.curr_conditon = self.cond_rmt_lt_lms
  def check_condition(self,world):
    if self.condition(world):
      print("Condition met on World {}".format(world.id))
      return True
    else:
     # print("Condition not met on World {}".format(world.id))
      return False
    #return self.condition(world)
  def niteshs_goal(self,world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    return less_then(rightmost_circle, leftmost_triangle) and less_then(rightmost_triangle, leftmost_square) and less_then(rightmost_circle, leftmost_square)
  
  def reset_curr_condition(self):
    self.curr_condition = self.cond_rmt_lt_lms
  
  def update_curr_condition(self,world):
    conds = [self.cond_rmt_lt_lms,self.cond_rmc_lt_lms,self.cond_rmc_lt_lmt]
    for cond in conds:
      if not cond(world):
        self.curr_condition = cond
        break
    
  #conditions for hier
  def cond_rmc_lt_lmt(self,world):
    rightmost_circle = world.get_rightmost_circle()
    leftmost_triangle = world.get_leftmost_triangle()
    return less_then(rightmost_circle, leftmost_triangle)
  def cond_rmt_lt_lms(self,world):
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_square = world.get_leftmost_square()
    return less_then(rightmost_triangle, leftmost_square)
  def cond_rmc_lt_lms(self,world):
    rightmost_circle = world.get_rightmost_circle()
    leftmost_square = world.get_leftmost_square()
    return less_then(rightmost_circle, leftmost_square)
    
    
  '''def check_curr_condition(self,world):
    conds = [cond_rmc_lt_lmt,cond_rmt_lt_lms,cond_rmc_lt_lms]
    output = self.curr_condition(world)
    if output:
      i = conds.index(self.curr_condition)
      self.curr_condition = conds[(i+1)%len(conds)]
    return output'''
