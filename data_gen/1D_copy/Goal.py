from utils import less_then, filter_less_then
from World import World

class Goal:
  def __init__(self):
    self.condition = self.niteshs_goal
    self.curr_conditon = None
    #this to be changed if roll back
    self.move_obj = None
    self.rel_to_obj = None
  def check_condition(self,world):
    if self.condition(world):
      print("Condition met on World {}".format(world.id))
      return True
    else:
      # print("Condition not met on World {}".format(world.id))
      return False
    #return self.condition(world)
  def check_filter_cond(self,world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    return filter_less_then(rightmost_circle, leftmost_triangle) and filter_less_then(rightmost_triangle, leftmost_square) and filter_less_then(rightmost_circle, leftmost_square)
  def niteshs_goal(self,world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    return less_then(rightmost_circle, leftmost_triangle) and less_then(rightmost_triangle, leftmost_square) and less_then(rightmost_circle, leftmost_square)
  
  #this to be changed if roll back
  def reset_curr_condition(self):
    self.curr_condition = None
    self.move_obj = None     #
    self.rel_to_obj = None   #
  
  #this to be changed if roll back
  def update_curr_condition(self,world):
    conds = [self.cond_rmc_lt_lms,self.cond_rmc_lt_lmt,self.cond_rmt_lt_lms]
    for cond in conds:
      if not cond(world):
        self.curr_condition = cond
        self.update_obj_of_interest(world)
        return
    
  #this to be changed if roll back
  def update_obj_of_interest(self,world):
    if self.curr_condition == self.cond_rmc_lt_lmt:
      self.move_obj = world.get_rightmost_circle()
      self.rel_to_obj = world.get_leftmost_triangle()
    elif self.curr_condition == self.cond_rmt_lt_lms:
      self.move_obj = world.get_rightmost_triangle()
      self.rel_to_obj = world.get_leftmost_square()
    elif self.curr_condition == self.cond_rmc_lt_lms:
      self.move_obj = world.get_rightmost_circle()
      self.rel_to_obj = world.get_leftmost_square()
    else:
      print("Error in update_obj_of_interest in Goal")
    
  #conditions for hier these to change if rollback
  def cond_rmc_lt_lmt(self,world):
    if self.move_obj is not None:
      return less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_circle = world.get_rightmost_circle()
      leftmost_triangle = world.get_leftmost_triangle()
      return less_then(rightmost_circle, leftmost_triangle)
  def cond_rmt_lt_lms(self,world=None):
    if self.move_obj is not None:
      return less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_triangle = world.get_rightmost_triangle()
      leftmost_square = world.get_leftmost_square()
      return less_then(rightmost_triangle, leftmost_square)
  def cond_rmc_lt_lms(self,world=None):
    if self.move_obj is not None:
      return less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_circle = world.get_rightmost_circle()
      leftmost_square = world.get_leftmost_square()
      return less_then(rightmost_circle, leftmost_square)
  
  def filter_examples_check(self,exmp):
    state = exmp.init_state
    move_obj = state.get_object(exmp.hier[1])
    rel_to_obj = state.get_object(exmp.hier[2])
    return move_obj.x < rel_to_obj.x  
    
  '''def check_curr_condition(self,world):
    conds = [cond_rmc_lt_lmt,cond_rmt_lt_lms,cond_rmc_lt_lms]
    output = self.curr_condition(world)
    if output:
      i = conds.index(self.curr_condition)
      self.curr_condition = conds[(i+1)%len(conds)]
    return output'''
