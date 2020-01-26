from utils import less_then, filter_less_then, filter_less_thenY
from World import World
import random

def smallestX(a,b):
  if a is None and b is None:
    return None
  elif a is None:
    return b
  elif b is None:
    return a
  else:
    return a if a.x < b.x else b
def biggestY(a,b):
  if a is None and b is None:
    return None
  elif a is None:
    return b
  elif b is None:
    return a
  else:
    return a if a.y > b.y else b

class Goal:
  def __init__(self):
    self.condition = self.check_filter_cond
    #self.curr_conditon = None
    self.move_obj  = None
    self.left_obj  = None
    self.north_obj = None
    
  def check_condition(self,world):
    if self.condition(world):
      print("Condition met on World {}".format(world.id))
      return True
    else:
      # print("Condition not met on World {}".format(world.id))
      return False
   
  def curr_condition(self,world):
    move_obj = world.get_object(self.move_obj.id)
    left_obj = world.get_object(self.left_obj.id) if self.left_obj is not None else None
    north_obj = world.get_object(self.north_obj.id) if self.north_obj is not None else None
    return filter_less_then(move_obj,left_obj) and filter_less_thenY(north_obj,move_obj) 
    
  def curr_condition_north(self,world):
    move_obj = world.get_object(self.move_obj.id)
    if self.north_obj is None:
      return False
    else:
      north_obj = world.get_object(self.north_obj.id)
      return filter_less_thenY(north_obj,move_obj)
  
  def curr_condition_left(self,world):
    move_obj = world.get_object(self.move_obj.id)
    if self.left_obj is None:
      return False
    else:
      left_obj = world.get_object(self.left_obj.id)
      return filter_less_then(move_obj,left_obj)
        
    
  def check_filter_cond(self,world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    
    southmost_circle = world.get_southmost_circle()
    northmost_square = world.get_northmost_square()
    northmost_triangle = world.get_northmost_triangle()
    southmost_triangle = world.get_southmost_triangle()
        
    return filter_less_then(rightmost_circle, leftmost_triangle) and filter_less_then(rightmost_triangle, leftmost_square) and filter_less_then(rightmost_circle, leftmost_square) and filter_less_thenY(northmost_square, southmost_circle) and filter_less_thenY(northmost_triangle, southmost_circle) and filter_less_thenY(northmost_square, southmost_triangle)  
  
  def reset_curr_condition(self):
    self.reset_move_object()     
    self.reset_left_object()   
    self.reset_north_object()
  def reset_move_object(self):
    self.move_obj  = None
  def reset_north_object(self):
    self.north_obj = None      
  def reset_left_object(self):
    self.left_obj = None 
    
  def check_conditions_for_obj(self,obj,world):
    if obj.shape == 'square':
      print("Error in check_conditions_for_obj at Goal, squares connot be considered")
      return None
    elif obj.shape == 'triangle':
      return filter_less_thenY(world.get_northmost_square(),obj) and filter_less_then(obj, world.get_leftmost_square())
    elif obj.shape == 'circle':
      return filter_less_thenY(biggestY(world.get_northmost_square(),world.get_northmost_triangle()),obj) and filter_less_then(obj,smallestX(world.get_leftmost_square(),world.get_leftmost_triangle()))
    else:
      print("Error in check_conditions_for_obj at Goal, unknown shape")
      return None
       
  #this to be changed if roll back
  def update_curr_condition(self,world):
    objs = [o for o in world.objects if not o.shape == 'square']
    random.shuffle(objs)
    for obj in objs:
      if not self.check_conditions_for_obj(obj,world):
        self.update_obj_of_interest(obj, world)
        return
    
  def update_obj_of_interest(self, obj, world):
    self.move_obj = obj
    if obj.shape == 'triangle':
      northmost_square = world.get_northmost_square()
      leftmost_square  = world.get_leftmost_square()
      if not filter_less_thenY(northmost_square,obj):
        self.north_obj = northmost_square
      if not filter_less_then(obj, leftmost_square):
        self.left_obj = leftmost_square
    else:  #circle case
      northmost_square = world.get_northmost_square()
      northmost_triangle = world.get_northmost_triangle()
      leftmost_square = world.get_leftmost_square()
      leftmost_triangle = world.get_leftmost_triangle()
      north_obj = biggestY(northmost_square,northmost_triangle)
      left_obj = smallestX(leftmost_square,leftmost_triangle)
      if not filter_less_thenY(north_obj, obj):
        self.north_obj = north_obj
      if not filter_less_then(obj, left_obj):
        self.left_obj = left_obj
    
  def filter_examples_check(self,exmp):
    state = exmp.init_state
    move_obj = state.get_object(exmp.hier[1])
    rel_to_obj = state.get_object(exmp.hier[2])
    return move_obj.x < rel_to_obj.x  
    
  #this to be changed if roll back
  '''def update_obj_of_interest(self,world):
    if self.curr_condition == self.cond_rmc_lt_lmt:
      self.move_obj = world.get_rightmost_circle()
      self.rel_to_obj = world.get_leftmost_triangle()
    elif self.curr_condition == self.cond_rmt_lt_lms:
      self.move_obj = world.get_rightmost_triangle()
      self.rel_to_obj = world.get_leftmost_square()
    elif self.curr_condition == self.cond_rmc_lt_lms:
      self.move_obj = world.get_rightmost_circle()
      self.rel_to_obj = world.get_leftmost_square()
    elif self.curr_condition == self.cond_nms_lt_smc:
      self.move_obj = world.get_southmost_circle()
      self.rel_to_obj = world.get_northmost_square()
    elif self.curr_condition == self.cond_nmt_lt_smc:
      self.move_obj = world.get_southmost_circle()
      self.rel_to_obj = world.get_northmost_triangle()
    elif self.curr_condition == self.cond_nms_lt_smt:
      self.move_obj = world.get_southmost_triangle()
      self.rel_to_obj = world.get_northmost_square()
    else:
      print("Error in update_obj_of_interest in Goal")
    
  #conditions for hier these to change if rollback
  def cond_rmc_lt_lmt(self,world):
    if self.move_obj is not None:
      return filter_less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_circle = world.get_rightmost_circle()
      leftmost_triangle = world.get_leftmost_triangle()
      return filter_less_then(rightmost_circle, leftmost_triangle)
  def cond_rmt_lt_lms(self,world=None):
    if self.move_obj is not None:
      return filter_less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_triangle = world.get_rightmost_triangle()
      leftmost_square = world.get_leftmost_square()
      return filter_less_then(rightmost_triangle, leftmost_square)
  def cond_rmc_lt_lms(self,world=None):
    if self.move_obj is not None:
      return filter_less_then(world.get_object(self.move_obj.id), world.get_object(self.rel_to_obj.id))
    else:
      rightmost_circle = world.get_rightmost_circle()
      leftmost_square = world.get_leftmost_square()
      return filter_less_then(rightmost_circle, leftmost_square)
      
      
  #here    
  def cond_nms_lt_smc(self,world):
    if self.move_obj is not None:
      return filter_less_thenY(world.get_object(self.rel_to_obj.id), world.get_object(self.move_obj.id))
    else:
      southmost_circle = world.get_southmost_circle()
      northmost_square = world.get_northmost_square()
      return filter_less_thenY(northmost_square, southmost_circle)
  def cond_nmt_lt_smc(self,world=None):
    if self.move_obj is not None:
      return filter_less_thenY(world.get_object(self.rel_to_obj.id), world.get_object(self.move_obj.id))
    else:
      southmost_circle = world.get_southmost_circle()
      northmost_triangle = world.get_northmost_triangle()
      return filter_less_thenY(northmost_triangle, southmost_circle)
  def cond_nms_lt_smt(self,world=None):
    if self.move_obj is not None:
      return filter_less_thenY(world.get_object(self.rel_to_obj.id), world.get_object(self.move_obj.id))
    else:
      southmost_triangle = world.get_southmost_triangle()
      northmost_square = world.get_northmost_square()
      return filter_less_thenY(northmost_square, southmost_triangle)'''  
    
  '''def check_curr_condition(self,world):
    conds = [cond_rmc_lt_lmt,cond_rmt_lt_lms,cond_rmc_lt_lms]
    output = self.curr_condition(world)
    if output:
      i = conds.index(self.curr_condition)
      self.curr_condition = conds[(i+1)%len(conds)]
    return output'''
