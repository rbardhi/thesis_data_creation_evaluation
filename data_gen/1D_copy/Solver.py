from Example import Example
from Action import Action
from SearchTree import Node, SearchTree
from utils import draw_image
from utils import draw_example
from pprint import pprint

max_tries = 15

class Solver:
  def __init__(self, goal, policy):
    self.goal = goal
    self.policy = policy
  
  def next_step(self,world):
    temp = world.copy()
    tries = 0
    self.goal.update_curr_condition(temp)
    while not self.goal.check_condition(temp):
      hier = temp.copy()
      tries += 1
      #draw_image(temp)
      if tries > max_tries:
        raise Exception("World {} can't be solved at Solver".format(world.id))
        #break
      #this to be changed if roll back
      if self.goal.curr_condition(temp):
        self.goal.reset_curr_condition()
        self.goal.update_curr_condition(temp)
      try:  
        output = self.policy.improve_state(temp, self.goal)
      except Exception as e:
        raise e
      temp = output[0].copy()
      yield temp, output[1], output[2], self.get_hier_move(output[3],hier)
  
  def get_hier_move(self, cond_name, world):
    #leftmost_square = world.get_leftmost_square()
    #rightmost_triangle = world.get_rightmost_triangle()
    #leftmost_triangle = world.get_leftmost_triangle()
    #rightmost_circle = world.get_rightmost_circle()
    if cond_name == 'cond_rmc_lt_lmt':
      return 'move_left_of', self.goal.move_obj.id, self.goal.rel_to_obj.id
    elif cond_name == 'cond_rmt_lt_lms':
      return 'move_left_of', self.goal.move_obj.id, self.goal.rel_to_obj.id
    elif cond_name == 'cond_rmc_lt_lms':
      return 'move_left_of', self.goal.move_obj.id, self.goal.rel_to_obj.id
    else:
      return None
  
  def solve(self, world):
    temp = world.copy()
    try:
      trajectory = list(self.next_step(temp))
    except Exception as e:
      print(e)
      trajectory = []
    finally:
      self.goal.reset_curr_condition()
    return self.create_examples(world,trajectory)
  
  def transform_trajectories(self, trjs):
    if len(trjs) == 0:
      return []
    elif len(trjs) == 1:
      return trjs
    else:
      out = []
      ci = 0
      cv = trjs[ci]
      for i in range(len(trjs)-1):
        if trjs[i+1].same_hier_action(cv):
          continue
        else:
          out += [Example(cv.init_state,cv.action,trjs[i].next_state,cv.hier)]
          ci = i+1
          cv = trjs[ci]
      out += [Example(cv.init_state,cv.action,trjs[i+1].next_state,cv.hier)]
      for e in out:
        if e.init_state.id == 179:
          draw_example(e)
      return out
  
  def create_examples(self,world,trajectory):
    output = []
    last_state = world
    for state_action in trajectory:
      action = Action(state_action[1],state_action[2])
      output += [Example(last_state,action,state_action[0],state_action[3])]
      last_state = output[-1].next_state
    return self.transform_trajectories(output)
