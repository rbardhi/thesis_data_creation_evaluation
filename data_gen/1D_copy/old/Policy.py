from utils import less_then

from SearchTree import Node, SearchTree



class Policy:
  def __init__(self):
    self.apply_step = self.new_policy
    self.scoring_fn = self.new_manhattan
  def improve_state(self,world, goal):
    return self.apply_step(world, goal)
  
  def move_left(self,obj,amount):
    obj.x -= amount #+ uniform(-0.25,0.25)
  def move_right(self,obj,amount):
    obj.x += amount #+ uniform(-0.25,0.25)
  def move_up(self,obj,amount):
    obj.y += amount #+ uniform(-0.25,0.25)
  def move_down(self,obj,amount):
    obj.y -= amount #+ uniform(-0.25,0.25)
   
  def new_manhattan(self,world):
    score = 0
    for obj in world.objects:
      if obj.shape == 'square':
        score += (5 - obj.x) * 10
      elif obj.shape == 'triangle':
        score += abs(2.5 - obj.x) 
      elif obj.shape == 'circle':
        score += abs(0 - obj.x) * 10
      else:
        pass
    return score
  #Scoring functions for hier  
  def score_rmc_lt_lmt(self,base,world):
    rightmost_circle = world.get_rightmost_circle()
    leftmost_triangle = world.get_leftmost_triangle()
    normal = abs(rightmost_circle.x - leftmost_triangle.x + 1)
    penalty = 0
    for obj in base.objects:
      if obj.id == rightmost_circle.id:
        continue
      penalty += abs(obj.x - world.objects[obj.id].x)
      penalty += abs(obj.y - world.objects[obj.id].y)  
    return normal + penalty*2
    
  def score_rmt_lt_lms(self,base,world):
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_square = world.get_leftmost_square()
    normal = abs(rightmost_triangle.x - leftmost_square.x + 1)
    penalty = 0
    for obj in base.objects:
      if obj.id == rightmost_triangle.id:
        continue
      penalty += abs(obj.x - world.objects[obj.id].x)
      penalty += abs(obj.y - world.objects[obj.id].y)  
    return normal + penalty**2
    
  def score_rmc_lt_lms(self,base,world):
    rightmost_circle = world.get_rightmost_circle()
    leftmost_square = world.get_leftmost_square()
    normal = abs(rightmost_circle.x - leftmost_square.x + 1)  
    penalty = 0
    for obj in base.objects:
      if obj.id == rightmost_circle.id:
        continue
      penalty += abs(obj.x - world.objects[obj.id].x)
      penalty += abs(obj.y - world.objects[obj.id].y)  
    return normal + penalty*2
  
  def get_score_fn(self,goal,base):
    if goal.curr_condition == goal.cond_rmc_lt_lmt:
      return lambda world: self.score_rmc_lt_lmt(base,world)
    elif goal.curr_condition == goal.cond_rmt_lt_lms:
      return lambda world: self.score_rmt_lt_lms(base,world)
    elif goal.curr_condition == goal.cond_rmc_lt_lms:
      return lambda world: self.score_rmc_lt_lms(base,world)
    else:
      return None
      
  def new_policy(self, world, goal):
    moves = [self.move_left] # simple case
    root = Node(world.copy())
    root.action = goal.curr_condition.__name__
    breadthFirst = SearchTree(goal,root,moves)
    try:
      output = breadthFirst.search(self.get_score_fn(goal,world),0) #state,move,obj_id,hier
    except Exception as e:
      raise e
    return output
  
  
  '''#if improve this much better results
  def my_manhattan(self,world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    score = 0
    if not less_then(rightmost_circle, leftmost_triangle):
      score += abs(rightmost_circle.x - leftmost_triangle.x)/displ
    else:
      score -= 5
    if not less_then(rightmost_triangle, leftmost_square):
      score += abs(rightmost_triangle.x - leftmost_square.x)/displ
    else:
      score -= 5
    if not less_then(rightmost_circle, leftmost_square):
      score += abs(rightmost_circle.x - leftmost_square.x)/displ
    else:
      score -= 5
    return score
  
  
  def my_policy(self, world):
    leftmost_square = world.get_leftmost_square()
    rightmost_triangle = world.get_rightmost_triangle()
    leftmost_triangle = world.get_leftmost_triangle()
    rightmost_circle = world.get_rightmost_circle()
    
    going_left = [self.move_left, self.move_up, self.move_down]
    going_right = [self.move_right, self.move_down, self.move_up]
    
    if not less_then(rightmost_circle, leftmost_triangle):
      #move rightmost_circle or leftmost_triangle
      output = self.apply_and_check(world, going_left, rightmost_circle.id)
      if output is not None:
        return output 
      output = self.apply_and_check(world, going_right, leftmost_triangle.id)
      if output is not None:
        return output
    if not less_then(rightmost_triangle, leftmost_square):
      #move leftmost_square or rightmost_triangle
      output = self.apply_and_check(world, going_right, leftmost_square.id)
      if output is not None:
        return output
      output = self.apply_and_check(world, going_left, rightmost_triangle.id)
      if output is not None:
        return output
    if not less_then(rightmost_circle, leftmost_square):
      #move rightmost_circle or leftmost_square
      output = self.apply_and_check(world, going_left, rightmost_circle.id)
      if output is not None:
        return output
      output = self.apply_and_check(world, going_right, leftmost_square.id)
      if output is not None:
        return output
    return None

  def search(self, actions, goal, queue):
    current = queue.pop(0)
    if goal.check_condition(current.state):
      return current.give_action(None)
    else:
      queue += current.create_children
      return search(actions, goal, queue)
      #apply all the moves to all the objects
      #insert the objects to children
      #continue search on next child '''
     
    
  def apply_and_check(self, world, fns, id):
    temp = world.copy()
    for fn in fns:
        if not temp.apply_move(fn,id):
          temp = world.copy()
        else:
          return temp, fn.__name__, id
    return None
