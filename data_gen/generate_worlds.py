import random
import numpy as np
from Obj import Obj
from World import World
from Goal import Goal
from Policy import Policy
from Solver import Solver
from utils import draw_example,draw_image
import pickle

import sys
from pprint import pprint

sys.path.insert(1,'../')
from stats import find_statistics

num_worlds = 100000
num_objects = 345
twoD = True

num_examples = 10000    # now we only consider unbalanced worlds
                       # since they inculde balanced worlds
balanced = False       #3: about 9000 for 4000, 
                       #4: about 15000 for 4000
                       #5: about 15000 for 4000
random.seed(a=num_objects) #3: 200, 4: 300, 5: 400
recur_limit = 10000

sys.setrecursionlimit(recur_limit)

'''
  Script to generate Examples to be input to the learner.
  Each example is a tuple of <init_state,action,next_state>.
  
  Parameters:
    num_worlds : number of total initials states generated (Here)
    recur_limit : recursion limit (Here)
    num_objects : number of objects per world (World.py)
    lowPos : low range of unfiorm distribution from which objects get position (Obj.py)
    highPos : high range of uniform distribution from which objects get position (Obj.py)
    lowSize : low range of unfiorm distribution from which objects get size (Obj.py)
    highSize : high range of uniform distribution from which objects get size (Obj.py)
    shapes : the shapes objects can take (Obj.py)
    displ : ammount that objects move when action applied (World.py)
    max_tries : maximal ammount of tries to solve a worls (Policy.py)
    condition : goal to be reached, defined as a function (Goal.py)
    apply_step : policy to improve position, defined as a function (Policy.py)
    scoring_fn : scoring function to be used as heuristic in search (Policy.py)
'''

def main(constrained):
  worlds = [World(id,balanced=balanced,constrained=constrained) for id in range(num_worlds)]
  goal = Goal()
  policy = Policy()
  solver = Solver(goal,policy)

  for world in worlds:
    assert(world.check_intersections()==False)  
    assert(world.check_out_of_bounds()==False)
    #goal.check_condition(world)
  
  print("#################TEST######################")
  unsolved = [world for world in worlds if not goal.check_filter_cond(world)]
  solved = []
  print(len(unsolved))
  #random.shuffle(unsolved)
  for world in unsolved:
    temp = solver.solve(world)
    #for e in temp:
    #  draw_example(e)
    solved += temp
    #solved += [exmp for exmp in temp if not goal.filter_examples_check(exmp)]
    if len(solved) > num_examples:
      break
  #for example in solved:
  #  draw_example(example)
  exmp_num = len(solved)
  print(exmp_num)
  
  #find_statistics(solved,num_objects)
  
  outname = 'examples.pkl'
  
  #solver.solve(worlds[5])
  with open('../data/{}/{}/'.format(dim(twoD), scenario(constrained)) + outname, 'wb') as f:
    pickle.dump(solved, f)

def scenario(constrained):
  if constrained:
    return 'constrained'
  else:
    return 'simple'  
def dim(twoD):
  if twoD:
    return '2D'
  else:
    return '1D' 
  
if __name__ == '__main__':
  for constrained in [False]:
    main(constrained)
