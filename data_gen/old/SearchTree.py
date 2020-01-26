from World import World
from random import shuffle
from utils import opposite
from utils import draw_image

max_depth = 9900

class Node:
  def __init__(self,world):
    self.state = world.copy()
    self.parent = None
    self.action = None
    self.children = []
  def insert_child(self,new_state,action,obj_id):
    child = Node(new_state)
    child.parent = self
    child.action = action, obj_id
    self.children += [child]
  def give_result(self, last_state, last_action):
    if self.parent == None:
      return last_state, last_action[0], last_action[1], self.action
    else:
      last_state = self.state
      last_action = self.action
      return self.parent.give_result(last_state,last_action)
    
class SearchTree:
  def __init__(self, goal, tree, actions):
    self.curr = tree
    self.goal = goal
    self.queue = [self.curr]
    self.actions = actions
  
  def not_allowed(self, curr, lst, id):
    if curr.parent is None:
      return lst
    else:
      if id == curr.action[1]:
        lst += [opposite(curr.action[0])]
      return self.not_allowed(curr.parent, lst, id)
    
  def create_children(self, tree):
    curr_state = tree.state.copy()
    for obj in tree.state.objects:
      for action in self.actions:
        #print("Not allowed for Obj{}".format(obj.id))
        lst = self.not_allowed(tree,[],obj.id)
        #print(lst)
        if action.__name__ in lst:
          continue
        curr_state = tree.state.copy()
        if curr_state.apply_move(action,obj.id):
          #print(action.__name__,obj.id)
          #draw_image(curr_state)
          tree.insert_child(curr_state.copy(),action.__name__,obj.id)
    #print("end of children")
    
  def search(self, score_fn, depth):
    #print(depth)
    #input("Begin of search")
    curr = self.queue.pop(0)
    #if curr.state.id == 5 or curr.state.id == 14:
    #  return curr.state,"",None
    #curr.state.to_string()
    #print("Queue size after pop is {}".format(len(self.queue)))
    #print()
    if self.goal.curr_condition(curr.state) or depth > max_depth:
      return curr.give_result(curr.state,curr.action)
    else:
      depth += 1
      self.create_children(curr)
      curr.children = sorted(curr.children, key=lambda x: score_fn(x.state))
      #input("Children")
      #for child in curr.children:
      #  child.state.to_string()
      #  print(child.action)
      #print(score_fn(curr.state))
      self.queue += curr.children
      
      if len(self.queue) < 1:
        raise Exception("World {} can't be solved".format(curr.state.id))
      else:
        return self.search(score_fn, depth)
