from World import World
from random import shuffle
from utils import opposite
from utils import draw_image
from utils import draw_child
max_depth = 19

class Node:
  ID = 0
  def set_id(self):
    self.id = Node.ID
    Node.ID += 1
  def reset_id(self):
    Node.ID = 0
  def __init__(self,world,depth):
    self.set_id()
    self.state = world.copy()
    self.parent = None
    self.action = [] #this should be list
    self.depth = depth
    self.children = []
  def insert_child(self,new_state,action,obj_id,depth):
    child = Node(new_state,depth)
    child.parent = self
    child.action = action, obj_id
    self.children += [child]
  def give_result(self, last_state, last_action):
    if self.parent == None:
      self.reset_id()
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
  #do smth here 
  '''def create_children(self, tree):
    curr_state = tree.state.copy()
    obj = curr_state.get_object(self.goal.move_obj.id)
    for action in self.actions[0]:
      lst = self.not_allowed(tree,[],obj.id)
      if action.__name__ in lst:
        continue
      curr_state = tree.state.copy()
      if curr_state.apply_move(action,obj.id):
        tree.insert_child(curr_state.copy(),action.__name__,obj.id,tree.depth+1)'''
  def create_children(self, tree):
    curr_state = tree.state.copy()
    #case where object to move is heavy
    if curr_state.get_object(self.goal.move_obj.id).heavy:
      rel_objs = [obj for obj in [self.goal.left_obj,self.goal.north_obj] if obj is not None]
      heavy = 0
      for rel_to_obj in rel_objs:
        if rel_to_obj.heavy:
          heavy += 1
          continue
        else:
          obj = curr_state.get_object(rel_to_obj.id)
          for action in self.actions[1]:
            lst = self.not_allowed(tree,[],obj.id)
            if action.__name__ in lst:
              continue
            curr_state = tree.state.copy()
            if curr_state.apply_move(action,obj.id):
              tree.insert_child(curr_state.copy(),action.__name__,obj.id,tree.depth+1)
      #case if all objects of interest are heavy
      if heavy == len(rel_objs):
        raise Exception("World {} can't be solved at SearchTree".format(curr_state.id))
    #case where object to move is not heavy
    else:
      obj = curr_state.get_object(self.goal.move_obj.id)
      for action in self.actions[0]:
        lst = self.not_allowed(tree,[],obj.id)
        if action.__name__ in lst:
          continue
        curr_state = tree.state.copy()
        if curr_state.apply_move(action,obj.id):
          tree.insert_child(curr_state.copy(),action.__name__,obj.id,tree.depth+1)
    
  def search(self, score_fn):
    curr = self.queue.pop(0)
    if self.goal.curr_condition(curr.state):
      return curr.give_result(curr.state,curr.action)
    elif curr.depth > max_depth:
      return self.search(score_fn)
    else:
      try:
        self.create_children(curr)
      except Exception as e:
        raise e 
      curr.children = sorted(curr.children, key=lambda x: score_fn(x.state))
      self.queue += curr.children
      self.queue = sorted(self.queue, key=lambda x: score_fn(x.state))
      if len(self.queue) < 1:
        raise Exception("World {} can't be solved at SearchTree, empty queue.".format(curr.state.id))
      else:
        return self.search(score_fn)
