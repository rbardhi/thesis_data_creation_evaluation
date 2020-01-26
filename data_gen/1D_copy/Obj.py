import random
import math
import numpy as np

lowPos = 0
highPos = 5 
lowSize = 0.25
highSize = 0.7
obj_shapes = ['square', 'circle', 'triangle']

def get_random(l,h):
  return random.uniform(l,h)

class Obj:
  def __init__(self,id,constrained=False,shape=None,size=None,x=None,y=None,heavy=None):
    self.id = id
    self.constrained = constrained
    if heavy == None:
      if self.constrained:
        self.heavy = self.choose_heavy()
      else:
        self.heavy = False
    else:
      self.heavy = heavy
    if shape == None:
      self.shape = self.choose_shape()
    else:
      self.shape = shape
    if size == None:
      self.size = self.choose_size()
    else:
      self.size = size
    self.r = self.size/2
    if x == None or y == None:
      self.x, self.y = self.choose_pos()
    else:
      self.x, self.y = x, y
  def copy(self):
    output = Obj(self.id,constrained=self.constrained,shape=self.shape,size=self.size,x=self.x,y=self.y,heavy=self.heavy)
    return output
  def choose_heavy(self):
    return np.random.choice([False,True], p=[0.5, 0.5])
  def choose_shape(self):
    return random.choice(obj_shapes)
  def choose_pos(self):
    return get_random(lowPos,highPos), get_random(lowPos,highPos)
  def choose_size(self):
    return get_random(lowSize,highSize)
  def intersects(self,other):
    d = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    rr = self.r + other.r
    return d <= rr
  def out_of_bounds(self):
    return self.x - self.r < lowPos or self.x + self.r > highPos or self.y - self.r < lowPos or self.y + self.r > highPos
  def inline(self,other):
    return abs(self.y - other.y) <= self.r + other.r    
  def to_string(self):
    print("Obj{} pos({},{}) size({:.2f}) shape({})".format(self.id,self.x,self.y,self.size,self.shape))
