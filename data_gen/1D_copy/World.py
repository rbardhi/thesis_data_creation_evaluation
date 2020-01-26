import random
from random import uniform
from Obj import Obj

num_objects_choise = [3,4,5]
displ = 0.75


def get_key():
  return lambda obj: obj.x



class World:
  def __init__(self,id,balanced=True,constrained=False,objects = None):
    self.id = id
    self.balanced = balanced
    self.constrained = constrained
    self.num_objects = self.choose_num_objects()
    if objects == None:
      self.objects = self.get_objects(constrained=constrained)
    else:
      self.objects = objects
  def copy(self):
    output = World(self.id, balanced=self.balanced,constrained=self.constrained, objects=[obj.copy() for obj in self.objects])
    return output
  def get_objects(self,constrained=False):
    output = []
    shapes = []
    i = 0
    while len(output) < self.num_objects:
      new_obj = Obj(i,constrained=constrained)
      if new_obj.out_of_bounds():
        continue
      flag = False
      for obj in output:
        if new_obj.intersects(obj):
          flag = True
        if new_obj.inline(obj): # for simple and complex_1
          flag = True
        if self.balanced:
          if new_obj.shape in shapes:
            flag = True
        if flag == True:
          break
      if flag == False:
        output += [new_obj]
        shapes += [new_obj.shape]
        i += 1 
    return output  
  
  def get_shape_key(self,obj):
    if obj.shape == 'square':
        return 's'
    elif obj.shape == 'triangle':
      return 't'
    elif obj.shape == 'circle':
      return 'c'
    else:
      print("Error in get_shape_key at World.py. Unknown shape.")
      return None
  
  def get_relations(self):
    def lt(x,y):
      return x < y
    def gt(x,y):
      return x > y
    comp = [lt,gt]
    output = {'left':[],'right':[]}
    for i,dir in enumerate(['left','right']):
      for obj1 in self.objects:
        rels = []
        for obj2 in self.objects:
          if obj1.id == obj2.id:
            continue
          else:
            if comp[i](obj2.x,obj1.x):
              rels += [obj2.id]
        output[dir] += [rels]
    return output
    
  def get_object_orderings(self):
    srt = sorted(self.objects, key=get_key())
    key = ''
    for obj in srt:
      key += self.get_shape_key(obj)
    return key
    
  def check_intersections(self):
    for obj1 in self.objects:
      for obj2 in self.objects:
        if obj1.id == obj2.id : 
          continue
        else:
          if obj1.intersects(obj2):
            #print("Objects intersect in World {}".format(self.id))
            return True
    #print("Objects don't intersect in World {}".format(self.id))
    return False
    
  def check_out_of_bounds(self):
    for obj in self.objects:
      if obj.out_of_bounds():
        #print("Obj{} is out of bounds in World {}".format(obj.id,self.id))
        return True
    #print("All objetcs are in bounds in World {}".format(self.id))
    return False
   
  def get_object(self,id):
    for obj in self.objects:
      if obj.id == id:
        return obj
    return None 
    
  def get_leftmost_square(self):
    squares = sorted(list(filter(lambda obj: obj.shape == "square", self.objects)), key=get_key())
    return squares[0] if len(squares) > 0 else None
  def get_rightmost_triangle(self):
    triangles = sorted(list(filter(lambda obj: obj.shape == "triangle", self.objects)), key=get_key())
    return triangles[-1] if len(triangles) > 0 else None
  def get_leftmost_triangle(self):
    triangles = sorted(list(filter(lambda obj: obj.shape == "triangle", self.objects)), key=get_key())
    return triangles[0] if len(triangles) > 0 else None
  def get_rightmost_circle(self):
    circles = sorted(list(filter(lambda obj: obj.shape == "circle", self.objects)), key=get_key())
    return circles[-1] if len(circles) > 0 else None
  
  def apply_move(self,move,id):
    tot = displ + uniform(-0.25,0.25)
    amount = 0.1
    while tot > 0:
      if tot < amount:
        amount = tot
      move(self.get_object(id), amount)
      if self.check_intersections() or self.check_out_of_bounds():
        return False
      else:
        tot -= amount
    return True
  
  def choose_num_objects(self):
    return random.choice(num_objects_choise)
  
  def to_string(self):
    print("**********************************************")
    print("World {}".format(self.id))
    print("**********************************************")
    for obj in self.objects:
      obj.to_string()
