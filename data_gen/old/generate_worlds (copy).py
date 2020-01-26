import random
from Obj import Obj
from World import World

random.seed(a=1)

'''lowPos = 0
highPos = 5 
lowSize = 0.25
highSize = 0.7
num_objects = 3'''
num_worlds = 15
#obj_shapes = ['square', 'circle', 'triangle']

'''def get_random(l,h):
  return random.uniform(l,h)

class Obj:
  def __init__(self,id):
    self.id = id
    self.shape = self.choose_shape()
    self.size = self.choose_size()
    self.r = self.size/2
    self.x, self.y = self.choose_pos()
  def choose_shape(self):
    return random.choice(obj_shapes)
  def choose_pos(self):
    return get_random(lowPos,highPos), get_random(lowPos,highPos)
  def choose_size(self):
    return get_random(lowSize,highSize)
  def intersects(self,obj):
    d = (self.x - obj.x)**2 + (self.y - obj.y)**2
    rr = (self.r + obj.r)**2
    return d < rr
  def out_of_bounds(self):
    return self.x - self.r < lowPos or self.x + self.r > highPos or self.y - self.r < lowPos or self.y + self.r > highPos
        
  def to_string(self):
    print("Obj{} pos({},{}) size({:.2f}) shape({})".format(self.id,self.x,self.y,self.size,self.shape))

class World:
  def __init__(self,id):
    self.id = id
    self.objects = self.get_objects()
    
  def get_objects(self):
    output = []
    i = 0
    while len(output) < num_objects:
      o = Obj(i)
      if o.out_of_bounds():
        continue
      flag = False
      for obj in output:
        if o.intersects(obj):
          flag = True
        if flag == True:
          break
      if flag == False:
        output += [o]
        i += 1 
    return output
    
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
           
  def to_string(self):
    print("**********************************************")
    print("World {}".format(self.id))
    print("**********************************************")
    [obj.to_string() for obj in self.objects]'''

def main():
  worlds = [World(id) for id in range(num_worlds)]
  [world.to_string() for world in worlds] 
  
  for world in worlds:
    assert(world.check_intersections()==False)  
    assert(world.check_out_of_bounds()==False)
    
if __name__ == '__main__':
  main()
 
 
 
 
 
'''Obj0 pos(0.423401152082421,0.8484707089719379) size(0.58) shape(square)
Obj1 pos(3.795580913582201,3.0010441506612477) size(0.35) shape(triangle)
Obj2 pos(1.7014261750099402,1.4560764370556734) size(0.42) shape(circle)
 '''
 
 
 
 
 
 
