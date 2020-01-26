import numpy as np

Round = 8
e = 1e-1
class Example:
  def __init__(self,init,action,next,hier):
    self.init_state = init.copy()
    self.action = action
    self.next_state = next.copy()
    self.hier = hier
  
  def same_hier_action(self,other):
    return self.hier[0] == other.hier[0] and self.hier[1] == other.hier[1] and self.hier[2] == other.hier[2] 
  
  def to_string(self):
    print("~~~~~~~~~~~~~~~~~~Example~~~~~~~~~~~~~~~~~~~~~~")
    self.init_state.to_string()
    self.action.to_string()
    self.next_state.to_string()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")  
  
  def format_hier(self):
    return '{}({},{})'.format(self.hier[0],self.hier[2],self.hier[1])
  
  def get_action_hier(self,id):
    return self.hier[0] + '({},{},{},true).\n'.format(id, self.hier[2], self.hier[1])
  
  def get_action_hier_for_hier(self,id):
    out = ''
    for i in range(len(self.init_state.objects)):
      for j in range(len(self.init_state.objects)):
        if i == self.hier[2] and j == self.hier[1]:
          out += self.hier[0] + '({},{},{},true).\n'.format(id, self.hier[2], self.hier[1])
        else:
          out += self.hier[0] + '({},{},{},false).\n'.format(id, i, j)
    return out
    
  def get_action_hier_for_hier_dc(self,id):
    out = ''
    for i in range(len(self.init_state.objects)):
      for j in range(len(self.init_state.objects)):
        if i == self.hier[2] and j == self.hier[1]:
          out += self.hier[0] + '({},{},{}) ~ val(true).\n'.format(id, self.hier[2], self.hier[1])
        else:
          out += self.hier[0] + '({},{},{}) ~ val(false).\n'.format(id, i, j)
    return out    
    
  def get_action_hier_dc(self,id):
    return self.hier[0] + '({},{},{})~val(true) := true.\n'.format(id, self.hier[2], self.hier[1])  
  
  def get_action_hier_ev(self,id):
    return [self.hier[0] + '({},{},{})~=true'.format(id, self.hier[2], self.hier[1])]
  
  def get_displX(self,id):
    output = ''
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].x == self.init_state.objects[i].x:
        output += 'displX({},{},{}).\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x,Round))
      else:
        output += 'displX({},{},{}).\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x + e*np.random.normal(0,1),Round))
    return output 
   
  def get_displY(self,id):
    output = ''
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].y == self.init_state.objects[i].y:
        output += 'displY({},{},{}).\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y,Round))
      else:
        output += 'displY({},{},{}).\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y + e*np.random.normal(0,1),Round))
    return output  
    
  #'displ({},{})~val({}) := true.\n'  
       
  def get_displX_dc(self,id):
    output = ''
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].x == self.init_state.objects[i].x:
        output += 'displX({},{})~val({}) := true.\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x,Round))
      else:
        output += 'displX({},{})~val({}) := true.\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x + e*np.random.normal(0,1),Round)) 
    return output
    
  def get_displY_dc(self,id):
    output = ''
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].y == self.init_state.objects[i].y:
        output += 'displY({},{})~val({}) := true.\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y,Round))
      else:
        output += 'displY({},{})~val({}) := true.\n'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y + e*np.random.normal(0,1),Round)) 
    return output
    ['displ({},{})~=({})']
  
  def get_displX_ev(self,id):
    output = []
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].x == self.init_state.objects[i].x:
        output += ['displX({},{})~=({})'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x,Round))]
      else:
        output += ['displX({},{})~=({})'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].x - self.init_state.objects[i].x + e*np.random.normal(0,1),Round))] 
    return output
    
  def get_displY_ev(self,id):
    output = []
    for i in range(len(self.init_state.objects)):
      if self.next_state.objects[i].y == self.init_state.objects[i].y:
        output += ['displY({},{})~=({})'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y))]
      else:
        output += ['displY({},{})~=({})'.format(id,self.init_state.objects[i].id, round(self.next_state.objects[i].y - self.init_state.objects[i].y + e*np.random.normal(0,1),Round))] 
    return output
    
  def get_move_key(self):
    obj = self.init_state.get_object(self.action.obj_id)
    if obj is None:
      return 'none'
    move_key = '{}|'.format(self.init_state.get_shape_key(self.init_state.get_object(self.action.obj_id)))
    rels = self.init_state.get_relations()['left'][self.action.obj_id]
    for rel in rels:
      move_key += self.init_state.get_shape_key(self.init_state.get_object(rel))
    return move_key
