import pickle
from utils import draw_example
import random
import json
import math
from random import uniform
import numpy as np
#with open('../data/simple/balanced_1002','rb') as f:
#    e1 = pickle.load(f)

with open('../data/2D/constrained/examples.pkl','rb') as f:
  e2 = pickle.load(f)

e = e2#1 + e2
random.seed(a=25)
random.shuffle(e)
print(len(e))
moveX = 0.75
moveY = 0.75  
highPos = 5
lowPos = 0  
move_leftX = -(moveX)# + uniform(-0.25,0.25)
move_leftY = 0
move_rightX= moveX# + uniform(-0.25,0.25)
move_rightY= 0
move_southX = 0
move_southY = -(moveY)# + uniform(-0.25,0.25)
move_northX = 0
move_northY = moveY# + uniform(-0.25,0.25)
  
def check_intersect(o1,o2,displX,displY,x,y):
  t,f=0,0
  #for _ in range(100):
  displX = displX #+ uniform(-0.25,0.25)*x
  displY = displY #+ uniform(-0.25,0.25)*y
  d = math.sqrt(((o1.x+displX) - o2.x)**2 + ((o1.y+displY) - o2.y)**2)
  rr = o1.r + o2.r
  if d <= rr:
    return True
    #t+=1
  else:
    return False
    #f+=1
  #trate = t/(t+f)
  #frate = 1 - trate
  #print("check_intersect({},{})~finite({}:true,{}:false).".format(o1.id,o2.id,trate,frate))
  #return np.random.choice([True,False],p=[trate,frate])  
def check_out_of_bounds(o,displX,displY,x,y,dir):
  t,f=0,0
  #for _ in range(100):
  displX = displX #+ uniform(-0.25,0.25)*x
  displY = displY #+ uniform(-0.25,0.25)*y
  if dir == 'left':
    if (o.x+displX) - o.r < lowPos:
      return True
      #t+=1
    else:
      return False
      #f+=1
  elif dir == 'right':
    if (o.x+displX) + o.r > highPos:
      return True
      #t+=1
    else:
      return False
      #f+=1
  elif dir == 'north':
    if (o.y+displY) + o.r > highPos:
      return True
      #t+=1
    else:
      return False
      #f+=1
  else:
    if (o.y+displY) - o.r < lowPos:
      return True
      #t+=1
    else:
      return False
      #f+=1
  #trate = t/(t+f)
  #frate = 1 - trate
  #print("check_out_of_bounds({})~finite({}:true,{}:false).".format(o.id,trate,frate))
  #return np.random.choice([True,False],p=[trate,frate])
  
def blocked_left(example, i):
  #print('Left')
  obj =  example.init_state.get_object(i)
  if check_out_of_bounds(obj,move_leftX,move_leftY,1,0,'left'):
    return True
  rels = example.init_state.get_relations()['left'][i]
  for j in rels:
    obj2 =  example.init_state.get_object(j)
    if check_intersect(obj,obj2,move_leftX,move_leftY,1,0):
      return True
  return False
  
def blocked_right(example, i):
  #print('Right')
  obj =  example.init_state.get_object(i)
  if check_out_of_bounds(obj,move_rightX,move_rightY,1,0,'right'):
    return True
  rels = example.init_state.get_relations()['right'][i]
  for j in rels:
    obj2 =  example.init_state.get_object(j)
    if check_intersect(obj,obj2,move_rightX,move_rightY,1,0):
      return True
  return False  

def blocked_south(example, i):
  #print('South')
  obj =  example.init_state.get_object(i)
  if check_out_of_bounds(obj,move_southX,move_southY,0,1,'south'):
    return True
  rels = example.init_state.get_relations()['south'][i]
  for j in rels:
    obj2 =  example.init_state.get_object(j)
    if check_intersect(obj,obj2,move_southX,move_southY,0,1):
      return True
  return False
  
def blocked_north(example, i):
  #print('North')
  obj = example.init_state.get_object(i)
  if check_out_of_bounds(obj,move_northX,move_northY,0,1,'north'):
    return True
  rels = example.init_state.get_relations()['north'][i]
  for j in rels:
    obj2 = example.init_state.get_object(j)
    if check_intersect(obj,obj2,move_northX,move_northY,0,1):
      return True
  return False

def create_moves_dict():
  return {'move_left':0,'move_right':0,'move_up':0,'move_down':0}

def create_blocked_dict():
  out = {}
  for l in ['l','']:
    for r in ['r','']:
      for n in ['n','']:
        for s in ['s','']:
          out[l+r+n+s] = create_moves_dict()
  return out

def cond(n):
  obj = n.hier[0][1]
  blocked = ''
  if blocked_left(n,obj):
    blocked += 'l'
  if blocked_right(n,obj):
    blocked += 'r'
  if blocked_north(n,obj):
    blocked += 'n'
  if blocked_south(n,obj):
    blocked += 's'
  action = n.action.name
  if ('l' in blocked and action=='move_left') or ('r' in blocked and action=='move_right') or ('n' in blocked and action=='move_up') or ('s' in blocked and action=='move_down'):
    return True
  else:
    return False

#cases = {'move_north_of|move_left_of':create_blocked_dict(),'move_north_of':create_blocked_dict(),'move_left_of':create_blocked_dict()}
#dict = create_blocked_dict()
#bad = 0
for n in e:
  '''case = ''
  for i,hier in enumerate(n.hier):
    case += hier[0]
    if len(n.hier)>1 and i == 0:
      case += '|'
  obj = n.hier[0][1]
  blocked = ''
  if blocked_left(n,obj):
    blocked += 'l'
  if blocked_right(n,obj):
    blocked += 'r'
  if blocked_north(n,obj):
    blocked += 'n'
  if blocked_south(n,obj):
    blocked += 's'
  action = n.action.name
  
  if ('l' in blocked and action=='move_left') or ('r' in blocked and action=='move_right') or ('n' in blocked and action=='move_up') or ('s' in blocked and action=='move_down'):
    bad += 1

  cases[case][blocked][action] += 1
  dict[blocked][action] += 1'''
    #print(blocked,obj)
  draw_example(n)
  
    

'''print(json.dumps(cases, indent = 2))
print("Bad examples: ",bad)'''

#examples = [n for n in e if not cond(n)]

#print(len(e)-len(examples))
#with open('../data/2D/simple/examples.pkl','wb') as f:
#    pickle.dump(examples, f)

#tests about checking good movements
'''def filter_rels(shape, rels, example):
  out = []
  if shape == 'circle':
    for i in rels:
      if example.init_state.objects[i].shape == 'triangle' or example.init_state.objects[i].shape == 'square':
        out += [i]
    out = sorted(out, key=lambda x: example.init_state.objects[x].x)
  elif shape == 'triangle':
    for i in rels:
      if example.init_state.objects[i].shape == 'square':
        out += [i]
    out = sorted(out, key=lambda x: example.init_state.objects[x].x)
  elif shape == 'square':
    pass
  return out

def no_square(rel,exmp):
  for i in rel:
    if exmp.init_state.objects[i].shape == 'square':
      return False
  return True
  
def cond(exmp):
  rels = exmp.init_state.get_relations()['left']
  move_obj = exmp.init_state.get_object(exmp.hier[1])
  return move_obj.shape == 'triangle' and no_square(rels[move_obj.id],exmp)

e = [n for n in e if cond(n)]'''


'''example = e[0]

print("Noisy displ")
for i in example.init_state.objects:
  if i.id == example.hier[1]:
    #print(i.x,example.init_state.get_object(example.hier[2]).x)
    print(example.init_state.get_object(example.hier[2]).x-i.x-0.5)
  else:
    print(0.0)
print("True displ")
for obj1,obj2 in zip(example.next_state.objects,example.init_state.objects): 
  print(obj1.x - obj2.x)'''
