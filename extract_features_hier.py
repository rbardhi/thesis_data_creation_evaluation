
seperator = '%*******************************************\n'
Round = 8
def extract_features(id,example):
  output = seperator + '% Example {}\n'.format(id) + seperator
  for obj in example.init_state.objects:
    output += 'shape({},{},{}).\n'.format(id,obj.id,obj.shape)
    output += 'size({},{},{}).\n'.format(id,obj.id,round(obj.r,Round))
    output += 'heavy({},{},{}).\n'.format(id,obj.id,heavy(obj.heavy))
    output += 'posX_t0({},{},{}).\n'.format(id,obj.id,round(obj.x,Round))
    output += 'posY_t0({},{},{}).\n'.format(id,obj.id,round(obj.y,Round))
  output += get_relations(id,example)
  output += example.get_displX(id)
  output += example.get_displY(id)
  output += example.get_action_hier_for_hier(id)
  for obj in example.next_state.objects:
    output += 'posX_t1({},{},{}).\n'.format(id,obj.id,round(obj.x,Round))
    output += 'posY_t1({},{},{}).\n'.format(id,obj.id,round(obj.y,Round))
  return output
 
def extract_features_det(id,example, noise_type):
  output = seperator + '% Example {}\n'.format(id) + seperator
  for obj in example.init_state.objects:
    if noise_type == 'shape':
      output += 'shape({},{},square).\n'.format(id,obj.id)
      output += 'shape({},{},triangle).\n'.format(id,obj.id)
      output += 'shape({},{},circle).\n'.format(id,obj.id)
    else:
      output += 'shape({},{},{}).\n'.format(id,obj.id,obj.shape)
    output += 'size({},{},{}).\n'.format(id,obj.id,round(obj.r,Round))
    output += 'heavy({},{},{}).\n'.format(id,obj.id,heavy(obj.heavy))
    output += 'posX_t0({},{},{}).\n'.format(id,obj.id,round(obj.x,Round))
    output += 'posY_t0({},{},{}).\n'.format(id,obj.id,round(obj.y,Round))
  output += get_relations(id,example)
  output += example.get_displX(id)
  output += example.get_displY(id)
  output += example.get_action_hier_for_hier(id)
  for obj in example.next_state.objects:
    output += 'posX_t1({},{},{}).\n'.format(id,obj.id,round(obj.x,Round))
    output += 'posY_t1({},{},{}).\n'.format(id,obj.id,round(obj.y,Round))
  return output 
 
def extract_features_noisy(id, example, noise_type, noise_val):
  output = seperator + '% Example {}\n'.format(id) + seperator
  for obj in example.init_state.objects:
    if noise_type == 'shape':
      shapes = ['square','triangle','circle']
      shapes.remove(obj.shape)
      output += 'shape({},{})~finite([{}:{},{}:{},{}:{}]) := true.\n'.format(id,obj.id,1-noise_val,obj.shape,noise_val/2,shapes.pop(0),noise_val/2,shapes.pop(0))
    else:
      output += 'shape({},{})~val({}) := true.\n'.format(id,obj.id,obj.shape)
    output += 'size({},{})~val({}).\n'.format(id,obj.id,round(obj.r,Round))
    output += 'heavy({},{})~val({}) := true.\n'.format(id,obj.id,heavy(obj.heavy))
    if noise_type == 'pos':
      output += 'posX_t0({},{}) ~ gaussian({},{}) := true.\n'.format(id,obj.id,round(obj.x,Round),noise_val)
      output += 'posY_t0({},{}) ~ gaussian({},{}) := true.\n'.format(id,obj.id,round(obj.y,Round),noise_val)
    else:
      output += 'posX_t0({},{})~val({}) := true.\n'.format(id,obj.id,round(obj.x,Round))
      output += 'posY_t0({},{})~val({}) := true.\n'.format(id,obj.id,round(obj.y,Round))
  output += get_relations_dc(id,example)
  output += example.get_displX_dc(id)
  output += example.get_displY_dc(id)
  output += example.get_action_hier_for_hier_dc(id)
  for obj in example.next_state.objects:
    output += 'posX_t1({},{})~val({}) := true.\n'.format(id,obj.id,round(obj.x,Round))
    output += 'posY_t1({},{})~val({}) := true.\n'.format(id,obj.id,round(obj.y,Round))
  return output 
 
def extract_test_features(example):
  outX = []
  outY = []
  for obj in example.next_state.objects: 
    outX += [round(obj.x,Round)]
    outY += [round(obj.y,Round)]
  return outX, outY
  
def extract_test_features_for_action(example):
  outDisplX = []
  outDisplY = []
  for obj1,obj2 in zip(example.next_state.objects,example.init_state.objects): 
    outDisplX += [round(obj1.x - obj2.x,Round)]
    outDisplY += [round(obj1.y - obj2.y,Round)]
  return outDisplX, outDisplY
  
def extract_evidence_0(id,example,constrained,twoD):
  output = ['n({})~={}'.format(id,len(example.init_state.objects))]
  for obj in example.init_state.objects:
    output += ['shape({},{})~={}'.format(id,obj.id,obj.shape)]
    output += ['posX_t0({},{})~={}'.format(id,obj.id,round(obj.x,Round))]
    if constrained:
      output += ['heavy({},{})~={}'.format(id,obj.id,heavy(obj.heavy))]
    if twoD:  
      output += ['posY_t0({},{})~={}'.format(id,obj.id,round(obj.y,Round))]
  output += get_relations_ev(id,example)
  output += example.get_displX_ev(id)
  if twoD:
    output += example.get_displY_ev(id)
  return output
def extract_evidence_1(id,example,constrained,twoD):
  output = ['n({})~={}'.format(id,len(example.init_state.objects))]
  for obj in example.init_state.objects:
    output += ['shape({},{})~={}'.format(id,obj.id,obj.shape)]
    output += ['size({},{})~={}'.format(id,obj.id,round(obj.r,Round))]
    output += ['posX_t0({},{})~={}'.format(id,obj.id,round(obj.x,Round))]
    if constrained:
      output += ['heavy({},{})~={}'.format(id,obj.id,heavy(obj.heavy))]
    if twoD:  
      output += ['posY_t0({},{})~={}'.format(id,obj.id,round(obj.y,Round))]
  output += get_relations_ev(id,example)
  return output
def extract_evidence_2(id,example,constrained,twoD):
  output = ['n({})~={}'.format(id,len(example.init_state.objects))]
  for obj in example.init_state.objects:
    output += ['shape({},{})~={}'.format(id,obj.id,obj.shape)]
    output += ['posX_t0({},{})~={}'.format(id,obj.id,round(obj.x,Round))]
    if constrained:
      output += ['heavy({},{})~={}'.format(id,obj.id,heavy(obj.heavy))]
    if twoD:  
      output += ['posY_t0({},{})~={}'.format(id,obj.id,round(obj.y,Round))]
  output += get_relations_ev(id,example)
  for obj in example.next_state.objects:
    output += ['posX_t1({},{})~={}'.format(id,obj.id,round(obj.x,Round))]
    if twoD:  
      output += ['posY_t1({},{})~={}'.format(id,obj.id,round(obj.y,Round))]
  return output
  
def get_relations(id,example):
  rels = example.init_state.get_relations()
  output = ''
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['left'][first]:
        output += 'left_of({},{},{},true).\n'.format(id,first,second)
      else:
        output += 'left_of({},{},{},false).\n'.format(id,first,second)
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['north'][first]:
        output += 'north_of({},{},{},true).\n'.format(id,first,second)
      else:
        output += 'north_of({},{},{},false).\n'.format(id,first,second)
  
  '''for i,rel in enumerate(rels['left']):
    for j in rel:
      output += 'left_of({},{},{},true).\n'.format(id,i,j)
  for i,rel in enumerate(rels['right']):
    for j in rel:
      output += 'right_of({},{},{},true).\n'.format(id,i,j)'''
  return output
 
def get_relations_dc(id,example):
  rels = example.init_state.get_relations()
  output = ''
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['left'][first]:
        output += 'left_of({},{},{})~val(true).\n'.format(id,first,second)
      else:
        output += 'left_of({},{},{})~val(false).\n'.format(id,first,second)
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['north'][first]:
        output += 'north_of({},{},{})~val(true).\n'.format(id,first,second)
      else:
        output += 'north_of({},{},{})~val(false).\n'.format(id,first,second)
  
  '''for i,rel in enumerate(rels['left']):
    for j in rel:
      output += 'left_of({},{},{})~val(true).\n'.format(id,i,j)
  for i,rel in enumerate(rels['right']):
    for j in rel:
      output += 'right_of({},{},{})~val(true).\n'.format(id,i,j)'''
  return output
  
def get_relations_ev(id,example):
  rels = example.init_state.get_relations()
  output = []
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['left'][first]:
        output += ['left_of({},{},{})~=true'.format(id,first,second)]
      else:
        output += ['left_of({},{},{})~=false'.format(id,first,second)]
  for first in range(len(example.init_state.objects)):
    for second in range(len(example.init_state.objects)):
      if second in rels['north'][first]:
        output += ['north_of({},{},{})~=true'.format(id,first,second)]
      else:
        output += ['north_of({},{},{})~=false'.format(id,first,second)]
  
  '''for i,rel in enumerate(rels['left']):
    for j in rel:
      output += ['left_of({},{},{})~=true'.format(id,i,j)]
  for i,rel in enumerate(rels['right']):
    for j in rel:
      output += ['right_of({},{},{})~=true'.format(id,i,j)]'''
  return output
  
def heavy(heavy):
  if heavy:
    return 'true'
  else:
    return 'false'  
  
'''if __name__ == '__main__':
  with open('data/simple/examples_103','rb') as f:
    examples = pickle.load(f)
  print(extract_features(0,examples[0]))
  draw_example(examples[0])'''
  
