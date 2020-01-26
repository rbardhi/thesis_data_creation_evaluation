from extract_features import extract_features, extract_features_det, extract_features_noisy, extract_test_features, extract_test_features_for_action, extract_evidence_0, extract_evidence_1, extract_evidence_2

EXMP_LIM = 1

dir_path = 'declarative_biases/flat/'

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

def get_declarative_bias(constrained,twoD):
  file_name = '{}_flat_{}'.format(scenario(constrained),dim(twoD))
  with open(dir_path+file_name,'r') as f:
    declarative_bias = f.read()
  return declarative_bias
 
def get_declarative_bias_det(constrained,twoD):
  file_name = '{}_flat_{}_det'.format(scenario(constrained),dim(twoD))
  with open(dir_path+file_name,'r') as f:
    declarative_bias = f.read()
  return declarative_bias  
  
def get_declarative_bias_dc(constrained,twoD):
  file_name = '{}_flat_{}_dc'.format(scenario(constrained),dim(twoD))
  with open(dir_path+file_name,'r') as f:
    declarative_bias = f.read()
  return declarative_bias   
  
def get_test_features(i,examples,twoD):
  if i == 0:
    return get_test_features_0(examples,twoD)
  elif i == 1:
    return get_test_features_1(examples,twoD)
  elif i == 2:
    return get_test_features_2(examples,twoD)
  else:
    print("Error in get_test_features at create_input_program")
    return None
    
def get_test_features_0(examples,twoD):
  outX = []
  outY = []
  for example in examples:
    X, Y = extract_test_features(example)
    outX += [X]
    outY += [Y]
  if twoD:
    return outX, outY
  else:
    return outX
  
  
'''def get_test_features_1(examples,twoD):
  out = []
  for example in examples:
    out += [example.extract_displ_of_move_obj()] 
  return out''' 
def get_test_features_1(examples,twoD):
  outDX = []
  outDY = []
  outX = []
  outY = []
  for example in examples:
    X, Y = extract_test_features(example)
    dX, dY = extract_test_features_for_action(example)
    outX += X
    outY += Y
    outDX += dX
    outDY += dY
  if twoD:
    return outDX, outDY#, outX, outY 
  else:
    return outDX#, outX
  
def get_test_features_2(examples,twoD):
  displX = []
  displY = []
  for example in examples:
    dX, dY = extract_test_features_for_action(example)
    displX += [dX]
    displY += [dY]
  if twoD:
    return displX, displY
  else:
    return displX

def get_evidence_features(i,examples,constrained,twoD):
  if i == 0:
    return get_evidence_features_0(examples,constrained,twoD)
  elif i == 1:
    return get_evidence_features_1(examples,constrained,twoD)
  elif i == 2:
    return get_evidence_features_2(examples,constrained,twoD) 
  else:
    print("Error in get_evidence_features at create_input_program")
    return None
  
def get_evidence_features_0(examples,constrained,twoD):
  output = []
  temp = []
  rangeI = []
  for i, example in enumerate(examples):
    if i % EXMP_LIM == 0 and not i == 0: #there must be a fixed
      output += [[",".join(temp),(min(rangeI),max(rangeI))]]
      temp = []
      rangeI = []
    temp += extract_evidence_0(i,example,constrained,twoD)
    rangeI += [i]
  output += [[",".join(temp),(min(rangeI),max(rangeI))]]
  return output
def get_evidence_features_1(examples,constrained,twoD):
  output = []
  temp = []
  W = None
  I = None
  for i, example in enumerate(examples):
    if i % EXMP_LIM == 0 and not i == 0:
      output += [[",".join(temp),(W,I)]]
      temp = []
    temp += extract_evidence_1(i,example,constrained,twoD)
    W = i
    I = len(example.init_state.objects)
  output += [[",".join(temp),(W,I)]]
  return output
def get_evidence_features_2(examples,constrained,twoD):
  output = []
  temp = []                                     
  rangeI = []
  for i, example in enumerate(examples):
    if i % EXMP_LIM == 0 and not i == 0:
      output += [[",".join(temp),(min(rangeI),max(rangeI))]]   
      temp = []
      rangeI = []
    temp += extract_evidence_2(i,example,constrained,twoD)
    rangeI += [i]
  output += [[",".join(temp),(min(rangeI),max(rangeI))]]
  return output  
  
def create_input_program(examples,constrained,twoD):
  output = get_declarative_bias(constrained,twoD)
  for i, example in enumerate(examples):
    output += extract_features(i,example)
  return output

def create_input_program_det(examples,constrained,twoD, noise_type):
  output = get_declarative_bias_det(constrained,twoD)  
  for i, example in enumerate(examples):
    output += extract_features_det(i,example, noise_type)
  return output
  
def create_input_program_noisy(examples,constrained,twoD, noise_type, noise_val):
  output = get_declarative_bias_dc(constrained,twoD)  
  for i, example in enumerate(examples):
    output += extract_features_noisy(i,example, noise_type, noise_val)
  return output
