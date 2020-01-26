from pprint import pprint
import json

shapes_list = ['s','t','c']

def permut(list, curr, results,num_obj):
  if num_obj==len(curr):
    return results + [curr]
  else:
    for el in list:
      results = permut(list,curr + [el], results, num_obj)
    return results
    
def get_dict(num_objects):
  out = {}
  pmt = permut(shapes_list,[],[],num_objects) 
  for p in pmt:
    out[''.join(p)] = 0
  return out
  
def find_statistics(examples,num_objects):
  order_dict = get_dict(num_objects)
  balanced = 0
  move_dict = {}
  for example in examples:
    order_dict[example.init_state.get_object_orderings()] += 1
    if example.init_state.balanced:
        balanced += 1
    mk = example.get_move_key()
    if mk in move_dict:
      move_dict[mk] += 1
    else:
      move_dict[mk] = 1
  print("*******************************************************************")
  print("Orderinds of examples:",json.dumps(order_dict,indent = 2))
  #print("Moves of examples:",json.dumps(move_dict,indent = 2))
  #print('Example set has {} balanced and {} unbalanced.'.format(balanced,len(examples)-balanced))
  print("*******************************************************************")

if __name__ == '__main__':
  print(len(permut(shapes_list,[],[],4)))

    
