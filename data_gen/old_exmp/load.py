import pickle
from utils import draw_example
from random import shuffle

#with open('../data/simple/balanced_1002','rb') as f:
#    e1 = pickle.load(f)

with open('../data/1D/simple/examples.pkl','rb') as f:
    e2 = pickle.load(f)

e = e2#1 + e2
shuffle(e)
print(len(e))

'''nones = [e for e in es if e.init_state.get_object(e.action.obj_id) is None]

print(len(nones))'''

#e = [i for i in e if i.init_state.id == 27]

for n in e:
  draw_example(n)


#print(len(examples))
#with open('data/examples','wb') as f:
#    pickle.dump(examples, f)


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
