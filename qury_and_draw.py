from learn_rules import sample_values
import pickle
import numpy as np
import cv2
def color(shape):
  if shape == 'square':
    return (225,0,0)
  elif shape == 'triangle':
    return (0,225,0)
  elif shape == 'circle':
    return (0,0,225)
  else:
    return None
def text(shape):
  if shape == 'square':
    return 'S'
  elif shape == 'triangle':
    return 'T'
  elif shape == 'circle':
    return 'C'
  else:
    return None

def draw_state(world,smth=None):
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 0.5
  fontColor = (0,0,0)
  lineType = 2

  img = np.zeros((500,500,3),np.uint8)
  img.fill(255)
  i=0
  if smth == None:
    for x,y,s in world:
      img = cv2.circle(img, (int(x*100),int(500 - y*100)),int(0.25*100) ,color(s),-1)
      cv2.putText(img, "{}:".format(i)+text(s), (int(x*100),int(500 - y*100)), font, fontScale, fontColor, lineType)
      i += 1
      #img = cv2.flip(img,0)
  else:
    for x,y,s in world:
      img = cv2.circle(img, (int(x*100),int(500 - y*100)),int(0.25*100) ,color(s),-1)
      cv2.putText(img, "{}:".format(i)+text(s)+":d({})".format(round(x-smth[i],5)), (int(x*100),int(500 - y*100)), font, fontScale, fontColor, lineType)
      i += 1
      #img = cv2.flip(img,0)
  return img
  
def draw_example(init,next,smth):
  init = draw_state(init)
  div = img = np.zeros((500,2,3),np.uint8)
  next = draw_state(next,smth)
  stacked_img = np.hstack((init, div))
  stacked_img = np.hstack((stacked_img, next))
  cv2.imshow("Test", stacked_img)
  cv2.waitKey()


evidence = [['n(0)~=3,shape(0,0)~=circle,posX_t0(0,0)~=2.6543542376852973,posY_t0(0,0)~=3.9985084215823117,shape(0,1)~=square,posX_t0(0,1)~=4.0736662148181955,posY_t0(0,1)~=2.8693622992944134,shape(0,2)~=triangle,posX_t0(0,2)~=1.312203041061255,posY_t0(0,2)~=1.4232764688016935,left_of(0,0,2)~=true,left_of(0,1,0)~=true,left_of(0,1,2)~=true'+',left_of(0,0,0)~=false,left_of(0,0,1)~=false,left_of(0,1,1)~=false,left_of(0,2,0)~=false,left_of(0,2,1)~=false,left_of(0,2,2)~=false',(0,0)]]

sample_values(1,1,evidence,False)

res = 'results/results_%s_%s.pkl' % (str(1),str(1))

with open(res, 'rb') as f:
  results = pickle.load(f, encoding="bytes")
shapes = ['circle','square','triangle']
init = zip([2.6543542376852973,4.0736662148181955,1.312203041061255],[3.9985084215823117,2.8693622992944134,1.4232764688016935],shapes)
smth = [2.6543542376852973,4.0736662148181955,1.312203041061255]
mX = np.array(results[1][0])
print(mX)
next = zip(mX[0],[3.9985084215823117,2.8693622992944134,1.4232764688016935],shapes)
draw_example(init,next,smth)
print(list(zip([1,2],[3,4])))
print(list(next))
