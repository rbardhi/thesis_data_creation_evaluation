import subprocess
import os
import time

py27 = '/home/dell/anaconda3/envs/Python27/bin/python'
tree_learner = 'TreeLearnerProbabilistic.py'
test = 'test.py'

def call_python_command(python, file_name, arglist):
  
  process = subprocess.Popen([python,file_name]+arglist, stdout=subprocess.PIPE)
  
  return process.communicate() 

def learn_rules_det(in_name, out_name,scen, num_exmp,first_feat):
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/probabilistic-dc-learner/core/')
  start = time.time()
  result, error = call_python_command(py27,tree_learner,['false',in_name,out_name,scen, str(num_exmp),first_feat])
  runtime = time.time() - start
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/new_experiments_test/')
  if error is not None:
    print("Error when learning rules for {}".format(in_name))
  return runtime
    
def learn_rules_noisy(in_name_det, in_name_noisy, out_name,scen, num_exmp,first_feat):
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/probabilistic-dc-learner/core/')
  result, error = call_python_command(py27,tree_learner,['true',in_name_det, in_name_noisy, out_name,scen, str(num_exmp),first_feat])
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/new_experiments_test/')
  if error is not None:
    print("Error when learning rules for {}".format(in_name))
    
def sample_values(scen,step,i,evidence,twoD):
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/probabilistic-dc-learner/core/')
  ev_name = 'evidence'
  with open(ev_name,'w') as f:
    f.write(str(evidence))
  result, error = call_python_command(py27,test,[str(step),str(i), ev_name, str(twoD),scen])
  os.chdir('/home/dell/KU Leuven/thesis/Repositories/new_experiments_test/')
  if error is not None:
    print("Error when sampling values for {}".format(in_name))
