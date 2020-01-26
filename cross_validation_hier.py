import os
import pickle
import sys
import random
import numpy as np
from write_dc_input_hier import write_dc_input_det, write_dc_input_noisy
from learn_rules import learn_rules_det, learn_rules_noisy, sample_values
from stats import find_statistics
from create_input_program_hier import get_evidence_features, get_test_features, get_evidence_features_1
from accuracy_test import accuracy_test, overall_accuracy_test
from DCProgram import DCProgram
sys.path.insert(1,'data_gen/')

from Example import Example
from utils import draw_example

bin_size = 500
num_bins = 10

noise = False
noise_type = 'pos'
noise_val = 0.1

twoD = False
num_obj = 3

TEST = True
DEMO = False

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
    
def load_examples(constrained):
  #with open('data/3_obj_simple/balanced_1002','rb') as f:
  #  examples = pickle.load(f)
  with open('data/{}/{}/examples.pkl'.format(dim(twoD),scenario(constrained)),'rb') as f:
    examples2 = pickle.load(f)
  return examples2# + examples2
  
def create_bins(examples,bin_size,num_bins):  
  bins = []
  idx1 = 0
  for idx2 in range(bin_size,bin_size*num_bins+1,bin_size):
    bins += [examples[idx1:idx2]]
    idx1 = idx2
  return bins
  
def generate_bins(examples):
  random.shuffle(examples)
  bins = create_bins(examples,bin_size,num_bins)
  for i in range(num_bins):
    yield sum(bins[:i]+bins[i+1:],[]), bins[i]
  
def cross_validation(noise,noise_type,noise_val,constrained, scen, seed, num_exmp,first_feat):
  random.seed(a=seed)
  
  examples = load_examples(constrained)
  dir_name = '../probabilistic-dc-learner/data/cross_validation/'
  in_binn_temp_det = 'in_bin_det_%s.pl'
  in_binn_temp_noisy = 'in_bin_noisy_%s.pl'
  out_binn_temp = 'out_bin_%s.pl'
  results_temp = 'results_%s_%s.pkl'
  results_dir = 'results/'
  acc_res = []
  i = 1
  times = []
  for train, test in generate_bins(examples):
    in_name_det = in_binn_temp_det % (str(i))
    in_name_noisy = in_binn_temp_noisy % (str(i))
    out_name = out_binn_temp % (str(i))
    if DEMO:
      ev = get_evidence_features_1(test,constrained,twoD)
      with open('files_for_demo/{}evidence_file'.format(scen), 'w') as f:
        f.write(str(ev))
      with open('files_for_demo/{}examples_file'.format(scen),'wb') as f:
        pickle.dump(test,f,protocol=2)
      return
    if not TEST:
      if noise:
        write_dc_input_noisy(train,constrained,twoD, dir_name+scen + in_name_det, dir_name+scen + in_name_noisy, noise_type, noise_val)
        learn_rules_noisy(in_name_det,in_name_noisy,out_name,scen,  num_exmp,first_feat)
      else:
        write_dc_input_det(train,constrained,twoD, dir_name+scen + in_name_det)
        tmp = learn_rules_det(in_name_det,out_name,scen, num_exmp,first_feat)
        times += [tmp]
        continue
        #print(tmp)
        #return
      program = DCProgram(out_name,scen)
    #return
    acc_temp = []
    #for j in range(3): #the 3 predictions
    j=1
    if not TEST:
      with open(dir_name+scen+'test.pl', 'w') as f:
        f.write(program.get_program(j))
      evidence = get_evidence_features(j,test,constrained,twoD)
      #print(j,i,evidence)
      #break
      sample_values(scen,j,i,evidence,twoD)
    test_features = get_test_features(j,test,twoD)
    acc_temp += [accuracy_test(j,results_dir+scen + results_temp % (str(j),str(i)), test_features,test,twoD,constrained)]
    #return
    #return
    i += 1
    acc_res += [acc_temp]
    #if 'simple' in scen and 'noise' in scen and '2D' in scen and i > 7:
    #  break
    #if 'constrained' in scen and 'noise' in scen and '2D' in scen and i > 4:
    #  break
  #with open('results/{}times.pkl'.format(scen),'wb') as f:
  #  pickle.dump(times, f)
  with open('results/{}results.pkl'.format(scen),'wb') as f:
    pickle.dump(acc_res, f)
  
if __name__ == '__main__':
  seed = 25
  NUM_BIN = 10
  BIN_SIZE = 500
  NUM_OBJ = 3
  NUM_EXMP = (NUM_BIN - 1)*BIN_SIZE
  formula = int(NUM_EXMP*NUM_OBJ/NUM_OBJ**NUM_OBJ)
  
  for constrained in [False,True]:
    if noise:
      for noise_val in [0.05,0.1,0.2,0.5,0.7]:
        if twoD:
          if constrained:
            num_exmp = 10
            first_feat = 'heavy'
          else:
            num_exmp = formula
            first_feat = ''
          if noise:
            scen = '2D_hier_noise_{}/{}/'.format(str(noise_val),scenario(constrained))
          else:
            scen = '2D_hier/{}/'.format(scenario(constrained))
        else:
          num_exmp = formula
          first_feat = ''
          if noise:
             scen = 'hier_noise_{}/{}/'.format(str(noise_val),scenario(constrained))
          else:
            scen = 'hier/{}/'.format(scenario(constrained))
        if not TEST:
          if not os.path.isfile('results/{}results.pkl'.format(scen)):
            cross_validation(noise,noise_type,noise_val,constrained,scen,seed,num_exmp,first_feat)
        else:
          cross_validation(noise,noise_type,noise_val,constrained,scen,seed,num_exmp,first_feat)
        with open('results/{}results.pkl'.format(scen),'rb') as f:
          acc_res = pickle.load(f)
        overall_accuracy_test(acc_res,twoD)
    else:
      if twoD:
        if constrained:
          num_exmp = 10
          first_feat = 'heavy'
        else:
          num_exmp = formula
          first_feat = ''
        if noise:
          scen = '2D_hier_noise_{}/{}/'.format(str(noise_val),scenario(constrained))
        else:
          scen = '2D_hier/{}/'.format(scenario(constrained))
      else:
        num_exmp = formula
        first_feat = ''
        if noise:
           scen = 'hier_noise_{}/{}/'.format(str(noise_val),scenario(constrained))
        else:
          scen = 'hier/{}/'.format(scenario(constrained))
      if not TEST:
        if not os.path.isfile('results/{}results.pkl'.format(scen)):
          cross_validation(noise,noise_type,noise_val,constrained,scen,seed,num_exmp,first_feat)
      else:
        cross_validation(noise,noise_type,noise_val,constrained,scen,seed,num_exmp,first_feat)
      with open('results/{}results.pkl'.format(scen),'rb') as f:
        acc_res = pickle.load(f)
      overall_accuracy_test(acc_res,twoD)
