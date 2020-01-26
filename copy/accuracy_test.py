import pickle
import numpy as np
from pprint import pprint
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.stats import multivariate_normal, norm

ROUND = 100
RANGE = 5

def compute_loglikelihood_univar(features, means, stds):
  return np.mean(np.nan_to_num(norm.logpdf(features, loc=means, scale=stds)))

def compute_loglikelihood_multivar(features, means, covs):
  ll = []
  for feat, mean, cov in zip(features, means, covs):
    ll += [multivariate_normal.logpdf(feat, mean=mean, cov=cov, allow_singular=True)]
  return np.mean(np.array(ll))
  
def accuracy_test(i, file_name, test_features, twoD):
  if i == 0:
    return accuracy_test_0(file_name, test_features, twoD)
  elif i == 1:
    return accuracy_test_1(file_name, test_features, twoD)
  elif i == 2:
    return  accuracy_test_2(file_name, test_features, twoD)
  else:
    print("Error in accuracy test in accuracy_test")
    return None

def accuracy_test_0(file_name, test_features, twoD):
  with open(file_name, 'rb') as f:
    results = pickle.load(f, encoding="bytes")
  if twoD:
    mXs,stdXs = results[0]
    mYs,stdYs = results[1]
  else:  
    mXs,stdXs = results
  if twoD:
    rmsx = sqrt(mse(test_features[0], mXs))/RANGE
    rmsy = sqrt(mse(test_features[1], mYs))/RANGE
    r2x = r2_score(test_features[0], mXs)
    r2y = r2_score(test_features[1], mYs)
    llx = compute_loglikelihood_multivar(test_features[0], mXs, stdXs)
    lly = compute_loglikelihood_multivar(test_features[1], mYs, stdYs)
    print(file_name," Step 1")
    print("NRMSE for X: {}".format(rmsx))
    print("NRMSE for Y: {}".format(rmsy))
    print("R2 score for X: {}".format(r2x))
    print("R2 score for Y: {}".format(r2y))
    print("Log-likelihood for X: {}".format(llx))
    print("Log-likelihood for Y: {}".format(lly))
    return (rmsx,r2x,llx,rmsy,r2y,lly)
  else:
    rms = sqrt(mse(test_features, mXs))/RANGE
    r2 = r2_score(test_features, mXs)
    ll = compute_loglikelihood_multivar(test_features, mXs, stdXs)
    print(file_name," Step 1")
    print("NRMSE for X: {}".format(rms))
    print("R2 score for X: {}".format(r2))
    print("Log-likelihood for X: {}".format(ll))
    return (rms,r2,ll)

  
def accuracy_test_1(file_name, test_features, twoD):
  with open(file_name, 'rb') as f:
    results = pickle.load(f, encoding="bytes")
  mDs,stdDs = results
  if twoD:
    rmsdx = sqrt(mse(test_features[0], mDXs))/RANGE
    rmsdy = sqrt(mse(test_features[1], mDYs))/RANGE
    rmsx = sqrt(mse(test_features[2], mXs))/RANGE
    rmsy = sqrt(mse(test_features[3], mYs))/RANGE
    r2dx = r2_score(test_features[0], mDXs)
    r2dy = r2_score(test_features[1], mDYs)
    r2x = r2_score(test_features[2], mXs)
    r2y = r2_score(test_features[3], mYs)
    lldx = compute_loglikelihood_multivar(test_features[0], mDXs, stdDXs)
    lldy = compute_loglikelihood_multivar(test_features[1], mDYs, stdDYs)
    llx = compute_loglikelihood_multivar(test_features[2], mXs, stdXs)
    lly = compute_loglikelihood_multivar(test_features[3], mYs, stdYs)
    print(file_name," Step 2")
    print("NRMSE for displX: {}".format(rmsdx))
    print("NRMSE for displY: {}".format(rmsdy))
    print("NRMSE for X: {}".format(rmsx))
    print("NRMSE for Y: {}".format(rmsy))
    print("R2 score for displX: {}".format(r2dx))
    print("R2 score for displY: {}".format(r2dy))
    print("R2 score for X: {}".format(r2x))
    print("R2 score for Y: {}".format(r2y))
    print("Log-likelihood for displX: {}".format(lldx))
    print("Log-likelihood for displY: {}".format(lldy))
    print("Log-likelihood for X: {}".format(llx))
    print("Log-likelihood for Y: {}".format(lly))
    return (rmsdx,rmsx,r2dx,r2x,lldx,llx,rmsdy,rmsy,r2dy,r2y,lldy,lly)
  else:
    rmsd = sqrt(mse(test_features[0], mDs))/RANGE
    rmsx = sqrt(mse(test_features[1], mXs))/RANGE
    r2d = r2_score(test_features[0], mDs)
    r2x = r2_score(test_features[1], mXs)
    lld = compute_loglikelihood_multivar(test_features[0], mDs, stdDs)
    llx = compute_loglikelihood_multivar(test_features[1], mXs, stdXs)
    print(file_name," Step 2")
    print("NRMSE for displX: {}".format(rmsd))
    print("NRMSE for X: {}".format(rmsx))
    print("R2 score for displX: {}".format(r2d))
    print("R2 score for X: {}".format(r2x))
    print("Log-likelihood for displX: {}".format(lld))
    print("Log-likelihood for X: {}".format(llx))
    return (rmsd,rmsx,r2d,r2x,lld,llx)
  
def accuracy_test_2(file_name, test_features, twoD):
  with open(file_name, 'rb') as f:
    results = pickle.load(f, encoding="bytes")
  if twoD:
    mDXs,stdDXs = results[0]
    mDYs,stdDYs = results[1]
  else:
    mDs,stdDs = results
  if twoD:
    rmsx = sqrt(mse(test_features[0], mDXs))/RANGE
    rmsy = sqrt(mse(test_features[1], mDYs))/RANGE
    r2x = r2_score(test_features[0], mDXs)
    r2y = r2_score(test_features[1], mDYs)
    llx = compute_loglikelihood_multivar(test_features[0], mDXs, stdDXs)
    lly = compute_loglikelihood_multivar(test_features[1], mDYs, stdDYs)
    print(file_name, "Step 3")
    print("NRMSE for displX: {}".format(rmsx))
    print("NRMSE for displY: {}".format(rmsy))
    print("R2 score for displX: {}".format(r2x))
    print("R2 score for displY: {}".format(r2y))
    print("Log-likelihood for displX: {}".format(llx))
    print("Log-likelihood for displY: {}".format(lly))
    return (rmsx,r2x,llx,rmsy,r2y,lly)
  else:
    rms = sqrt(mse(test_features, mDs))/RANGE
    r2 = r2_score(test_features, mDs)
    ll = compute_loglikelihood_multivar(test_features, mDs, stdDs)
    print(file_name, "Step 3")
    print("NRMSE for displX: {}".format(rms))
    print("R2 score for displX: {}".format(r2))
    print("Log-likelihood for displX: {}".format(ll))
    return (rms,r2,ll)
  
def overall_accuracy_test(acc_res, twoD):
  st1 = []
  st2 = []
  st3 = []
  for acc in acc_res:
    st1 += [acc[0]]
    st2 += [acc[1]]
    st3 += [acc[2]]
  accuracy_step_1(st1, twoD)
  accuracy_step_2(st2, twoD)
  accuracy_step_3(st3, twoD)

def accuracy_step_1(res, twoD):
  if twoD:
    rmsx,r2x,llx,rmsy,r2y,lly = [],[],[],[],[],[]
    for r in res:
      rmsx += [r[0]]
      r2x  += [r[1]]
      llx  += [r[2]]
      rmsy += [r[3]]
      r2y  += [r[4]]
      lly  += [r[5]]
    print("###################################################")
    print("   Predicting state_t1 given state_t0 and action   ")
    print("###################################################")
    mrmsx, stdrmsx = np.mean(rmsx), np.std(rmsx) 
    mr2x, stdr2x = np.mean(r2x), np.std(r2x)
    mllx, stdllx = np.mean(llx), np.std(llx)
    mrmsy, stdrmsy = np.mean(rmsy), np.std(rmsy) 
    mr2y, stdr2y = np.mean(r2y), np.std(r2y)
    mlly, stdlly = np.mean(lly), np.std(lly)
    print("NRMSE for posX_t1 : {} ".format(round(mrmsx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsx,ROUND)))
    print("NRMSE for posY_t1 : {} ".format(round(mrmsy,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsy,ROUND)))
    print("R2 for posX_t1 : {} ".format(round(mr2x,ROUND)) + u"\u00B1"+" {}".format(round(stdr2x,ROUND)))
    print("R2 for posY_t1 : {} ".format(round(mr2y,ROUND)) + u"\u00B1"+" {}".format(round(stdr2y,ROUND)))
    print("Log-likelihood for posX_t1 : {} ".format(round(mllx,ROUND)) + u"\u00B1"+" {}".format(round(stdllx,ROUND)))
    print("Log-likelihood for posY_t1 : {} ".format(round(mlly,ROUND)) + u"\u00B1"+" {}".format(round(stdlly,ROUND)))
  else:
    rms,r2,ll = [],[],[]
    for r in res:
      rms += [r[0]]
      r2  += [r[1]]
      ll  += [r[2]]
    print("###################################################")
    print("   Predicting state_t1 given state_t0 and action   ")
    print("###################################################")
    mrms, stdrms = np.mean(rms), np.std(rms) 
    mr2, stdr2 = np.mean(r2), np.std(r2)
    mll, stdll = np.mean(ll), np.std(ll)
    print("NRMSE for posX_t1 : {} ".format(round(mrms,ROUND)) + u"\u00B1"+" {}".format(round(stdrms,ROUND)))
    print("R2 for posX_t1 : {} ".format(round(mr2,ROUND)) + u"\u00B1"+" {}".format(round(stdr2,ROUND)))
    print("Log-likelihood for posX_t1 : {} ".format(round(mll,ROUND)) + u"\u00B1"+" {}".format(round(stdll,ROUND)))

def accuracy_step_2(res, twoD):
  if twoD:
    rmsdx,rmsx,r2dx,r2x,lldx,llx,rmsdy,rmsy,r2dy,r2y,lldy,lly = [],[],[],[],[],[],[],[],[],[],[],[]
    for r in res:
      rmsdx += [r[0]]
      rmsx  += [r[1]]
      r2dx  += [r[2]]
      r2x   += [r[3]]
      lldx  += [r[4]]
      llx   += [r[5]]
      rmsdy += [r[6]]
      rmsy  += [r[7]]
      r2dy  += [r[8]]
      r2y   += [r[9]]
      lldy  += [r[10]]
      lly   += [r[11]]
    print("###################################################")
    print("       Predicting state_t1 given state_t0          ")
    print("###################################################")
    mrmsdx, stdrmsdx = np.mean(rmsdx), np.std(rmsdx) 
    mrmsx, stdrmsx = np.mean(rmsx), np.std(rmsx)
    mr2dx, stdr2dx = np.mean(r2dx), np.std(r2dx) 
    mr2x, stdr2x = np.mean(r2x), np.std(r2x)
    mlldx, stdlldx = np.mean(lldx), np.std(lldx)
    mllx, stdllx = np.mean(llx), np.std(llx)
    mrmsdy, stdrmsdy = np.mean(rmsdy), np.std(rmsdy) 
    mrmsy, stdrmsy = np.mean(rmsy), np.std(rmsy)
    mr2dy, stdr2dy = np.mean(r2dy), np.std(r2dy) 
    mr2y, stdr2y = np.mean(r2y), np.std(r2y)
    mlldy, stdlldy = np.mean(lldy), np.std(lldy)
    mlly, stdlly = np.mean(lly), np.std(lly)
    print("NRMSE for displX : {} ".format(round(mrmsdx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsdx,ROUND)))
    print("NRMSE for displY : {} ".format(round(mrmsdy,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsdy,ROUND)))
    print("NRMSE for posX_t1 : {} ".format(round(mrmsx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsx,ROUND)))
    print("NRMSE for posY_t1 : {} ".format(round(mrmsy,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsy,ROUND)))
    print("R2 for displX : {} ".format(round(mr2dx,ROUND)) + u"\u00B1"+" {}".format(round(stdr2dx,ROUND)))
    print("R2 for displY : {} ".format(round(mr2dy,ROUND)) + u"\u00B1"+" {}".format(round(stdr2dy,ROUND)))
    print("R2 for posX_t1 : {} ".format(round(mr2x,ROUND)) + u"\u00B1"+" {}".format(round(stdr2x,ROUND)))
    print("R2 for posY_t1 : {} ".format(round(mr2y,ROUND)) + u"\u00B1"+" {}".format(round(stdr2y,ROUND)))
    print("Log-likelihood for displX : {} ".format(round(mlldx,ROUND)) + u"\u00B1"+" {}".format(round(stdlldx,ROUND)))
    print("Log-likelihood for displY : {} ".format(round(mlldy,ROUND)) + u"\u00B1"+" {}".format(round(stdlldy,ROUND)))
    print("Log-likelihood for posX_t1 : {} ".format(round(mllx,ROUND)) + u"\u00B1"+" {}".format(round(stdllx,ROUND)))
    print("Log-likelihood for posY_t1 : {} ".format(round(mlly,ROUND)) + u"\u00B1"+" {}".format(round(stdlly,ROUND)))
  else:
    rmsd,rmsx,r2d,r2x,lld,llx = [],[],[],[],[],[]
    for r in res:
      rmsd += [r[0]]
      rmsx += [r[1]]
      r2d  += [r[2]]
      r2x  += [r[3]]
      lld  += [r[4]]
      llx  += [r[5]]
    print("###################################################")
    print("       Predicting state_t1 given state_t0          ")
    print("###################################################")
    mrmsd, stdrmsd = np.mean(rmsd), np.std(rmsd) 
    mrmsx, stdrmsx = np.mean(rmsx), np.std(rmsx)
    mr2d, stdr2d = np.mean(r2d), np.std(r2d) 
    mr2x, stdr2x = np.mean(r2x), np.std(r2x)
    mlld, stdlld = np.mean(lld), np.std(lld)
    mllx, stdllx = np.mean(llx), np.std(llx)
    print("NRMSE for displ : {} ".format(round(mrmsd,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsd,ROUND)))
    print("NRMSE for posX_t1 : {} ".format(round(mrmsx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsx,ROUND)))
    print("R2 for displ : {} ".format(round(mr2d,ROUND)) + u"\u00B1"+" {}".format(round(stdr2d,ROUND)))
    print("R2 for posX_t1 : {} ".format(round(mr2x,ROUND)) + u"\u00B1"+" {}".format(round(stdr2x,ROUND)))
    print("Log-likelihood for displ : {} ".format(round(mlld,ROUND)) + u"\u00B1"+" {}".format(round(stdlld,ROUND)))
    print("Log-likelihood for posX_t1 : {} ".format(round(mllx,ROUND)) + u"\u00B1"+" {}".format(round(stdllx,ROUND)))

def accuracy_step_3(res, twoD):
  if twoD:
    rmsx, r2x, llx, rmsx, r2x, llx = [], [], [], [], [], []
    for r in res:
      rmsx += [r[0]]
      r2x  += [r[1]]
      llx  += [r[2]]
      rmsy += [r[3]]
      r2y  += [r[4]]
      lly  += [r[5]]
    print("###################################################")
    print("   Predicting action given state_t0 and state_t1   ")
    print("###################################################")
    mrmsx, stdrmsx = np.mean(rmsx), np.std(rmsx)
    mr2x, stdr2x = np.mean(r2x), np.std(r2x)
    mllx, stdllx = np.mean(llx), np.std(llx)
    mrmsy, stdrmsy = np.mean(rmsy), np.std(rmsy)
    mr2y, stdr2y = np.mean(r2y), np.std(r2y)
    mlly, stdlly = np.mean(lly), np.std(lly)
    print("NRMSE for displX : {} ".format(round(mrmsx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsx,ROUND)))
    print("NRMSE for displY : {} ".format(round(mrmsy,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsy,ROUND)))
    print("R2 for displX : {} ".format(round(mr2x,ROUND)) + u"\u00B1"+" {}".format(round(stdr2x,ROUND)))
    print("R2 for displY : {} ".format(round(mr2y,ROUND)) + u"\u00B1"+" {}".format(round(stdr2y,ROUND)))
    print("Log-likelihood for displX : {} ".format(round(mllx,ROUND)) + u"\u00B1"+" {}".format(round(stdllx,ROUND)))
    print("Log-likelihood for displY : {} ".format(round(mlly,ROUND)) + u"\u00B1"+" {}".format(round(stdlly,ROUND)))
  else:
    rms, r2, ll = [], [], []
    for r in res:
      rms += [r[0]]
      r2  += [r[1]]
      ll  += [r[2]]
    print("###################################################")
    print("   Predicting action given state_t0 and state_t1   ")
    print("###################################################")
    mrms, stdrms = np.mean(rms), np.std(rms)
    mr2, stdr2 = np.mean(r2), np.std(r2)
    mll, stdll = np.mean(ll), np.std(ll)
    print("NRMSE for displ : {} ".format(round(mrms,ROUND)) + u"\u00B1"+" {}".format(round(stdrms,ROUND)))
    print("R2 for displ : {} ".format(round(mr2,ROUND)) + u"\u00B1"+" {}".format(round(stdr2,ROUND)))
    print("Log-likelihood for displ : {} ".format(round(mll,ROUND)) + u"\u00B1"+" {}".format(round(stdll,ROUND)))
