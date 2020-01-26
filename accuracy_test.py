import sys
import pickle
import numpy as np
from pprint import pprint
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import auc
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

sys.path.insert(1,'data_gen/')
from utils import draw_example
from Example import Example


def less_then(a,b):
  return a < b - 0.5
def filter_less_then(a,b):
  return a < b
  
ROUND = 4
RANGE = 5

def compute_loglikelihood_univar(features, means, stds):
  return np.mean(np.nan_to_num(norm.logpdf(features, loc=means, scale=stds)))

def compute_loglikelihood_multivar(features, means, covs):
  ll = []
  for feat, mean, cov in zip(features, means, covs):
    ll += [multivariate_normal.logpdf(feat, mean=mean, cov=cov, allow_singular=True)]
  return np.mean(np.array(ll))
  
def accuracy_test(i, file_name, test_features,examples, twoD,constrained):
  if i == 0:
    return accuracy_test_0(file_name, test_features, twoD)
  elif i == 1:
    return accuracy_test_1(file_name, test_features, examples, twoD,constrained)
  elif i == 2:
    return  accuracy_test_2(file_name, test_features, twoD)
  else:
    print("Error in accuracy test in accuracy_test")
    return None

'''def accuracy_test_0(file_name, test_features, twoD):
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
    return (rms,r2,ll)'''

  
def accuracy_test_1(file_name, test_features, examples, twoD,constrained):
  with open(file_name, 'rb') as f:
    results = pickle.load(f, encoding="bytes")
  if twoD:
    mDXs, stdDXs = results[0]
    mDYs, stdDYs = results[1]
    rmsdx = sqrt(mse(test_features[0], mDXs))/RANGE
    r2dx = r2_score(test_features[0], mDXs)
    lldx = compute_loglikelihood_univar(test_features[0], mDXs, stdDXs)
    rmsdy = sqrt(mse(test_features[1], mDYs))/RANGE
    r2dy = r2_score(test_features[1], mDYs)
    lldy = compute_loglikelihood_univar(test_features[1], mDYs, stdDYs)
    print(file_name," Step 2")
    print("NRMSE for displX: {}".format(rmsdx))
    print("R2 score for displX: {}".format(r2dx))
    print("Log-likelihood for displX: {}".format(lldx))
    print(file_name," Step 2")
    print("NRMSE for displY: {}".format(rmsdy))
    print("R2 score for displY: {}".format(r2dy))
    print("Log-likelihood for displY: {}".format(lldy))
    roc_auc_mx, pr_auc_mx, roc_auc_my, pr_auc_my, perc_gmx, perc_gmy, perc_gzx, perc_gzy, roc_auc_mxc, pr_auc_mxc, roc_auc_myc, pr_auc_myc, perc_gmxc, perc_gmyc, perc_gzxc, perc_gzyc, roc_auc_mxt, pr_auc_mxt, roc_auc_myt, pr_auc_myt, perc_gmxt, perc_gmyt, perc_gzxt, perc_gzyt, roc_auc_mxs, pr_auc_mxs, roc_auc_mys, pr_auc_mys, perc_gmxs, perc_gmys, perc_gzxs, perc_gzys, perc_gmxl, perc_gmxr, perc_gmyu, perc_gmyd, perc_gmxlc, perc_gmxrc, perc_gmyuc, perc_gmydc, perc_gmxlt, perc_gmxrt, perc_gmyut, perc_gmydt, perc_gmxls, perc_gmxrs, perc_gmyus, perc_gmyds, perc_interx, perc_intery, perc_interxl, perc_interxr, perc_interxlc, perc_interxlt, perc_interxls, perc_interxrc, perc_interxrt, perc_interxrs, perc_interyu, perc_interyd, perc_interyuc, perc_interyut, perc_interyus, perc_interydc, perc_interydt, perc_interyds = the_new_tests_2D(mDXs,mDYs,examples,constrained)
    return (rmsdx, r2dx, lldx, rmsdy, r2dy, lldy, roc_auc_mx, pr_auc_mx, roc_auc_my, pr_auc_my, perc_gmx, perc_gmy, perc_gzx, perc_gzy, roc_auc_mxc, pr_auc_mxc, roc_auc_myc, pr_auc_myc, perc_gmxc, perc_gmyc, perc_gzxc, perc_gzyc, roc_auc_mxt, pr_auc_mxt, roc_auc_myt, pr_auc_myt, perc_gmxt, perc_gmyt, perc_gzxt, perc_gzyt, roc_auc_mxs, pr_auc_mxs, roc_auc_mys, pr_auc_mys, perc_gmxs, perc_gmys, perc_gzxs, perc_gzys, perc_gmxl, perc_gmxr, perc_gmyu, perc_gmyd, perc_gmxlc, perc_gmxrc, perc_gmyuc, perc_gmydc, perc_gmxlt, perc_gmxrt, perc_gmyut, perc_gmydt, perc_gmxls, perc_gmxrs, perc_gmyus, perc_gmyds, perc_interx, perc_intery, perc_interxl, perc_interxr, perc_interxlc, perc_interxlt, perc_interxls, perc_interxrc, perc_interxrt, perc_interxrs, perc_interyu, perc_interyd, perc_interyuc, perc_interyut, perc_interyus, perc_interydc, perc_interydt, perc_interyds)
  else:
    mDs,stdDs = results
    rmsd = sqrt(mse(test_features, mDs))/RANGE
    r2d = r2_score(test_features, mDs)
    lld = compute_loglikelihood_univar(test_features, mDs, stdDs)
    print(file_name," Step 2")
    print("NRMSE for displX: {}".format(rmsd))
    print("R2 score for displX: {}".format(r2d))
    print("Log-likelihood for displX: {}".format(lld))
    
    good_moves, subgoal, rmseg, avg_sub_dist, constr, rmsed, avg_constr_dist, gmc,gmt,gms,sgc,sgt,sgs,cc,ct,cs,rmsez,good_zeros,gzc,gzt,gzs,g_auc,g_aucc,g_auct,g_aucs,g_pr,g_prc,g_prt,g_prs = the_new_tests(mDs, examples, constrained)
    
    return (rmsd,r2d,lld,good_moves, subgoal, rmseg, avg_sub_dist, constr, rmsed, avg_constr_dist, gmc,gmt,gms,sgc,sgt,sgs,cc,ct,cs,rmsez,good_zeros,gzc,gzt,gzs,g_auc,g_aucc,g_auct,g_aucs,g_pr,g_prc,g_prt,g_prs)
  
'''def accuracy_test_2(file_name, test_features, twoD):
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
    return (rms,r2,ll)'''
  
def overall_accuracy_test(acc_res, twoD):
  #st1 = []
  #st2 = []
  #st3 = []
  #for acc in acc_res:
  #  st1 += [acc[0]]
  #  st2 += [acc[1]]
  #  st3 += [acc[2]]
  if twoD:
    accuracy_step_2D(acc_res)
  else:
    res = accuracy_step(acc_res)
    return res
  #accuracy_step_1(st1, twoD)
  #accuracy_step_2(st2, twoD)
  #accuracy_step_3(st3, twoD)

def accuracy_step_2D(res):
  rmsdx, r2dx, lldx, rmsdy, r2dy, lldy, roc_auc_mx, pr_auc_mx, roc_auc_my, pr_auc_my, perc_gmx, perc_gmy, perc_gzx, perc_gzy, roc_auc_mxc, pr_auc_mxc, roc_auc_myc, pr_auc_myc, perc_gmxc, perc_gmyc, perc_gzxc, perc_gzyc, roc_auc_mxt, pr_auc_mxt, roc_auc_myt, pr_auc_myt, perc_gmxt, perc_gmyt, perc_gzxt, perc_gzyt, roc_auc_mxs, pr_auc_mxs, roc_auc_mys, pr_auc_mys, perc_gmxs, perc_gmys, perc_gzxs, perc_gzys, perc_gmxl, perc_gmxr, perc_gmyu, perc_gmyd, perc_gmxlc, perc_gmxrc, perc_gmyuc, perc_gmydc, perc_gmxlt, perc_gmxrt, perc_gmyut, perc_gmydt, perc_gmxls, perc_gmxrs, perc_gmyus, perc_gmyds, perc_interx, perc_intery, perc_interxl, perc_interxr, perc_interxlc, perc_interxlt, perc_interxls, perc_interxrc, perc_interxrt, perc_interxrs, perc_interyu, perc_interyd, perc_interyuc, perc_interyut, perc_interyus, perc_interydc, perc_interydt, perc_interyds = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
  for r in res:
    r = r[0]
    rmsdx += [r[0]]
    r2dx += [r[1]]
    lldx += [r[2]]
    rmsdy += [r[3]]
    r2dy += [r[4]]
    lldy += [r[5]]
    roc_auc_mx += [r[6]]
    pr_auc_mx += [r[7]]
    roc_auc_my += [r[8]]
    pr_auc_my += [r[9]]
    perc_gmx += [r[10]]
    perc_gmy += [r[11]]
    perc_gzx += [r[12]]
    perc_gzy += [r[13]]
    roc_auc_mxc += [r[14]]
    pr_auc_mxc += [r[15]]
    roc_auc_myc += [r[16]]
    pr_auc_myc += [r[17]]
    perc_gmxc += [r[18]]
    perc_gmyc += [r[19]]
    perc_gzxc += [r[20]]
    perc_gzyc += [r[21]]
    roc_auc_mxt += [r[22]]
    pr_auc_mxt += [r[23]]
    roc_auc_myt += [r[24]]
    pr_auc_myt += [r[25]]
    perc_gmxt += [r[26]]
    perc_gmyt += [r[27]]
    perc_gzxt += [r[28]]
    perc_gzyt += [r[29]]
    roc_auc_mxs += [r[30]]
    pr_auc_mxs += [r[31]]
    roc_auc_mys += [r[32]]
    pr_auc_mys += [r[33]]
    perc_gmxs += [r[34]]
    perc_gmys += [r[35]]
    perc_gzxs += [r[36]]
    perc_gzys += [r[37]]
    perc_gmxl += [r[38]]
    perc_gmxr += [r[39]]
    perc_gmyu += [r[40]]
    perc_gmyd += [r[41]]
    perc_gmxlc += [r[42]]
    perc_gmxrc += [r[43]]
    perc_gmyuc += [r[44]]
    perc_gmydc += [r[45]]
    perc_gmxlt += [r[46]]
    perc_gmxrt += [r[47]]
    perc_gmyut += [r[48]]
    perc_gmydt += [r[49]]
    perc_gmxls += [r[50]]
    perc_gmxrs += [r[51]]
    perc_gmyus += [r[52]]
    perc_gmyds += [r[53]]
    perc_interx += [r[54]]
    perc_intery += [r[55]]
    perc_interxl += [r[56]]
    perc_interxr += [r[57]]
    perc_interxlc += [r[58]]
    perc_interxlt += [r[59]]
    perc_interxls += [r[60]]
    perc_interxrc += [r[61]]
    perc_interxrt += [r[62]]
    perc_interxrs += [r[63]]
    perc_interyu += [r[64]]
    perc_interyd += [r[65]]
    perc_interyuc += [r[66]]
    perc_interyut += [r[67]]
    perc_interyus += [r[68]]
    perc_interydc += [r[69]]
    perc_interydt += [r[70]]
    perc_interyds += [r[71]]
    
  mrmsdx, stdrmsdx = np.mean(rmsdx), np.std(rmsdx)
  mr2dx, stdr2dx = np.mean(r2dx), np.std(r2dx)
  mlldx, stdlldx = np.mean(lldx), np.std(lldx)
  mrmsdy, stdrmsdy = np.mean(rmsdy), np.std(rmsdy)
  mr2dy, stdr2dy = np.mean(r2dy), np.std(r2dy)
  mlldy, stdlldy = np.mean(lldy), np.std(lldy)
  mroc_auc_mx, stdroc_auc_mx = np.mean(roc_auc_mx), np.std(roc_auc_mx)
  mpr_auc_mx, stdpr_auc_mx = np.mean(pr_auc_mx), np.std(pr_auc_mx)
  mroc_auc_my, stdroc_auc_my = np.mean(roc_auc_my), np.std(roc_auc_my)
  mpr_auc_my, stdpr_auc_my = np.mean(pr_auc_my), np.std(pr_auc_my)
  mperc_gmx, stdperc_gmx = np.mean(perc_gmx), np.std(perc_gmx)
  mperc_gmy, stdperc_gmy = np.mean(perc_gmy), np.std(perc_gmy)
  mperc_gzx, stdperc_gzx = np.mean(perc_gzx), np.std(perc_gzx)
  mperc_gzy, stdperc_gzy = np.mean(perc_gzy), np.std(perc_gzy)
  mroc_auc_mxc, stdroc_auc_mxc = np.mean(roc_auc_mxc), np.std(roc_auc_mxc)
  mpr_auc_mxc, stdpr_auc_mxc = np.mean(pr_auc_mxc), np.std(pr_auc_mxc)
  mroc_auc_myc, stdroc_auc_myc = np.mean(roc_auc_myc), np.std(roc_auc_myc)
  mpr_auc_myc, stdpr_auc_myc = np.mean(pr_auc_myc), np.std(pr_auc_myc)
  mperc_gmxc, stdperc_gmxc = np.mean(perc_gmxc), np.std(perc_gmxc)
  mperc_gmyc, stdperc_gmyc = np.mean(perc_gmyc), np.std(perc_gmyc)
  mperc_gzxc, stdperc_gzxc = np.mean(perc_gzxc), np.std(perc_gzxc)
  mperc_gzyc, stdperc_gzyc = np.mean(perc_gzyc), np.std(perc_gzyc)
  mroc_auc_mxt, stdroc_auc_mxt = np.mean(roc_auc_mxt), np.std(roc_auc_mxt)
  mpr_auc_mxt, stdpr_auc_mxt = np.mean(pr_auc_mxt), np.std(pr_auc_mxt)
  mroc_auc_myt, stdroc_auc_myt = np.mean(roc_auc_myt), np.std(roc_auc_myt)
  mpr_auc_myt, stdpr_auc_myt = np.mean(pr_auc_myt), np.std(pr_auc_myt)
  mperc_gmxt, stdperc_gmxt = np.mean(perc_gmxt), np.std(perc_gmxt)
  mperc_gmyt, stdperc_gmyt = np.mean(perc_gmyt), np.std(perc_gmyt)
  mperc_gzxt, stdperc_gzxt = np.mean(perc_gzxt), np.std(perc_gzxt)
  mperc_gzyt, stdperc_gzyt = np.mean(perc_gzyt), np.std(perc_gzyt)
  mroc_auc_mxs, stdroc_auc_mxs = np.mean(roc_auc_mxs), np.std(roc_auc_mxs)
  mpr_auc_mxs, stdpr_auc_mxs = np.mean(pr_auc_mxs), np.std(pr_auc_mxs)
  mroc_auc_mys, stdroc_auc_mys = np.mean(roc_auc_mys), np.std(roc_auc_mys)
  mpr_auc_mys, stdpr_auc_mys = np.mean(pr_auc_mys), np.std(pr_auc_mys)
  mperc_gmxs, stdperc_gmxs = np.mean(perc_gmxs), np.std(perc_gmxs)
  mperc_gmys, stdperc_gmys = np.mean(perc_gmys), np.std(perc_gmys)
  mperc_gzxs, stdperc_gzxs = np.mean(perc_gzxs), np.std(perc_gzxs)
  mperc_gzys, stdperc_gzys = np.mean(perc_gzys), np.std(perc_gzys)
  mperc_gmxl, stdperc_gmxl = np.mean(perc_gmxl), np.std(perc_gmxl)
  mperc_gmxr, stdperc_gmxr = np.mean(perc_gmxr), np.std(perc_gmxr)
  mperc_gmyu, stdperc_gmyu = np.mean(perc_gmyu), np.std(perc_gmyu)
  mperc_gmyd, stdperc_gmyd = np.mean(perc_gmyd), np.std(perc_gmyd)
  mperc_gmxlc, stdperc_gmxlc = np.mean(perc_gmxlc), np.std(perc_gmxlc)
  mperc_gmxrc, stdperc_gmxrc = np.mean(perc_gmxrc), np.std(perc_gmxrc)
  mperc_gmyuc, stdperc_gmyuc = np.mean(perc_gmyuc), np.std(perc_gmyuc)
  mperc_gmydc, stdperc_gmydc = np.mean(perc_gmydc), np.std(perc_gmydc)
  mperc_gmxlt, stdperc_gmxlt = np.mean(perc_gmxlt), np.std(perc_gmxlt)
  mperc_gmxrt, stdperc_gmxrt = np.mean(perc_gmxrt), np.std(perc_gmxrt)
  mperc_gmyut, stdperc_gmyut = np.mean(perc_gmyut), np.std(perc_gmyut)
  mperc_gmydt, stdperc_gmydt = np.mean(perc_gmydt), np.std(perc_gmydt)
  mperc_gmxls, stdperc_gmxls = np.mean(perc_gmxls), np.std(perc_gmxls)
  mperc_gmxrs, stdperc_gmxrs = np.mean(perc_gmxrs), np.std(perc_gmxrs)
  mperc_gmyus, stdperc_gmyus = np.mean(perc_gmyus), np.std(perc_gmyus)
  mperc_gmyds, stdperc_gmyds = np.mean(perc_gmyds), np.std(perc_gmyds)
  mperc_interx, stdperc_interx = np.mean(perc_interx), np.std(perc_interx)
  mperc_intery, stdperc_intery = np.mean(perc_intery), np.std(perc_intery)
  mperc_interxl, stdperc_interxl = np.mean(perc_interxl), np.std(perc_interxl)
  mperc_interxr, stdperc_interxr = np.mean(perc_interxr), np.std(perc_interxr)
  mperc_interxlc, stdperc_interxlc = np.mean(perc_interxlc), np.std(perc_interxlc)
  mperc_interxlt, stdperc_interxlt = np.mean(perc_interxlt), np.std(perc_interxlt)
  mperc_interxls, stdperc_interxls = np.mean(perc_interxls), np.std(perc_interxls)
  mperc_interxrc, stdperc_interxrc = np.mean(perc_interxrc), np.std(perc_interxrc)
  mperc_interxrt, stdperc_interxrt = np.mean(perc_interxrt), np.std(perc_interxrt)
  mperc_interxrs, stdperc_interxrs = np.mean(perc_interxrs), np.std(perc_interxrs)
  mperc_interyu, stdperc_interyu = np.mean(perc_interyu), np.std(perc_interyu)
  mperc_interyd, stdperc_interyd = np.mean(perc_interyd), np.std(perc_interyd)
  mperc_interyuc, stdperc_interyuc = np.mean(perc_interyuc), np.std(perc_interyuc)
  mperc_interyut, stdperc_interyut = np.mean(perc_interyut), np.std(perc_interyut)
  mperc_interyus, stdperc_interyus = np.mean(perc_interyus), np.std(perc_interyus)
  mperc_interydc, stdperc_interydc = np.mean(perc_interydc), np.std(perc_interydc)
  mperc_interydt, stdperc_interydt = np.mean(perc_interydt), np.std(perc_interydt)
  mperc_interyds, stdperc_interyds = np.mean(perc_interyds), np.std(perc_interyds)
  
  print("NRMSE for displX: {} ".format(round(mrmsdx,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsdx,ROUND)))
  print("R2 score for displX: {} ".format(round(mr2dx,ROUND)) + u"\u00B1"+" {}".format(round(stdr2dx,ROUND)))
  print("Log-likelihood for displX: {} ".format(round(mlldx,ROUND)) + u"\u00B1"+" {}".format(round(stdlldx,ROUND)))
  print("NRMSE for displY: {} ".format(round(mrmsdy,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsdy,ROUND)))
  print("R2 score for displY: {} ".format(round(mr2dy,ROUND)) + u"\u00B1"+" {}".format(round(stdr2dy,ROUND)))
  print("Log-likelihood for displY: {} ".format(round(mlldy,ROUND)) + u"\u00B1"+" {}".format(round(stdlldy,ROUND)))
  print("Percentage good move displX: {} ".format(round(mperc_gmx,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmx,ROUND)))
  print("Percentage good move displX circle: {} ".format(round(mperc_gmxc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxc,ROUND)))
  print("Percentage good move displX triangle: {} ".format(round(mperc_gmxt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxt,ROUND)))
  print("Percentage good move displX square: {} ".format(round(mperc_gmxs,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxs,ROUND)))
  print("Percentage good move displX left: {} ".format(round(mperc_gmxl,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxl,ROUND)))
  print("Percentage good move displX left circle: {} ".format(round(mperc_gmxlc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxlc,ROUND)))
  print("Percentage good move displX left triangle: {} ".format(round(mperc_gmxlt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxlt,ROUND)))
  print("Percentage good move displX left square: {} ".format(round(mperc_gmxls,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxls,ROUND)))
  print("Percentage good move displX right: {} ".format(round(mperc_gmxr,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxr,ROUND)))
  print("Percentage good move displX right circle: {} ".format(round(mperc_gmxrc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxrc,ROUND)))
  print("Percentage good move displX right triangle: {} ".format(round(mperc_gmxrt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxrt,ROUND)))
  print("Percentage good move displX right square: {} ".format(round(mperc_gmxrs,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmxrs,ROUND)))
  print("AUC ROC good move displX: {} ".format(round(mroc_auc_mx,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_mx,ROUND)))
  print("AUC PR good move displX: {} ".format(round(mpr_auc_mx,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_mx,ROUND)))
  print("AUC ROC good move displX circle: {} ".format(round(mroc_auc_mxc,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_mxc,ROUND)))
  print("AUC PR good move displX circle: {} ".format(round(mpr_auc_mxc,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_mxc,ROUND)))
  print("AUC ROC good move displX triangle: {} ".format(round(mroc_auc_mxt,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_mxt,ROUND)))
  print("AUC PR good move displX triangle: {} ".format(round(mpr_auc_mxt,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_mxt,ROUND)))
  print("AUC ROC good move displX square: {} ".format(round(mroc_auc_mxs,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_mxs,ROUND)))
  print("AUC PR good move displX square: {} ".format(round(mpr_auc_mxs,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_mxs,ROUND)))
  
  print("Percentage good move displY: {} ".format(round(mperc_gmy,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmy,ROUND)))
  print("Percentage good move displY circle: {} ".format(round(mperc_gmyc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyc,ROUND)))
  print("Percentage good move displY triangle: {} ".format(round(mperc_gmyt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyt,ROUND)))
  print("Percentage good move displY square: {} ".format(round(mperc_gmys,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmys,ROUND)))
  print("Percentage good move displY north: {} ".format(round(mperc_gmyu,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyu,ROUND)))
  print("Percentage good move displY north circle: {} ".format(round(mperc_gmyuc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyuc,ROUND)))
  print("Percentage good move displY north triangle: {} ".format(round(mperc_gmyut,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyut,ROUND)))
  print("Percentage good move displY north square: {} ".format(round(mperc_gmyus,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyus,ROUND)))
  print("Percentage good move displY south: {} ".format(round(mperc_gmyd,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyd,ROUND)))
  print("Percentage good move displY south circle: {} ".format(round(mperc_gmydc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmydc,ROUND)))
  print("Percentage good move displY south triangle: {} ".format(round(mperc_gmydt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmydt,ROUND)))
  print("Percentage good move displY south square: {} ".format(round(mperc_gmyds,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gmyds,ROUND)))
  print("AUC ROC good move displY: {} ".format(round(mroc_auc_my,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_my,ROUND)))
  print("AUC PR good move displY: {} ".format(round(mpr_auc_my,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_my,ROUND)))
  print("AUC ROC good move displY circle: {} ".format(round(mroc_auc_myc,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_myc,ROUND)))
  print("AUC PR good move displY circle: {} ".format(round(mpr_auc_myc,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_myc,ROUND)))
  print("AUC ROC good move displY triangle: {} ".format(round(mroc_auc_myt,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_myt,ROUND)))
  print("AUC PR good move displY triangle: {} ".format(round(mpr_auc_myt,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_myt,ROUND))) 
  print("AUC ROC good move displY square: {} ".format(round(mroc_auc_mys,ROUND)) + u"\u00B1"+" {}".format(round(stdroc_auc_mys,ROUND)))
  print("AUC PR good move displY square: {} ".format(round(mpr_auc_mys,ROUND)) + u"\u00B1"+" {}".format(round(stdpr_auc_mys,ROUND)))
  
  print("Percentage good zero displX: {} ".format(round(mperc_gzx,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzx,ROUND)))
  print("Percentage good zero displX circle: {} ".format(round(mperc_gzxc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzxc,ROUND)))
  print("Percentage good zero displX triangle: {} ".format(round(mperc_gzxt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzxt,ROUND)))
  print("Percentage good zero displX square: {} ".format(round(mperc_gzxs,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzxs,ROUND))) 
  
  print("Percentage good zero displY: {} ".format(round(mperc_gzy,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzy,ROUND)))
  print("Percentage good zero displY circle: {} ".format(round(mperc_gzyc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzyc,ROUND)))
  print("Percentage good zero displY triangle: {} ".format(round(mperc_gzyt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzyt,ROUND)))
  print("Percentage good zero displY square: {} ".format(round(mperc_gzys,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_gzys,ROUND))) 
  
  print("Percentage intersections X-axis: {} ".format(round(mperc_interx,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interx,ROUND)))
  print("Percentage intersections X-axis left: {} ".format(round(mperc_interxl,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxl,ROUND)))
  print("Percentage intersections X-axis left circle: {} ".format(round(mperc_interxlc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxlc,ROUND)))
  print("Percentage intersections X-axis left triangle: {} ".format(round(mperc_interxlt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxlt,ROUND)))
  print("Percentage intersections X-axis left square: {} ".format(round(mperc_interxls,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxls,ROUND)))
  print("Percentage intersections X-axis right: {} ".format(round(mperc_interxr,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxr,ROUND)))
  print("Percentage intersections X-axis right circle: {} ".format(round(mperc_interxrc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxrc,ROUND)))
  print("Percentage intersections X-axis right triangle: {} ".format(round(mperc_interxrt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxrt,ROUND)))
  print("Percentage intersections X-axis right square: {} ".format(round(mperc_interxrs,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interxrs,ROUND)))
  
  print("Percentage intersections Y-axis: {} ".format(round(mperc_intery,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_intery,ROUND)))
  print("Percentage intersections Y-axis north: {} ".format(round(mperc_interyu,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyu,ROUND)))
  print("Percentage intersections Y-axis north circle: {} ".format(round(mperc_interyuc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyuc,ROUND)))
  print("Percentage intersections Y-axis north triangle: {} ".format(round(mperc_interyut,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyut,ROUND)))
  print("Percentage intersections Y-axis north square: {} ".format(round(mperc_interyus,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyus,ROUND)))
  print("Percentage intersections Y-axis south: {} ".format(round(mperc_interyd,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyd,ROUND)))
  print("Percentage intersections Y-axis south circle: {} ".format(round(mperc_interydc,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interydc,ROUND)))
  print("Percentage intersections Y-axis south triangle: {} ".format(round(mperc_interydt,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interydt,ROUND)))
  print("Percentage intersections Y-axis south square: {} ".format(round(mperc_interyds,ROUND)) + u"\u00B1"+" {}".format(round(stdperc_interyds,ROUND)))

def accuracy_step(res):
  rmsd,r2d,lld,good_moves, subgoal, rmseg, avg_sub_dist, constr, rmsed, avg_constr_dist,gmc,gmt,gms,sgc,sgt,sgs,cc,ct,cs,rmsez,good_zeros,gzc,gzt,gzs,g_auc,g_aucc,g_auct,g_aucs,g_pr,g_prc,g_prt,g_prs = [], [] ,[] ,[], [], [], [], [], [], [], [], [] ,[] ,[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
  for r in res:
    r = r[0]
    rmsd += [r[0]]
    r2d  += [r[1]]
    lld  += [r[2]]
    good_moves += [r[3]]
    subgoal += [r[4]]
    rmseg += [r[5]]
    avg_sub_dist += [r[6]]
    constr += [r[7]]
    rmsed += [r[8]]
    avg_constr_dist += [r[9]]
    gmc += [r[10]]
    gmt += [r[11]]
    gms += [r[12]]
    sgc += [r[13]]
    sgt += [r[14]]
    sgs += [r[15]]
    cc += [r[16]]
    ct += [r[17]]
    cs += [r[18]]
    rmsez += [r[19]]
    good_zeros += [r[20]]
    gzc += [r[21]]
    gzt += [r[22]]
    gzs += [r[23]]
    g_auc += [r[24]]
    g_aucc += [r[25]]
    g_auct += [r[26]]
    g_aucs += [r[27]]
    g_pr += [r[28]]
    g_prc += [r[29]]
    g_prt += [r[30]]
    g_prs += [r[31]]
  mrmsd, stdrmsd = np.mean(rmsd), np.std(rmsd)
  mr2d, stdr2d = np.mean(r2d), np.std(r2d)
  mlld, stdlld = np.mean(lld), np.std(lld)
  mgood_moves, stdgood_moves = np.mean(good_moves), np.std(good_moves)
  msubgoal, stdsubgoal = np.mean(subgoal), np.std(subgoal)
  mrmseg, stdrmseg = np.mean(rmseg), np.std(rmseg)
  mavg_sub_dist, stdavg_sub_dist = np.mean(avg_sub_dist), np.std(avg_sub_dist)
  mconstr, stdconstr = np.mean(constr), np.std(constr)
  mrmsed, stdrmsed = np.mean(rmsed), np.std(rmsed)
  mavg_constr_dist, stdavg_constr_dist = np.mean(avg_constr_dist), np.std(avg_constr_dist)
  mgmc, stdgmc = np.mean(gmc), np.std(gmc)
  mgmt, stdgmt = np.mean(gmt), np.std(gmt)
  mgms, stdgms = np.mean(gms), np.std(gms)
  msgc, stdsgc = np.mean(sgc), np.std(sgc)
  msgt, stdsgt = np.mean(sgt), np.std(sgt)
  msgs, stdsgs = np.mean(sgs), np.std(sgs)
  mcc, stdcc = np.mean(cc), np.std(cc)
  mct, stdct = np.mean(ct), np.std(ct)
  mcs, stdcs = np.mean(cs), np.std(cs)
  mrmsez, stdrmsez = np.mean(rmsez), np.std(rmsez)
  mgood_zeros, stdgood_zeros = np.mean(good_zeros), np.std(good_zeros)
  mgzc, stdgzc = np.mean(gzc), np.std(gzc)
  mgzt, stdgzt = np.mean(gzt), np.std(gzt)
  mgzs, stdgzs = np.mean(gzs), np.std(gzs) 
  mg_auc, stdg_auc = np.mean(g_auc), np.std(g_auc)
  mg_aucc, stdg_aucc = np.mean(g_aucc), np.std(g_aucc)
  mg_auct, stdg_auct = np.mean(g_auct), np.std(g_auct)
  mg_aucs, stdg_aucs = np.mean(g_aucs), np.std(g_aucs)
  mg_pr, stdg_pr = np.mean(g_pr), np.std(g_pr)
  mg_prc, stdg_prc = np.mean(g_prc), np.std(g_prc)
  mg_prt, stdg_prt = np.mean(g_prt), np.std(g_prt)
  mg_prs, stdg_prs = np.mean(g_prs), np.std(g_prs)
  print("NRMSE for displX: {} ".format(round(mrmsd,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsd,ROUND)))
  print("R2 score for displX: {} ".format(round(mr2d,ROUND)) + u"\u00B1"+" {}".format(round(stdr2d,ROUND)))
  print("Log-likelihood for displX: {} ".format(round(mlld,ROUND)) + u"\u00B1"+" {}".format(round(stdlld,ROUND)))
  print("Percentage of movements that are good: {}% ".format(round(mgood_moves,ROUND)) + u"\u00B1"+" {}%".format(round(stdgood_moves,ROUND)))
  print("Percentage of movements that are good for circles: {}% ".format(round(mgmc,ROUND)) + u"\u00B1"+" {}%".format(round(stdgmc,ROUND)))
  print("Percentage of movements that are good for triangles: {}% ".format(round(mgmt,ROUND)) + u"\u00B1"+" {}%".format(round(stdgmt,ROUND)))
  print("Percentage of movements that are good for squares: {}% ".format(round(mgms,ROUND)) + u"\u00B1"+" {}%".format(round(stdgms,ROUND)))
  print("Percentage of zero displs that are good: {}% ".format(round(mgood_zeros,ROUND)) + u"\u00B1"+" {}%".format(round(stdgood_zeros,ROUND)))
  print("Percentage of zero displs that are good for circles: {}% ".format(round(mgzc,ROUND)) + u"\u00B1"+" {}%".format(round(stdgzc,ROUND)))
  print("Percentage of zero displs that are good for triangles: {}% ".format(round(mgzt,ROUND)) + u"\u00B1"+" {}%".format(round(stdgzt,ROUND)))
  print("Percentage of zero displs that are good for squares: {}% ".format(round(mgzs,ROUND)) + u"\u00B1"+" {}%".format(round(stdgzs,ROUND)))
  print("NRMSE of zero displacement predicted: {} ".format(round(mrmsez,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsez,ROUND)))
  print("Percentage of movements that satisfy a subgoal: {}% ".format(round(msubgoal,ROUND)) + u"\u00B1"+" {}%".format(round(stdsubgoal,ROUND)))
  print("Percentage of movements that satisfy a subgoal for circles: {}% ".format(round(msgc,ROUND)) + u"\u00B1"+" {}%".format(round(stdsgc,ROUND)))
  print("Percentage of movements that satisfy a subgoal for triangles: {}% ".format(round(msgt,ROUND)) + u"\u00B1"+" {}%".format(round(stdsgt,ROUND)))
  print("Percentage of movements that satisfy a subgoal for squares: {}% ".format(round(msgs,ROUND)) + u"\u00B1"+" {}%".format(round(stdsgs,ROUND)))
  print("NRMSE of distance to reach goal: {} ".format(round(mrmseg,ROUND)) + u"\u00B1"+" {}".format(round(stdrmseg,ROUND)))
  print("AVG of distance to reach goal: {} ".format(round(mavg_sub_dist,ROUND)) + u"\u00B1"+" {}".format(round(stdavg_sub_dist,ROUND)))
  print("Percentage of movements that satisfy a constr: {}% ".format(round(mconstr,ROUND)) + u"\u00B1"+" {}".format(round(stdconstr,ROUND)))
  print("Percentage of movements that satisfy a constr for circles: {}% ".format(round(mcc,ROUND)) + u"\u00B1"+" {}".format(round(stdcc,ROUND)))
  print("Percentage of movements that satisfy a constr for triangles: {}% ".format(round(mct,ROUND)) + u"\u00B1"+" {}".format(round(stdct,ROUND)))
  print("Percentage of movements that satisfy a constr for squares: {}% ".format(round(mcs,ROUND)) + u"\u00B1"+" {}".format(round(stdcs,ROUND)))
  print("NRMSE of distance to reach constr: {} ".format(round(mrmsed,ROUND)) + u"\u00B1"+" {}".format(round(stdrmsed,ROUND)))
  print("AVG of distance to reach constr: {} ".format(round(mavg_constr_dist,ROUND)) + u"\u00B1"+" {}".format(round(stdavg_constr_dist,ROUND)))
  print("AUC ROC score for choosing to move all objects: {} ".format(round(mg_auc,ROUND)) + u"\u00B1"+" {}".format(round(stdg_auc,ROUND)))
  print("AUC PR score for choosing to move all objects: {} ".format(round(mg_pr,ROUND)) + u"\u00B1"+" {}".format(round(stdg_pr,ROUND)))
  print("AUC ROC score for choosing to move circles: {} ".format(round(mg_aucc,ROUND)) + u"\u00B1"+" {}".format(round(stdg_aucc,ROUND)))
  print("AUC PR score for choosing to move circles: {} ".format(round(mg_prc,ROUND)) + u"\u00B1"+" {}".format(round(stdg_prc,ROUND)))
  print("AUC ROC score for choosing to move triangles: {} ".format(round(mg_auct,ROUND)) + u"\u00B1"+" {}".format(round(stdg_auct,ROUND)))
  print("AUC PR score for choosing to move triangles: {} ".format(round(mg_prt,ROUND)) + u"\u00B1"+" {}".format(round(stdg_prt,ROUND)))
  print("AUC ROC score for choosing to move squares: {} ".format(round(mg_aucs,ROUND)) + u"\u00B1"+" {}".format(round(stdg_aucs,ROUND)))
  print("AUC PR score for choosing to move squares: {} ".format(round(mg_prs,ROUND)) + u"\u00B1"+" {}".format(round(stdg_prs,ROUND)))
  
  gm = {'all':(mgood_moves,stdgood_moves),'circ':(mgmc,stdgmc),'tri':(mgmt,stdgmt),'sqr':(mgms,stdgms)}
  gz = {'all':(mgood_zeros,stdgood_zeros),'circ':(mgzc,stdgzc),'tri':(mgzt,stdgzt),'sqr':(mgzs,stdgzs)}
  
  goal = {'all':(msubgoal,stdsubgoal),'circ':(msgc,stdsgc),'tri':(msgt,stdsgt),'sqr':(msgs,stdsgs)}
  
  constr = {'all':(mconstr,stdconstr),'circ':(mcc,stdcc),'tri':(mct,stdct),'sqr':(mcs,stdcs)}
  
  nrm = {'subg':(mrmseg,stdrmseg) ,'constr':(mrmsed,stdrmsed)}
  
  roc = {'all':(mg_auc,stdg_auc),'circ':(mg_aucc,stdg_aucc),'tri':(mg_auct,stdg_auct),'sqr':(mg_aucs,stdg_aucs)}
  
  pr = {'all':(mg_pr,stdg_pr),'circ':(mg_prc,stdg_prc),'tri':(mg_prt,stdg_prt),'sqr':(mg_prs,stdg_prs)}
  
  return {'move':gm,'zero':gz,'goal':goal,'constr':constr,'roc':roc,'pr':pr,'nrm':nrm}

'''def accuracy_step_1(res, twoD):
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
    print("Log-likelihood for displ : {} ".format(round(mll,ROUND)) + u"\u00B1"+" {}".format(round(stdll,ROUND)))'''
        
def the_new_tests_2D(X,Y,examples,constrained):
  displXs = []
  displYs = []    
  for e in examples:
    displXs += [X[0:len(e.init_state.objects)]]  
    X = X[len(e.init_state.objects):]
    displYs += [Y[0:len(e.init_state.objects)]]  
    Y = Y[len(e.init_state.objects):]
  assert(len(displXs) == len(examples))  
  assert(len(displYs) == len(examples))  
  gmX,gmY,bmX,bmY,gzX,gzY,bzX,bzY,interx,intery, gmXc,gmXt,gmXs, bmXc,bmXt,bmXs,gmYc,gmYt,gmYs,bmYc,bmYt,bmYs, gzXc,gzXt,gzXs,bzXc,bzXt,bzXs,gzYc,gzYt,gzYs,bzYc,bzYt,bzYs, gmXl,gmXr,gmYu,gmYd,bmXl,bmXr,bmYu,bmYd, gmXcl,gmXcr,gmXtl,gmXtr,gmXsl,gmXsr,bmXcl,bmXcr,bmXtl,bmXtr,bmXsl,bmXsr, gmYcu,gmYcd,gmYtu,gmYtd,gmYsu,gmYsd,bmYcu,bmYcd,bmYtu,bmYtd,bmYsu,bmYsd, interxl,interxr,interxcl,interxcr,interxtl,interxtr,interxsl,interxsr, interyu,interyd,interycu,interycd,interytu,interytd,interysu,interysd = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  #gmXc,gmXt,gmXs,bmXc,bmXt,bmXs,
  for displX, displY, example in zip(displXs,displYs,examples): 
    gmx,gmy,bmx,bmy,gzx,gzy,bzx,bzy,interX,interY, gmxc,gmxt,gmxs, bmxc,bmxt,bmxs,gmyc,gmyt,gmys,bmyc,bmyt,bmys, gzxc,gzxt,gzxs,bzxc,bzxt,bzxs,gzyc,gzyt,gzys,bzyc,bzyt,bzys, gmxl,gmxr,gmyu,gmyd,bmxl,bmxr,bmyu,bmyd, gmxcl,gmxcr,gmxtl,gmxtr,gmxsl,gmxsr,bmxcl,bmxcr,bmxtl,bmxtr,bmxsl,bmxsr, gmycu,gmycd,gmytu,gmytd,gmysu,gmysd,bmycu,bmycd,bmytu,bmytd,bmysu,bmysd, interXl,interXr,interXcl,interXcr,interXtl,interXtr,interXsl,interXsr, interYu,interYd,interYcu,interYcd,interYtu,interYtd,interYsu,interYsd = test_for_good_movements_and_no_intersections(displX,displY,example,constrained)
    gmX += gmx
    gmY += gmy
    bmX += bmx
    bmY += bmy
    gzX += gzx
    gzY += gzy
    bzX += bzx
    bzY += bzy
    interx += interX
    intery += interY
    gmXc += gmxc
    gmXt += gmxt
    gmXs += gmxs
    bmXc += bmxc
    bmXt += bmxt
    bmXs += bmxs
    gmYc += gmyc
    gmYt += gmyt
    gmYs += gmys
    bmYc += bmyc
    bmYt += bmyt
    bmYs += bmys
    gzXc += gzxc
    gzXt += gzxt
    gzXs += gzxs
    bzXc += bzxc
    bzXt += bzxt
    bzXs += bzxs
    gzYc += gzyc
    gzYt += gzyt
    gzYs += gzys
    bzYc += bzyc
    bzYt += bzyt
    bzYs += bzys
    gmXl += gmxl
    gmXr += gmxr
    gmYu += gmyu
    gmYd += gmyd
    bmXl += bmxl
    bmXr += bmxr
    bmYu += bmyu
    bmYd += bmyd
    gmXcl += gmxcl
    gmXcr += gmxcr
    gmXtl += gmxtl
    gmXtr += gmxtr
    gmXsl += gmxsl
    gmXsr += gmxsr
    bmXcl += bmxcl
    bmXcr += bmxcr
    bmXtl += bmxtl
    bmXtr += bmxtr
    bmXsl += bmxsl
    bmXsr += bmxsr
    gmYcu += gmycu
    gmYcd += gmycd
    gmYtu += gmytu
    gmYtd += gmytd
    gmYsu += gmysu
    gmYsd += gmysd
    bmYcu += bmycu
    bmYcd += bmycd
    bmYtu += bmytu
    bmYtd += bmytd
    bmYsu += bmysu
    bmYsd += bmysd
    interxl += interXl
    interxr += interXr
    interxcl += interXcl
    interxcr += interXcr
    interxtl += interXtl
    interxtr += interXtr
    interxsl += interXsl
    interxsr += interXsr
    interyu += interYu
    interyd += interYd
    interycu += interYcu
    interycd += interYcd 
    interytu += interYtu
    interytd += interYtd
    interysu += interYsu
    interysd += interYsd
  #TODO tot move/zero circ/tri/sqrt, tot move l/r/u/d circ,tri,sqrt
  # inter x l/r y u/d c/t/s l,r u/d
  #ALL moves
  tot_move_x = gmX+bmX
  tot_move_y = gmY+bmY
  tot_zero_x = gzX+bzX
  tot_zero_y = gzY+bzY
  roc_auc_mx, pr_auc_mx = compute_auc_roc_and_auc_pr(gmX,bmX,gzX,bzX)
  roc_auc_my, pr_auc_my = compute_auc_roc_and_auc_pr(gmY,bmY,gzY,bzY)
  perc_gmx = gmX/tot_move_x * 100 if not tot_move_x==0 else float('nan')
  perc_gmy = gmY/tot_move_y * 100 if not tot_move_y==0 else float('nan')
  perc_gzx = gzX/tot_zero_x * 100 if not tot_zero_x==0 else float('nan')
  perc_gzy = gzY/tot_zero_y * 100 if not tot_zero_y==0 else float('nan')
  
  #ALL moves circle
  tot_move_xc = gmXc+bmXc
  tot_move_yc = gmYc+bmYc
  tot_zero_xc = gzXc+bzXc
  tot_zero_yc = gzYc+bzYc
  roc_auc_mxc, pr_auc_mxc = compute_auc_roc_and_auc_pr(gmXc,bmXc,gzXc,bzXc)
  roc_auc_myc, pr_auc_myc = compute_auc_roc_and_auc_pr(gmYc,bmYc,gzYc,bzYc)
  perc_gmxc = gmXc/tot_move_xc * 100 if not tot_move_xc==0 else float('nan')
  perc_gmyc = gmYc/tot_move_yc * 100 if not tot_move_yc==0 else float('nan')
  perc_gzxc = gzXc/tot_zero_xc * 100 if not tot_zero_xc==0 else float('nan')
  perc_gzyc = gzYc/tot_zero_yc * 100 if not tot_zero_yc==0 else float('nan')
  
  #ALL moves triangle
  tot_move_xt = gmXt+bmXt
  tot_move_yt = gmYt+bmYt
  tot_zero_xt = gzXt+bzXt
  tot_zero_yt = gzYt+bzYt
  roc_auc_mxt, pr_auc_mxt = compute_auc_roc_and_auc_pr(gmXt,bmXt,gzXt,bzXt)
  roc_auc_myt, pr_auc_myt = compute_auc_roc_and_auc_pr(gmYt,bmYt,gzYt,bzYt)
  perc_gmxt = gmXt/tot_move_xt * 100 if not tot_move_xt==0 else float('nan')
  perc_gmyt = gmYt/tot_move_yt * 100 if not tot_move_yt==0 else float('nan')
  perc_gzxt = gzXt/tot_zero_xt * 100 if not tot_zero_xt==0 else float('nan')
  perc_gzyt = gzYt/tot_zero_yt * 100 if not tot_zero_yt==0 else float('nan')
  
  #ALL moves square
  tot_move_xs = gmXs+bmXs
  tot_move_ys = gmYs+bmYs
  tot_zero_xs = gzXs+bzXs
  tot_zero_ys = gzYs+bzYs
  roc_auc_mxs, pr_auc_mxs = compute_auc_roc_and_auc_pr(gmXs,bmXs,gzXs,bzXs)
  roc_auc_mys, pr_auc_mys = compute_auc_roc_and_auc_pr(gmYs,bmYs,gzYs,bzYs)
  perc_gmxs = gmXs/tot_move_xs * 100 if not tot_move_xs==0 else float('nan')
  perc_gmys = gmYs/tot_move_ys * 100 if not tot_move_ys==0 else float('nan')
  perc_gzxs = gzXs/tot_zero_xs * 100 if not tot_zero_xs==0 else float('nan')
  perc_gzys = gzYs/tot_zero_ys * 100 if not tot_zero_ys==0 else float('nan')
  
  #ALL moves lr ud
  tot_move_x_l = gmXl+bmXl
  tot_move_x_r = gmXr+bmXr
  tot_move_y_u = gmYu+bmYu
  tot_move_y_d = gmYd+bmYd
  perc_gmxl = gmXl/tot_move_x_l * 100 if not tot_move_x_l==0 else float('nan')
  perc_gmxr = gmXr/tot_move_x_r * 100 if not tot_move_x_r==0 else float('nan')
  perc_gmyu = gmYu/tot_move_y_u * 100 if not tot_move_y_u==0 else float('nan')
  perc_gmyd = gmYd/tot_move_y_d * 100 if not tot_move_y_d==0 else float('nan')
  
  #ALL moves lr ud circle
  tot_move_x_lc = gmXcl+bmXcl
  tot_move_x_rc = gmXcr+bmXcr
  tot_move_y_uc = gmYcu+bmYcu
  tot_move_y_dc = gmYcd+bmYcd
  perc_gmxlc = gmXcl/tot_move_x_lc * 100 if not tot_move_x_lc==0 else float('nan')
  perc_gmxrc = gmXcr/tot_move_x_rc * 100 if not tot_move_x_rc==0 else float('nan')
  perc_gmyuc = gmYcu/tot_move_y_uc * 100 if not tot_move_y_uc==0 else float('nan')
  perc_gmydc = gmYcd/tot_move_y_dc * 100 if not tot_move_y_dc==0 else float('nan')
  
  #ALL moves lr ud triangle
  tot_move_x_lt = gmXtl+bmXtl
  tot_move_x_rt = gmXtr+bmXtr
  tot_move_y_ut = gmYtu+bmYtu
  tot_move_y_dt = gmYtd+bmYtd
  perc_gmxlt = gmXtl/tot_move_x_lt * 100 if not tot_move_x_lt==0 else float('nan')
  perc_gmxrt = gmXtr/tot_move_x_rt * 100 if not tot_move_x_rt==0 else float('nan')
  perc_gmyut = gmYtu/tot_move_y_ut * 100 if not tot_move_y_ut==0 else float('nan')
  perc_gmydt = gmYtd/tot_move_y_dt * 100 if not tot_move_y_dt==0 else float('nan')
  
  #ALL moves lr ud square
  tot_move_x_ls = gmXsl+bmXsl
  tot_move_x_rs = gmXsr+bmXsr
  tot_move_y_us = gmYsu+bmYsu
  tot_move_y_ds = gmYsd+bmYsd
  perc_gmxls = gmXsl/tot_move_x_ls * 100 if not tot_move_x_ls==0 else float('nan')
  perc_gmxrs = gmXsr/tot_move_x_rs * 100 if not tot_move_x_rs==0 else float('nan')
  perc_gmyus = gmYsu/tot_move_y_us * 100 if not tot_move_y_us==0 else float('nan')
  perc_gmyds = gmYsd/tot_move_y_ds * 100 if not tot_move_y_ds==0 else float('nan')
  
  #18 intersections
  perc_interx = interx/tot_move_x * 100 if not tot_move_x==0 else float('nan')
  perc_intery = intery/tot_move_y * 100 if not tot_move_y==0 else float('nan')
  
  perc_interxl = interxl/interx * 100 if not interx==0 else float('nan')
  perc_interxr = interxr/interx * 100 if not interx==0 else float('nan')
  
  perc_interxlc = interxcl/interxl * 100 if not interxl==0 else float('nan')
  perc_interxlt = interxtl/interxl * 100 if not interxl==0 else float('nan')
  perc_interxls = interxsl/interxl * 100 if not interxl==0 else float('nan')
  
  perc_interxrc = interxcr/interxr * 100 if not interxr==0 else float('nan')
  perc_interxrt = interxtr/interxr * 100 if not interxr==0 else float('nan')
  perc_interxrs = interxsr/interxr * 100 if not interxr==0 else float('nan')
  
  perc_interyu = interyu/intery * 100 if not intery==0 else float('nan')
  perc_interyd = interyd/intery * 100 if not intery==0 else float('nan')
  
  perc_interyuc = interycu/interyu * 100 if not interyu==0 else float('nan')
  perc_interyut = interytu/interyu * 100 if not interyu==0 else float('nan')
  perc_interyus = interysu/interyu * 100 if not interyu==0 else float('nan')
  
  perc_interydc = interycd/interyd * 100 if not interyd==0 else float('nan')
  perc_interydt = interytd/interyd * 100 if not interyd==0 else float('nan')
  perc_interyds = interysd/interyd * 100 if not interyd==0 else float('nan')
  
  print('Total moves X ',gmX+bmX)
  print('Total moves Y ',gmY+bmY)
  print('Total zero X ',gzX+bzX)
  print('Total zero Y ',gzY+bzY)
  print('Total intersections X', interx)
  print('Total intersections Y', intery)
  
  print("Percentage good move displX ", round(perc_gmx,2))
  print("Percentage good move displX circle ", round(perc_gmxc,2))
  print("Percentage good move displX triangle ", round(perc_gmxt,2))
  print("Percentage good move displX square ", round(perc_gmxs,2))
  print("Percentage good move displX left ", round(perc_gmxl,2))
  print("Percentage good move displX left circle ", round(perc_gmxlc,2))
  print("Percentage good move displX left triangle ", round(perc_gmxlt,2))
  print("Percentage good move displX left square ", round(perc_gmxls,2))
  print("Percentage good move displX right ", round(perc_gmxr,2))
  print("Percentage good move displX right circle ", round(perc_gmxrc,2))
  print("Percentage good move displX right triangle ", round(perc_gmxrt,2))
  print("Percentage good move displX right square ", round(perc_gmxrs,2))
  print("AUC ROC good move displX ", round(roc_auc_mx,2))
  print("AUC PR good move displX ", round(pr_auc_mx,2))
  print("AUC ROC good move displX circle ", round(roc_auc_mxc,2))
  print("AUC PR good move displX circle ", round(pr_auc_mxc,2))
  print("AUC ROC good move displX triangle ", round(roc_auc_mxt,2))
  print("AUC PR good move displX triangle ", round(pr_auc_mxt,2))
  print("AUC ROC good move displX square ", round(roc_auc_mxs,2))
  print("AUC PR good move displX square ", round(pr_auc_mxs,2))
  
  print("Percentage good move displY ", round(perc_gmy,2))
  print("Percentage good move displY circle ", round(perc_gmyc,2))
  print("Percentage good move displY triangle ", round(perc_gmyt,2))
  print("Percentage good move displY square ", round(perc_gmys,2))
  print("Percentage good move displY north ", round(perc_gmyu,2))
  print("Percentage good move displY north circle ", round(perc_gmyuc,2))
  print("Percentage good move displY north triangle ", round(perc_gmyut,2))
  print("Percentage good move displY north square ", round(perc_gmyus,2))
  print("Percentage good move displY south ", round(perc_gmyd,2))
  print("Percentage good move displY south circle ", round(perc_gmydc,2))
  print("Percentage good move displY south triangle ", round(perc_gmydt,2))
  print("Percentage good move displY south square ", round(perc_gmyds,2))
  print("AUC ROC good move displY ", round(roc_auc_my,2))
  print("AUC PR good move displY ", round(pr_auc_my,2))
  print("AUC ROC good move displY circle ", round(roc_auc_myc,2))
  print("AUC PR good move displY circle ", round(pr_auc_myc,2))
  print("AUC ROC good move displY triangle ", round(roc_auc_myt,2))
  print("AUC PR good move displY triangle ", round(pr_auc_myt,2))
  print("AUC ROC good move displY square ", round(roc_auc_mys,2))
  print("AUC PR good move displY square ", round(pr_auc_mys,2))
  
  print("Percentage good zero displX ", round(perc_gzx,2))
  print("Percentage good zero displX circle ", round(perc_gzxc,2))
  print("Percentage good zero displX triangle ", round(perc_gzxt,2))
  print("Percentage good zero displX square ", round(perc_gzxs,2))
  
  print("Percentage good zero displY ", round(perc_gzy,2))
  print("Percentage good zero displY circle ", round(perc_gzyc,2))
  print("Percentage good zero displY triangle ", round(perc_gzyt,2))
  print("Percentage good zero displY square ", round(perc_gzys,2))
  
  print("Percentage intersections X-axis ", round(perc_interx,2))
  print("Percentage intersections X-axis left ", round(perc_interxl,2))
  print("Percentage intersections X-axis left circle ", round(perc_interxlc,2))
  print("Percentage intersections X-axis left triangle ", round(perc_interxlt,2))
  print("Percentage intersections X-axis left square ", round(perc_interxls,2))
  print("Percentage intersections X-axis right ", round(perc_interxr,2))
  print("Percentage intersections X-axis right circle ", round(perc_interxrc,2))
  print("Percentage intersections X-axis right triangle ", round(perc_interxrt,2))
  print("Percentage intersections X-axis right square ", round(perc_interxrs,2))
  
  print("Percentage intersections Y-axis ", round(perc_intery,2))
  print("Percentage intersections Y-axis north ", round(perc_interyu,2))
  print("Percentage intersections Y-axis north circle ", round(perc_interyuc,2))
  print("Percentage intersections Y-axis north triangle ", round(perc_interyut,2))
  print("Percentage intersections Y-axis north square ", round(perc_interyus,2))
  print("Percentage intersections Y-axis south ", round(perc_interyd,2))
  print("Percentage intersections Y-axis south circle ", round(perc_interydc,2))
  print("Percentage intersections Y-axis south triangle ", round(perc_interydt,2))
  print("Percentage intersections Y-axis south square ", round(perc_interyds,2))
  
  return roc_auc_mx, pr_auc_mx, roc_auc_my, pr_auc_my, perc_gmx, perc_gmy, perc_gzx, perc_gzy, roc_auc_mxc, pr_auc_mxc, roc_auc_myc, pr_auc_myc, perc_gmxc, perc_gmyc, perc_gzxc, perc_gzyc, roc_auc_mxt, pr_auc_mxt, roc_auc_myt, pr_auc_myt, perc_gmxt, perc_gmyt, perc_gzxt, perc_gzyt, roc_auc_mxs, pr_auc_mxs, roc_auc_mys, pr_auc_mys, perc_gmxs, perc_gmys, perc_gzxs, perc_gzys, perc_gmxl, perc_gmxr, perc_gmyu, perc_gmyd, perc_gmxlc, perc_gmxrc, perc_gmyuc, perc_gmydc, perc_gmxlt, perc_gmxrt, perc_gmyut, perc_gmydt, perc_gmxls, perc_gmxrs, perc_gmyus, perc_gmyds, perc_interx, perc_intery, perc_interxl, perc_interxr, perc_interxlc, perc_interxlt, perc_interxls, perc_interxrc, perc_interxrt, perc_interxrs, perc_interyu, perc_interyd, perc_interyuc, perc_interyut, perc_interyus, perc_interydc, perc_interydt, perc_interyds #66
  
def compute_auc_roc_and_auc_pr(tp,fp,tn,fn):
  fpr = fp/(fp+tn) if not fp+tn==0 else float('nan')
  tpr = tp/(tp+fn) if not tp+fn==0 else float('nan')
  ppr = tp/(tp+fp) if not tp+fp==0 else float('nan') 
  roc_auc = auc([0,fpr,1],[0,tpr,1])
  pr_auc = auc([0,tpr,1],[1,ppr,0])
  return roc_auc, pr_auc
  
def the_new_tests(predictions, examples,constrained):

  displs = []
  for e in examples:
    displs += [predictions[0:len(e.init_state.objects)]]
    predictions = predictions[len(e.init_state.objects):]
    
  assert(len(displs) == len(examples))
  length = 0
  gr = 0
  dk = 0
  bb = 0
  llc,llt,lls,bbc,bbt,bbs,ggc,ggt,ggs,ddc,ddt,dds=0,0,0,0,0,0,0,0,0,0,0,0
  ttot,zz = 0,0
  zzg,zzb,zzgc,zzgt,zzgs,zzbc,zzbt,zzbs = 0,0,0,0,0,0,0,0
  ttg,ttd,aag,aad,ttz,aaz = [], [], [], [], [], []
  for displ, example in zip(displs, examples):
    l, g, d, b , tg, td, ag, ad, tz, az, lc,lt,ls,bc,bt,bs,gc,gt,gs,dc,dt,ds,tot,z,zg,zb,zgc,zgt,zgs,zbc,zbt,zbs = test_for_goal_achieved_and_keeped_distance(displ, example, constrained)
    length += l
    gr += g
    dk += d
    bb += b
    ttg += tg
    ttd += td
    aag += ag
    aad += ad
    ttz += tz
    aaz += az
    llc += lc
    llt += lt
    lls += ls
    bbc += bc
    bbt += bt
    bbs += bs
    ggc += gc
    ggt += gt
    ggs += gs
    ddc += dc
    ddt += dt
    dds += ds
    ttot += tot
    zz += z
    zzg += zg
    zzb += zb
    zzgc += zgc
    zzgt += zgt
    zzgs += zgs
    zzbc += zbc
    zzbt += zbt
    zzbs += zbs
    #print(len(ttg),len(aag),len(ttd),len(aad))
    #print(l,g,d)
    #print(displ)
    #print(ag)
    #print(ad)
    #draw_example(example)
    
  rmseg = sqrt(mse(ttg, aag))/RANGE
  rmsed = sqrt(mse(ttd, aad))/RANGE
  rmsez = sqrt(mse(ttz, aaz))/RANGE
  good_moves = (length/(length+bb))*100
  gmc = (llc/(llc+bbc))*100 if not llc+bbc==0 else float('nan') #good_move_circle
  gmt = (llt/(llt+bbt))*100 if not llt+bbt==0 else float('nan')
  gms = (lls/(lls+bbs))*100 if not lls+bbs==0 else float('nan')
  good_zeros = (zzg/(zzg+zzb))*100
  gzc = (zzgc/(zzgc+zzbc))*100 if not zzgc+zzbc==0 else float('nan')
  gzt = (zzgt/(zzgt+zzbt))*100 if not zzgt+zzbt==0 else float('nan')
  gzs = (zzgs/(zzgs+zzbs))*100 if not zzgs+zzbs==0 else float('nan')
  subgoal = (gr/(length+bb))*100
  sgc = (ggc/(llc+bbc))*100 if not llc+bbc==0 else float('nan') #subgoal_circle
  sgt = (ggt/(llt+bbt))*100 if not llt+bbt==0 else float('nan')
  sgs = (ggs/(lls+bbs))*100 if not lls+bbs==0 else float('nan')
  avg_sub_dist = sum(aag)/len(aag)
  constr = (dk/(length+bb))*100
  cc = (ddc/(llc+bbc))*100 if not llc+bbc==0 else float('nan') #constr_circle
  ct = (ddt/(llt+bbt))*100 if not llt+bbt==0 else float('nan')
  cs = (dds/(lls+bbs))*100 if not lls+bbs==0 else float('nan')
  avg_constr_dist = sum(aad)/len(aad)
  print("Percentage of movements that are good: {}%".format(round(good_moves,2)))
  print("Percentage of movements that are good for circles: {}%".format(round(gmc,2)))
  print("Percentage of movements that are good for triangles: {}%".format(round(gmt,2)))
  print("Percentage of movements that are good for squares: {}%".format(round(gms,2)))
  print("Percentage of zero displs that are good: {}%".format(round(good_zeros,2)))
  print("Percentage of zero displs that are good for circles: {}%".format(round(gzc,2)))
  print("Percentage of zero displs that are good for triangles: {}%".format(round(gzt,2)))
  print("Percentage of zero displs that are good for squares: {}%".format(round(gzs,2))) 
  print("RMSE of zero displacement predicted: {}".format(round(rmsez,4)))
  print("Percentage of movements that satisfy a subgoal: {}%".format(round(subgoal,2)))
  print("Percentage of movements that satisfy a subgoal for circles: {}%".format(round(sgc,2)))
  print("Percentage of movements that satisfy a subgoal for triangles: {}%".format(round(sgt,2)))
  print("Percentage of movements that satisfy a subgoal for squares: {}%".format(round(sgs,2)))
  print("RMSE of distance to reach goal: {}".format(round(rmseg,4)))
  print("AVG of distance to reach goal: {}".format(round(avg_sub_dist,4)))
  print("Percentage of movements that satisfy a constr: {}%".format(round(constr,2)))
  print("Percentage of movements that satisfy a constr for circles: {}%".format(round(cc,2)))
  print("Percentage of movements that satisfy a constr for triangles: {}%".format(round(ct,2)))
  print("Percentage of movements that satisfy a constr for squares: {}%".format(round(cs,2)))
  print("RMSE of distance to reach constr: {}".format(round(rmsed,4)))
  print("AVG of distance to reach constr: {}".format(round(avg_constr_dist,4)))
  #print('Tot ',ttot,', zero ',zz)
  #print('AUC: TMPM ',length,', TZPM ',bb,', TZPZ', zzg,', TMPZ', zzb)
  #print('Goal+const ',gr,dk)
  #print('Circle auc ',llc,bbc,zzgc,zzbc)
  #print('Circle g+c ',ggc,ddc)
  #print('Triange auc ',llt,bbt,zzgt,zzbt)
  #print('Triange g+c ',ggt,ddt)
  #print('Square auc ',lls,bbs,zzgs,zzbs)
  #print('Square g+c ',ggs,dds)
        #rec #prec
  fprg, tprg, pppg = bb/(bb+zzg), length/(length+zzb), length/(length+bb)
  fprgc, tprgc, pppgc = bbc/(bbc+zzgc), llc/(llc+zzbc), llc/(llc+bbc)
  fprgt, tprgt, pppgt = bbt/(bbt+zzgt), llt/(llt+zzbt), llt/(llt+bbt)
  if not lls+zzbs==0:
    fprgs, tprgs, pppgs = bbs/(bbs+zzgs), lls/(lls+zzbs), lls/(lls+bbs)
  g_auc = auc([0,fprg,1],[0,tprg,1])
  g_pr = auc([0,tprg,1],[1,pppg,0])
  g_aucc = auc([0,fprgc,1],[0,tprgc,1])
  g_prc = auc([0,tprgc,1],[1,pppgc,0])
  g_auct = auc([0,fprgt,1],[0,tprgt,1])
  g_prt = auc([0,tprgt,1],[1,pppgt,0])
  if not lls+zzbs==0:
    g_aucs = auc([0,fprgs,1],[0,tprgs,1])
    g_prs = auc([0,tprgs,1],[1,pppgs,0])
  else:
    g_aucs = float('nan')
    g_prs = float('nan')
  print("AUC ROC for choosing to move all objects: {}".format(round(g_auc,4)))
  print("AUC PR for choosing to move all objects: {}".format(round(g_pr,4)))
  print("AUC ROC for choosing to move circles: {}".format(round(g_aucc,4)))
  print("AUC PR for choosing to move circles: {}".format(round(g_prc,4)))
  print("AUC ROC for choosing to move triangles: {}".format(round(g_auct,4)))
  print("AUC PR for choosing to move triangles: {}".format(round(g_prt,4)))
  print("AUC ROC for choosing to move squares: {}".format(round(g_aucs,4)))
  print("AUC PR for choosing to move squares: {}".format(round(g_prs,4)))
  #plot_roc("for Movement", [0,fprg,1], [0,tprg,1], g_auc)
  #plot_pr("for Movement", [0,tprg,1], [1,pppg,0], g_pr)
  #plot_roc("for Movement of circles", [0,fprgc,1], [0,tprgc,1], g_aucc)
  #plot_pr("for Movement of circles", [0,tprgc,1], [1,pppgc,0], g_prc)
  #plot_roc("for Movement of triangles", [0,fprgt,1], [0,tprgt,1], g_auct)
  #plot_pr("for Movement of triangles", [0,tprgt,1], [1,pppgt,0], g_prt)
  #if not lls+zzbs==0:
  #  plot_roc("for Movement of squares", [0,fprgs,1], [0,tprgs,1], g_aucs)
  #  plot_pr("for Movement of squares", [0,tprgs,1], [1,pppgs,0], g_prs)
  return (good_moves, subgoal, rmseg, avg_sub_dist, constr, rmsed, avg_constr_dist, gmc,gmt,gms,sgc,sgt,sgs,cc,ct,cs,rmsez,good_zeros,gzc,gzt,gzs,g_auc,g_aucc,g_auct,g_aucs,g_pr,g_prc,g_prt,g_prs)
  
def plot_roc(name, fpr, tpr, roc_auc):
  plt.title('Receiver Operating Characteristic '+name)
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()
def plot_pr(name, recall, precision, pr_auc):
  plt.title('Receiver Operating Characteristic '+name)
  plt.plot(recall, precision, 'b', label = 'AUC = %0.4f' % pr_auc)
  plt.legend(loc = 'lower left')
  plt.plot([0, 1], [0, 0],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('Precision')
  plt.xlabel('Recall')
  plt.show()
  
def test_for_good_movements_and_no_intersections(displX,displY,example,constrained):
  gmx,gmy,bmx,bmy,gzx,gzy,bzx,bzy,interX,interY = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  gmxc,gmxt,gmxs,bmxc,bmxt,bmxs,gmyc,gmyt,gmys,bmyc,bmyt,bmys, gzxc,gzxt,gzxs,bzxc,bzxt,bzxs,gzyc,gzyt,gzys,bzyc,bzyt,bzys = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  gmxl,gmxr,gmyu,gmyd,bmxl,bmxr,bmyu,bmyd = 0,0,0,0,0,0,0,0
  gmxcl,gmxcr,gmxtl,gmxtr,gmxsl,gmxsr,bmxcl,bmxcr,bmxtl,bmxtr,bmxsl,bmxsr = 0,0,0,0,0,0,0,0,0,0,0,0
  gmycu,gmycd,gmytu,gmytd,gmysu,gmysd,bmycu,bmycd,bmytu,bmytd,bmysu,bmysd = 0,0,0,0,0,0,0,0,0,0,0,0
  interXl,interXr,interXcl,interXcr,interXtl,interXtr,interXsl,interXsr, interYu,interYd,interYcu,interYcd,interYtu,interYtd,interYsu,interYsd = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  rels = example.init_state.get_relations()
  for i,(x,y) in enumerate(zip(displX,displY)):
    #case heavy object and constrained scenario
    if example.init_state.objects[i].heavy and constrained:
      if abs(x) < 0.1:
        gzx += 1
        if example.init_state.objects[i].shape == 'circle':
          gzxc += 1
        elif example.init_state.objects[i].shape == 'triangle':
          gzxt += 1
        elif example.init_state.objects[i].shape == 'square':
          gzxs += 1
      else:
        bmx += 1
        if example.init_state.objects[i].shape == 'circle':
          if x < 0.1:
            bmxl += 1
            bmxcl += 1
            bmxc += 1
          else:
            bmxc += 1
            bmxr += 1
            bmxcr += 1
        elif example.init_state.objects[i].shape == 'triangle':
          if x < 0.1:
            bmxl += 1
            bmxtl += 1
            bmxt += 1
          else:
            bmxt += 1
            bmxr += 1
            bmxtr += 1
        elif example.init_state.objects[i].shape == 'square':
          if x < 0.1:
            bmxs += 1
            bmxl += 1
            bmxsl += 1
          else:
            bmxs += 1
            bmxr += 1
            bmxsr += 1
      if abs(y) < 0.1:
        gzy += 1
        if example.init_state.objects[i].shape == 'circle':
          gzyc += 1
        elif example.init_state.objects[i].shape == 'triangle':
          gzyt += 1
        elif example.init_state.objects[i].shape == 'square':
          gzys += 1
      else:
        bmy += 1
        if example.init_state.objects[i].shape == 'circle':
          if y < 0.1:
            bmyd += 1
            bmycd += 1
            bmyc += 1
          else:
            bmyc += 1
            bmyu += 1
            bmycu += 1
        elif example.init_state.objects[i].shape == 'triangle':
          if y < 0.1:
            bmyt += 1
            bmyd += 1
            bmytd += 1
          else:
            bmyt += 1
            bmyu += 1
            bmytu += 1
        elif example.init_state.objects[i].shape == 'square':
          if y < 0.1:
            bmys += 1
            bmyd += 1
            bmysd += 1
          else:
            bmys += 1
            bmyu += 1
            bmysu += 1
    #CIRCLE
    if example.init_state.objects[i].shape == 'circle':
      circle_rels = filter_rels_2D('circle', rels, example, i)
      #case zero x
      if abs(x) < 0.1: 
        if not (circle_rels[0] or circle_rels[1]):
          gzx += 1 # no left,north obj
          gzxc += 1
        elif circle_rels[0] and circle_rels[1]:
          if blocked_left(i,example,0.75): # left,north obj but blocked left
            gzx += 1
            gzxc += 1
          else: # left,north obj and not blocked blocked left
            bzx += 1
            bzxc += 1
        elif circle_rels[0]: 
          if blocked_left(i,example,0.75): # left obj but blocked left
            gzx += 1
            gzxc += 1
          else: # left obj and not blocked left
            bzx += 1
            bzxc += 1
        elif circle_rels[1]:
          if not blocked_north(i,example,0.75): #north obj but not blocked north
            gzx += 1
            gzxc += 1
          else: # north obj and blocked north
            if blocked_left(i,example,0.75): # blocked left
              gzx += 1
              gzxc += 1
            else: # not blocked left
              bzx += 1
              bzxc += 1
      # case left
      elif x<0:
        if blocked_left(i, example, x):
          interX += 1
          interXl += 1
          interXcl += 1 
        if not (circle_rels[0] or circle_rels[1]): # no left, north obj 
          bmx += 1
          bmxl += 1
          bmxcl += 1
          bmxc += 1
        elif circle_rels[0] and circle_rels[1]:
          if blocked_left(i,example,0.75): # left,north obj but blocked left
            bmx += 1
            bmxl += 1
            bmxcl += 1
            bmxc += 1
          else: # left,north obj and not blocked left
            gmx += 1
            gmxl += 1
            gmxcl += 1
            gmxc += 1
        elif circle_rels[0]:
          if blocked_left(i,example,0.75): # left obj but blocked left
            bmx += 1
            bmxl += 1
            bmxcl += 1
            bmxc += 1
          else: # left obj and not blocked left
            gmx += 1
            gmxl += 1
            gmxcl += 1
            gmxc += 1
        elif circle_rels[1]:
          if blocked_north(i,example,0.75): # north object and blocked north
            if not blocked_left(i,example,0.75): # not blocked left
              gmx += 1
              gmxl += 1
              gmxcl += 1
              gmxc += 1
            else: # blocked left
              bmx += 1
              bmxl += 1
              bmxcl += 1
              bmxc += 1
          else: # north object and not blocked north
            bmx += 1
            bmxl += 1
            bmxcl += 1
            bmxc += 1
      # case right
      elif x>0: 
        bmx += 1 #bad move
        bmxr += 1
        bmxcr += 1
        bmxc += 1
        if blocked_right(i, example, x):
          interX += 1
          interXr += 1
          interXcr += 1
      #case zero y
      if abs(y) < 0.1: 
        if not (circle_rels[0] or circle_rels[1]): # no left,north obj
          gzy += 1
          gzyc += 1
        elif circle_rels[0] and circle_rels[1]:
          if blocked_north(i,example,0.75): # left,north obj and blocked north
            gzy += 1
            gzyc += 1
          else: # left,north obj and not blocked north
            bzy += 1
            bzyc += 1
        elif circle_rels[0]:
          if blocked_left(i,example,0.75): # left obj and blocked left
            if blocked_north(i,example,0.75): # blocked north
              gzy += 1
              gzyc += 1
            else: # not blocked north
              bzy += 1
              bzyc += 1
          else: # left obj and not blocked left
            gzy += 1
            gzyc += 1
        elif circle_rels[1]:
          if blocked_north(i,example,0.75): # north obj and blocked north
            gzy += 1
            gzyc += 1
          else: # north obj and not blocked north
            bzy += 1
            bzyc += 1
      # case south
      elif y<0: 
        bmy += 1
        bmyd += 1
        bmycd += 1
        bmyc += 1
        if blocked_south(i, example, y):
          interY += 1
          interYd += 1
          interYcd += 1
      # case north
      elif y>0:
        if blocked_north(i, example, y):
          interY += 1
          interYu += 1
          interYcu += 1
        if not (circle_rels[0] or circle_rels[1]): # no left,north obj
          bmy += 1
          bmyu += 1
          bmycu += 1
          bmyc += 1
        elif circle_rels[0] and circle_rels[1]:
          if blocked_north(i, example, 0.75): # left,north obj and blocked north
            bmy += 1
            bmyu += 1
            bmycu += 1
            bmyc += 1
          else: # left,north obj and not blocked north
            gmy += 1
            gmyu += 1
            gmycu += 1
            gmyc += 1
        elif circle_rels[0]:
          if blocked_left(i, example, 0.75): # left obj and blocked left
            if blocked_north(i, example, 0.75): # blocked north
              bmy += 1
              bmyu += 1
              bmycu += 1
              bmyc += 1
            else: # not blocked north
              gmy += 1
              gmyu += 1
              gmycu += 1
              gmyc += 1
          else: # left obj and not blocked left
            bmy += 1
            bmyu += 1
            bmycu += 1
            bmyc += 1
        elif circle_rels[1]:
          if blocked_north(i, example, 0.75): # north obj and blocked north
            bmy += 1
            bmyu += 1
            bmycu += 1
            bmyc += 1
          else: # north obj and not blocked north
            gmy += 1
            gmyu += 1
            gmycu += 1
            gmyc += 1
    #TRIANGLE
    elif example.init_state.objects[i].shape == 'triangle':
      triangle_rels = filter_rels_2D('triangle', rels, example, i)
      constr_rels = filter_constr_rels_2D('triangle', rels, example, i)
      #case zero x
      if abs(x) < 0.1:
        if not (triangle_rels[0] or triangle_rels[1]):
          if constrained: # no left,north object constrained
            if not (constr_rels[0] or constr_rels[1]): # no south, right obj
              gzx += 1
              gzxt += 1
            elif constr_rels[0] and constr_rels[1]: 
              if blocked_right(i, example, 0.75): # south, right obj and blocked r
                gzx += 1
                gzxt += 1
              else: # south, right obj and not blocked right
                bzx += 1
                bzxt += 1
            elif constr_rels[0]:
              if blocked_right(i, example, 0.75): # right obj and blocked right
                gzx += 1
                gzxt += 1
              else: # right obj and not blocked right
                bzx += 1
                bzxt += 1
            elif constr_rels[1]:
              if blocked_south(i, example, 0.75): # south obj and blocked south
                if blocked_right(i, example, 0.75): # blocked right
                  gzx += 1
                  gzxt += 1
                else: # not blocked right
                  bzx += 1
                  bzxt += 1
              else: # south obj and not blocked south
                gzx += 1
                gzxt += 1
          else: # no left, north obj and not constrained
            gzx += 1
            gzxt += 1
        elif triangle_rels[0] and triangle_rels[1]:
          if blocked_left(i, example, 0.75): # left, north objects and blocked left
            if constrained:
              if not (constr_rels[0] or constr_rels[1]): # no south, right obj
                gzx += 1
                gzxt += 1
              elif constr_rels[0] and constr_rels[1]: 
                if blocked_right(i, example, 0.75): # south, right obj and blocked r
                  gzx += 1
                  gzxt += 1
                else: # south, right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75): # right obj and blocked right
                  gzx += 1
                  gzxt += 1
                else: # right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75): # south obj and blocked south
                  if blocked_right(i, example, 0.75): # blocked right
                    gzx += 1
                    gzxt += 1
                  else: # not blocked right
                    bzx += 1
                    bzxt += 1
                else: # south obj and not blocked south
                  gzx += 1
                  gzxt += 1
            else: 
              gzx += 1
              gzxt += 1
          else: # left, north objects and not blocked left
            bzx += 1
            bzxt += 1
        elif triangle_rels[0]:
          if blocked_left(i, example, 0.75): # left obj and blocked left
            if constrained:
              if not (constr_rels[0] or constr_rels[1]): # no south, right obj
                gzx += 1
                gzxt += 1
              elif constr_rels[0] and constr_rels[1]: 
                if blocked_right(i, example, 0.75): # south, right obj and blocked r
                  gzx += 1
                  gzxt += 1
                else: # south, right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75): # right obj and blocked right
                  gzx += 1
                  gzxt += 1
                else: # right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75): # south obj and blocked south
                  if blocked_right(i, example, 0.75): # blocked right
                    gzx += 1
                    gzxt += 1
                  else: # not blocked right
                    bzx += 1
                    bzxt += 1
                else: # south obj and not blocked south
                  gzx += 1
                  gzxt += 1
            else:
              gzx += 1
              gzxt += 1
          else: # left obj and not blocked left
            bzx += 1
            bzxt += 1
        elif triangle_rels[1]:
          if blocked_north(i, example, 0.75): # north obj and blocked north
            if blocked_left(i, example, 0.75): # blocked left
              if constrained:
                if not (constr_rels[0] or constr_rels[1]): # no south, right obj
                  gzx += 1
                  gzxt += 1
                elif constr_rels[0] and constr_rels[1]: 
                  if blocked_right(i, example, 0.75): # south, right obj and blocked r
                    gzx += 1
                    gzxt += 1
                  else: # south, right obj and not blocked right
                    bzx += 1
                    bzxt += 1
                elif constr_rels[0]:
                  if blocked_right(i, example, 0.75): # right obj and blocked right
                    gzx += 1
                    gzxt += 1
                  else: # right obj and not blocked right
                    bzx += 1
                    bzxt += 1
                elif constr_rels[1]:
                  if blocked_south(i, example, 0.75): # south obj and blocked south
                    if blocked_right(i, example, 0.75): # blocked right
                      gzx += 1
                      gzxt += 1
                    else: # not blocked right
                      bzx += 1
                      bzxt += 1
                  else: # south obj and not blocked south
                    gzx += 1
                    gzxt += 1
              else:
                gzx += 1
                gzxt += 1
            else: # not blocked left
              bzx += 1
              bzxt += 1
          else: # north obj and not blocked north
            if constrained:
              if not (constr_rels[0] or constr_rels[1]): # no south, right obj
                gzx += 1
                gzxt += 1
              elif constr_rels[0] and constr_rels[1]: 
                if blocked_right(i, example, 0.75): # south, right obj and blocked r
                  gzx += 1
                  gzxt += 1
                else: # south, right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75): # right obj and blocked right
                  gzx += 1
                  gzxt += 1
                else: # right obj and not blocked right
                  bzx += 1
                  bzxt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75): # south obj and blocked south
                  if blocked_right(i, example, 0.75): # blocked right
                    gzx += 1
                    gzxt += 1
                  else: # not blocked right
                    bzx += 1
                    bzxt += 1
                else: # south obj and not blocked south
                  gzx += 1
                  gzxt += 1
            else:
              gzx += 1
              gzxt += 1
      #case left
      elif x<0:
        if blocked_left(i, example, x):
          interX += 1
          interXl += 1
          interXtl += 1
        if not (triangle_rels[0] or triangle_rels[1]): # no left, north obj
          bmx += 1
          bmxl += 1
          bmxtl += 1
          bmxt += 1
        elif triangle_rels[0] and triangle_rels[1]:
          if blocked_left(i, example, 0.75): # left,north obj and blocked left
            bmx += 1
            bmxl += 1
            bmxtl += 1
            bmxt += 1
          else: # left,north obj and and blocked left
            gmx += 1
            gmxl += 1
            gmxtl += 1
            gmxt += 1
        elif triangle_rels[0]: 
          if blocked_left(i, example, 0.75): # left obj and blocked left
            bmx += 1
            bmxl += 1
            bmxtl += 1
            bmxt += 1
          else: # left obj and not blocked left
            gmx += 1
            gmxl += 1
            gmxtl += 1
            gmxt += 1
        elif triangle_rels[1]:    
          if blocked_north(i, example, 0.75): # north obj and blocked north
            if blocked_left(i, example, 0.75): # blocke left
              bmx += 1
              bmxl += 1
              bmxtl += 1
              bmxt += 1
            else: # not blocked left
              gmx += 1
              gmxl += 1
              gmxtl += 1
              gmxt += 1
          else: # north object and not blocked north
            bmx += 1
            bmxl += 1
            bmxtl += 1
            bmxt += 1
      #case right
      elif x>0:
        if blocked_right(i, example, x):
          interX += 1
          interXr += 1
          interXtr += 1
        if not constrained:
          bmx += 1
          bmxr += 1
          bmxtr += 1
          bmxt += 1
        else:
          if not (constr_rels[0] or constr_rels[1]): # no right,south obj
            bmx += 1
            bmxr += 1
            bmxtr += 1
            bmxt += 1
          elif constr_rels[0] and constr_rels[1]: 
            if blocked_right(i, example, 0.75): # right,south obj and blocked right
              bmx += 1
              bmxr += 1
              bmxtr += 1
              bmxt += 1
            else: # right,south obj and not blocked right
              gmx += 1
              gmxr += 1
              gmxtr += 1
              gmxt += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75): # right obj and blocked right
              bmx += 1
              bmxr += 1
              bmxtr += 1
              bmxt += 1
            else: # right obj and not blocked right
              gmx += 1
              gmxr += 1
              gmxtr += 1
              gmxt += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75): # south obj and blocked south
              if blocked_right(i, example, 0.75): # blocked right
                bmx += 1
                bmxr += 1
                bmxtr += 1
                bmxt += 1
              else: # not blocked right
                gmx += 1
                gmxr += 1
                gmxtr += 1
                gmxt += 1
            else: # south obj and not blocked south
              bmx += 1
              bmxr += 1
              bmxtr += 1
              bmxt += 1
      # case zero y
      if abs(y) < 0.1:
        if not (triangle_rels[0] or triangle_rels[1]):
          if constrained:
            if not (constr_rels[0] or constr_rels[1]):
              gzy += 1
              gzyt += 1
            elif constr_rels[0] and constr_rels[1]:
              if blocked_south(i, example, 0.75):
                gzy += 1
                gzyt += 1
              else:
                bzy += 1
                bzyt += 1
            elif constr_rels[0]:
              if blocked_right(i, example, 0.75):
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
              else:
                gzy += 1
                gzyt += 1
            elif constr_rels[1]:
              if blocked_south(i, example, 0.75):
                gzy += 1
                gzyt += 1
              else:
                bzy += 1
                bzyt += 1
          else:
            gzy += 1
            gzyt += 1
        elif triangle_rels[0] and triangle_rels[1]:
          if blocked_north(i, example, 0.75):
            if constrained:
              if not (constr_rels[0] or constr_rels[1]):
                gzy += 1
                gzyt += 1
              elif constr_rels[0] and constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75):
                  if blocked_south(i, example, 0.75):
                    gzy += 1
                    gzyt += 1
                  else:
                    bzy += 1
                    bzyt += 1
                else:
                  gzy += 1
                  gzyt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
            else:
              gzy += 1
              gzyt += 1
          else:
            bzy += 1
            bzyt += 1
        elif triangle_rels[0]:
          if blocked_left(i, example, 0.75):
            if blocked_north(i, example, 0.75):
              if constrained:
                if not (constr_rels[0] or constr_rels[1]):
                  gzy += 1
                  gzyt += 1
                elif constr_rels[0] and constr_rels[1]:
                  if blocked_south(i, example, 0.75):
                    gzy += 1
                    gzyt += 1
                  else:
                    bzy += 1
                    bzyt += 1
                elif constr_rels[0]:
                  if blocked_right(i, example, 0.75):
                    if blocked_south(i, example, 0.75):
                      gzy += 1
                      gzyt += 1
                    else:
                      bzy += 1
                      bzyt += 1
                  else:
                    gzy += 1
                    gzyt += 1
                elif constr_rels[1]:
                  if blocked_south(i, example, 0.75):
                    gzy += 1
                    gzyt += 1
                  else:
                    bzy += 1
                    bzyt += 1
              else:
                gzy += 1
                gzyt += 1
            else:
              bzy += 1
              bzyt += 1
          else:
            if constrained:
              if not (constr_rels[0] or constr_rels[1]):
                gzy += 1
                gzyt += 1
              elif constr_rels[0] and constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75):
                  if blocked_south(i, example, 0.75):
                    gzy += 1
                    gzyt += 1
                  else:
                    bzy += 1
                    bzyt += 1
                else:
                  gzy += 1
                  gzyt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1 
                  bzyt += 1 
            else:
              gzy += 1
              gzyt += 1
        elif triangle_rels[1]:
          if blocked_north(i, example, 0.75):
            if constrained:
              if not (constr_rels[0] or constr_rels[1]):
                gzy += 1
                gzyt += 1
              elif constr_rels[0] and constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
              elif constr_rels[0]:
                if blocked_right(i, example, 0.75):
                  if blocked_south(i, example, 0.75):
                    gzy += 1
                    gzyt += 1
                  else:
                    bzy += 1
                    bzyt += 1
                else:
                  gzy += 1
                  gzyt += 1
              elif constr_rels[1]:
                if blocked_south(i, example, 0.75):
                  gzy += 1
                  gzyt += 1
                else:
                  bzy += 1
                  bzyt += 1
            else:
              gzy += 1
              gzyt += 1
          else:
            bzy += 1
            bzyt += 1
        
      # case south
      elif y<0:
        if blocked_south(i, example, y):
          interY += 1
          interYd += 1
          interYtd += 1
        if not constrained:
          bmy += 1
          bmyd += 1
          bmytd += 1
          bmyt += 1
        else:
          if not (constr_rels[0] or constr_rels[1]):
            bmy += 1
            bmyd += 1
            bmytd += 1
            bmyt += 1
          elif constr_rels[0] and constr_rels[1]:
            if blocked_south(i, example, 0.75):
              bmy += 1
              bmyd += 1
              bmytd += 1
              bmyt += 1
            else:
              gmy += 1
              gmyd += 1
              gmytd += 1
              gmyt += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75):
              if blocked_south(i, example, 0.75):
                bmy += 1
                bmyd += 1
                bmytd += 1
                bmyt += 1
              else:
                gmy += 1
                gmyd += 1
                gmytd += 1
                gmyt += 1
            else:
              bmy += 1
              bmyd += 1
              bmytd += 1
              bmyt += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75):
              bmy += 1
              bmyd += 1
              bmytd += 1
              bmyt += 1
            else:
              gmy += 1
              gmyd += 1
              gmytd += 1
              gmyt += 1
          
      # case north
      elif y>0:
        if blocked_north(i, example, y):
          interY += 1
          interYu += 1
          interYtu += 1
        if not (triangle_rels[0] or triangle_rels[1]):
          bmy += 1
          bmyu += 1
          bmytu += 1
          bmyt += 1
        elif triangle_rels[0] and triangle_rels[1]:
          if blocked_north(i, example, 0.75):
            bmy += 1
            bmyu += 1
            bmytu += 1
            bmyt += 1
          else:
            gmy += 1
            gmyu += 1
            gmytu += 1
            gmyt += 1
        elif triangle_rels[0]:
          if blocked_left(i, example, 0.75):
            if blocked_north(i, example, 0.75):
              bmy += 1
              bmyu += 1
              bmytu += 1
              bmyt += 1
            else:
              gmy += 1
              gmyu += 1
              gmytu += 1
              gmyt += 1
          else:
            bmy += 1
            bmyu += 1
            bmytu += 1
            bmyt += 1
        elif triangle_rels[1]: 
          if blocked_north(i, example, 0.75):
            bmy += 1
            bmyu += 1
            bmytu += 1
            bmyt += 1
          else:
            gmy += 1
            gmyu += 1
            gmytu += 1
            gmyt += 1
    #SQUARE  
    elif example.init_state.objects[i].shape == 'square':
      square_rels = filter_rels_2D('square', rels, example, i)
      constr_rels = filter_constr_rels_2D('square', rels, example, i)
      # case zero x
      if abs(x) < 0.1:
        if constrained:
          if not (constr_rels[0] or constr_rels[1]):
            gzx += 1
            gzxs += 1
          elif constr_rels[0] and constr_rels[1]:
            if blocked_right(i, example, 0.75):
              gzx += 1
              gzxs += 1
            else:
              bzx += 1
              bzxs += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75):
              gzx += 1
              gzxs += 1
            else:
              bzx += 1
              bzxs += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75):
              if blocked_right(i, example, 0.75):
                gzx += 1
                gzxs += 1
              else:
                bzx += 1
                bzxs += 1
            else:
              gzx += 1
              gzxs += 1
        else:
          gzx += 1
          gzxs += 1
      # case left
      elif x<0:
        bmx += 1
        bmxl += 1
        bmxsl += 1
        bmxs += 1
        if blocked_left(i, example, x):
          interX += 1
          interXl += 1
          interXsl += 1
      # case right
      elif x>0:
        if blocked_right(i, example, x):
          interX += 1
          interXr += 1
          interXsr += 1
        if not constrained:
          bmx += 1
          bmxr += 1
          bmxsr += 1
          bmxs += 1
        else:
          if not (constr_rels[0] or constr_rels[1]):
            bmx += 1
            bmxr += 1
            bmxsr += 1
            bmxs += 1
          elif constr_rels[0] and constr_rels[1]:
            if blocked_right(i, example, 0.75):
              bmx += 1
              bmxr += 1
              bmxsr += 1
              bmxs += 1
            else:
              gmx += 1
              gmxr += 1
              gmxsr += 1
              gmxs += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75):
              bmx += 1
              bmxr += 1
              bmxsr += 1
              bmxs += 1
            else:
              gmx += 1
              gmxr += 1
              gmxsr += 1
              gmxs += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75):
              if blocked_right(i, example, 0.75):
                bmx += 1
                bmxr += 1
                bmxsr += 1
                bmxs += 1
              else:
                gmx += 1
                gmxr += 1
                gmxsr += 1
                gmxs += 1
            else:
              bmx += 1
              bmxr += 1
              bmxsr += 1
              bmxs += 1
          
      # case zero y
      if abs(y) < 0.1:
        if constrained:
          if not (constr_rels[0] or constr_rels[1]):
            gzy += 1
            gzys += 1
          elif constr_rels[0] and constr_rels[1]:
            if blocked_south(i, example, 0.75):
              gzy += 1
              gzys += 1
            else:
              bzy += 1
              bzys += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75):
              if blocked_south(i, example, 0.75):
                gzy += 1
                gzys += 1
              else:
                bzy += 1
                bzys += 1
            else:
              gzy += 1
              gzys += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75):
              gzy += 1
              gzys += 1
            else:
              bzy += 1
              bzys += 1
        else:
          gzy += 1
          gzys += 1
      # case south
      elif y<0:
        if blocked_south(i, example, y):
          interY += 1
          interYd += 1
          interYsd += 1
        if not constrained:
          bmy += 1
          bmyd += 1
          bmysd += 1
          bmys += 1
        else:
          if not (constr_rels[0] or constr_rels[1]):
            bmy += 1
            bmyd += 1
            bmysd += 1
            bmys += 1
          elif constr_rels[0] and constr_rels[1]:
            if blocked_south(i, example, 0.75):
              bmy += 1
              bmyd += 1
              bmysd += 1
              bmys += 1
            else:
              gmy += 1
              gmyd += 1
              gmysd += 1
              gmys += 1
          elif constr_rels[0]:
            if blocked_right(i, example, 0.75):
              if blocked_south(i, example, 0.75):
                bmy += 1
                bmyd += 1
                bmysd += 1
                bmys += 1
              else:
                gmy += 1
                gmyd += 1
                gmysd += 1
                gmys += 1
            else:
              bmy += 1
              bmyd += 1
              bmysd += 1
              bmys += 1
          elif constr_rels[1]:
            if blocked_south(i, example, 0.75):
              bmy += 1
              bmyd += 1
              bmysd += 1
              bmys += 1
            else:
              gmy += 1
              gmyd += 1
              gmysd += 1
              gmys += 1
      # case north
      elif y>0:
        bmy += 1
        bmyu += 1
        bmysu += 1
        bmys += 1
        if blocked_north(i, example, y):
          interY += 1
          interYu += 1
          interYsu += 1
    
    '''blocked = ''
    if blocked_left(i,example,0.75):
      blocked+='left,'
    if constrained and blocked_right(i,example,0.75):
      blocked+='right,'
    if constrained and blocked_south(i,example,0.75):
      blocked+='south,'
    if blocked_north(i,example,0.75):
      blocked+='north,'
    print(i,x,y,blocked)'''
  #print("")
  #draw_example(example)
  return gmx,gmy,bmx,bmy,gzx,gzy,bzx,bzy,interX,interY,gmxc,gmxt,gmxs,bmxc,bmxt,bmxs,gmyc,gmyt,gmys,bmyc,bmyt,bmys, gzxc,gzxt,gzxs,bzxc,bzxt,bzxs,gzyc,gzyt,gzys,bzyc,bzyt,bzys, gmxl,gmxr,gmyu,gmyd,bmxl,bmxr,bmyu,bmyd, gmxcl,gmxcr,gmxtl,gmxtr,gmxsl,gmxsr,bmxcl,bmxcr,bmxtl,bmxtr,bmxsl,bmxsr, gmycu,gmycd,gmytu,gmytd,gmysu,gmysd,bmycu,bmycd,bmytu,bmytd,bmysu,bmysd, interXl,interXr,interXcl,interXcr,interXtl,interXtr,interXsl,interXsr, interYu,interYd,interYcu,interYcd,interYtu,interYtd,interYsu,interYsd

def test_for_goal_achieved_and_keeped_distance(displs, example, constrained):
  l,g,d,b,lc,lt,ls,bc,bt,bs,gc,gt,gs,dc,dt,ds,tot,z,zg,zb,zgc,zgt,zgs,zbc,zbt,zbs = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
  
  true_g, true_d, true_z = [], [], []
  almost_g, almost_d, almost_z = [],[],[]
  rels = example.init_state.get_relations()
  for i,displ in enumerate(displs):
    tot+=1
    #case circle
    if example.init_state.objects[i].shape == 'circle':
      #case no movement
      if abs(displ) < 0.2:
        z+=1
        circles_rels = filter_rels('circle', rels['left'][i], example, reverse=False)
        if not circles_rels or example.init_state.objects[i].heavy:
          zg += 1
          zgc += 1
          true_z += [0]
          almost_z += [displ]
        else:
          zb += 1
          zbc += 1
      #case left movement
      elif displ < 0:
        circles_rels = filter_rels('circle', rels['left'][i], example, reverse=False)
        if not circles_rels:
          b += 1
          bc +=1
          continue
        l+=1
        lc+=1
        circle = example.init_state.objects[i]
        left_obj = example.init_state.objects[circles_rels[-1]]
        if filter_less_then(circle.x + displ,left_obj.x):
          g += 1
          gc+=1
          true_g += [0]
          almost_g += [0]
        else:
          true_g += [0]
          almost_g += [left_obj.x - (circle.x + displ)]
        if less_then(circle.x + displ,left_obj.x):
          d += 1
          dc+=1
          true_d += [0]
          almost_d += [0]  
        else:
          true_d += [0]
          almost_d += [(left_obj.x - 0.5) - (circle.x + displ)]
      #case right movement
      else:
        b+=1
        bc+=1
    elif example.init_state.objects[i].shape == 'triangle':
      #case no movement
      if abs(displ) < 0.2:
        z+=1
        triangle_rels_left = filter_rels('triangle', rels['left'][i], example, reverse=False)
        triangle_rels_right = []
        if constrained:
          triangle_rels_right = filter_rels('triangle', rels['right'][i], example, reverse=True)
        if not (triangle_rels_left or (triangle_rels_right and constrained)) or example.init_state.objects[i].heavy:
          zg += 1
          zgt += 1
          true_z += [0]
          almost_z += [displ]
        else:
          zb += 1
          zbt += 1
      #case left movement
      elif displ < 0:
        triangle_rels = filter_rels('triangle', rels['left'][i], example, reverse=False)
        if not triangle_rels:
          b += 1
          bt += 1
          continue
        l+=1
        lt+=1
        triangle = example.init_state.objects[i]
        left_obj = example.init_state.objects[triangle_rels[-1]]
        if filter_less_then(triangle.x + displ,left_obj.x):
          g += 1
          gt+=1
          true_g += [0]
          almost_g += [0]
        else:
          true_g += [0]
          almost_g += [left_obj.x - (triangle.x + displ)]
        if less_then(triangle.x + displ,left_obj.x):
          d += 1
          dt+=1
          true_d += [0]
          almost_d += [0]  
        else:
          true_d += [0]
          almost_d += [(left_obj.x - 0.5) - (triangle.x + displ)]
      #case right movement constrained
      elif displ > 0 and constrained:
        triangle_rels = filter_rels('triangle', rels['right'][i], example, reverse=True)
        if not triangle_rels:
          b += 1
          bt+=1
          continue
        l+=1
        lt+=1
        triangle = example.init_state.objects[i]
        left_obj = example.init_state.objects[triangle_rels[-1]]
        if filter_less_then(left_obj.x, triangle.x + displ):
          g += 1
          gt+=1
          true_g += [0]
          almost_g += [0]
        else:
          true_g += [0]
          almost_g += [(triangle.x + displ) - left_obj.x]
        if less_then(left_obj.x, triangle.x + displ):
          d += 1
          dt+=1
          true_d += [0]
          almost_d += [0]
        else:
          true_d += [0]
          almost_d += [(triangle.x + displ - 0.5) - left_obj.x] 
      #case right movement not constrained
      else:
        b+=1
        bt+=1
    elif example.init_state.objects[i].shape == 'square':
      #case no movement
      if abs(displ) < 0.2:
        z+=1
        square_rels_left = filter_rels('square', rels['left'][i], example, reverse=False) # this will be empty
        square_rels_right = []
        if constrained:
          square_rels_right = filter_rels('square', rels['right'][i], example, reverse=True)
        if not (square_rels_left or (square_rels_right and constrained)) or example.init_state.objects[i].heavy:
          zg += 1
          zgs += 1
          true_z += [0]
          almost_z += [displ]
        else:
          zb += 1
          zbs += 1
      #case move right constrained
      elif displ > 0 and constrained:
        square_rels = filter_rels('square', rels['right'][i], example, reverse=True)
        if not square_rels:
          b += 1
          bs+=1
          continue
        l+=1
        ls+=1
        square = example.init_state.objects[i]
        left_obj = example.init_state.objects[square_rels[-1]]
        if filter_less_then(left_obj.x, square.x + displ):
          g += 1
          gs+=1
          true_g += [0]
          almost_g += [0]
        else:
          true_g += [0]
          almost_g += [(square.x + displ) - left_obj.x]
        if less_then(left_obj.x, square.x + displ):
          d += 1
          ds+=1 
          true_d += [0]
          almost_d += [0]
        else:
          true_d += [0]
          almost_d += [(square.x + displ - 0.5) - left_obj.x]
      #case move left
      else:
        b+=1
        bs+=1
    else:
      pass
  return l, g, d, b, true_g, true_d, almost_g, almost_d, true_z, almost_z,  lc,lt,ls,bc,bt,bs,gc,gt,gs,dc,dt,ds,tot,z,zg,zb,zgc,zgt,zgs,zbc,zbt,zbs

#circle left (s,t)| no move
#triangle left (s) | right (c)
#square no move | right (c,t)
def filter_rels(shape, rels, example, reverse=False):
  out = []
  #case circle
  if shape == 'circle':
    if not reverse:
      for i in rels:
        if example.init_state.objects[i].shape == 'triangle' or example.init_state.objects[i].shape == 'square':
          out += [i]
      out = sorted(out, key=lambda x: example.init_state.objects[x].x,reverse=reverse)
    else:
      pass#since circles never move right
  #case triangle
  elif shape == 'triangle':
    if not reverse:
      for i in rels:
        if example.init_state.objects[i].shape == 'square':
          out += [i]
      out = sorted(out, key=lambda x: example.init_state.objects[x].x,reverse=reverse)
    else:
      for i in rels:
        if example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy:
          out += [i]
      out = sorted(out, key=lambda x: example.init_state.objects[x].x,reverse=reverse)
  #case square
  elif shape == 'square':
    if not reverse:
      pass#since squares never move left
    else:
      for i in rels:
        if (example.init_state.objects[i].shape == 'triangle' and example.init_state.objects[i].heavy) or (example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy):
          out += [i]
      out = sorted(out, key=lambda x: example.init_state.objects[x].x,reverse=reverse)
  return out
  
def filter_rels_2D(shape, rels, example, index):
  out = []
  if shape == 'circle':
    tmp1 = []
    for i in rels['left'][index]:
      if example.init_state.objects[i].shape == 'triangle' or example.init_state.objects[i].shape == 'square':
        tmp1 += [i]
    out += [tmp1]
    tmp2 = []
    for i in rels['north'][index]:
      if example.init_state.objects[i].shape == 'triangle' or example.init_state.objects[i].shape == 'square':
        tmp2 += [i]
    out += [tmp2]
  #case triangle
  elif shape == 'triangle':
    tmp1 = []
    for i in rels['left'][index]:
      if example.init_state.objects[i].shape == 'square':
        tmp1 += [i]
    out += [tmp1]
    tmp2 = []
    for i in rels['north'][index]:
      if example.init_state.objects[i].shape == 'square':
        tmp2 += [i]
    out += [tmp2]
  #case square
  elif shape == 'square':
    out += [[],[]]
  return out   

def filter_constr_rels_2D(shape, rels, example, index):
  out = []
  if shape == 'circle':
    out += [[],[]]
  elif shape == 'triangle':
    tmp1 = []
    for i in rels['right'][index]:
      if (example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy):
        tmp1 += [i]
    out += [tmp1]
    tmp2 = []
    for i in rels['south'][index]:
      if (example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy):
        tmp2 += [i]
    out += [tmp2]
  elif shape == 'square':
    tmp1 = []
    for i in rels['right'][index]:
      if (example.init_state.objects[i].shape == 'triangle' and example.init_state.objects[i].heavy) or (example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy):
        tmp1 += [i]
    out += [tmp1]
    tmp2 = []
    for i in rels['south'][index]:
      if (example.init_state.objects[i].shape == 'triangle' and example.init_state.objects[i].heavy) or (example.init_state.objects[i].shape == 'circle' and example.init_state.objects[i].heavy):
        tmp2 += [i]
    out += [tmp2]
  return out
  
def blocked_left(i,example,displ):
  state = example.init_state.copy()
  if not state.apply_move_test(move_left,i,displ):
    return True
  else:
    return False 
def blocked_right(i,example,displ):
  state = example.init_state.copy()
  if not state.apply_move_test(move_right,i,displ):
    return True
  else:
    return False
def blocked_north(i,example,displ):
  state = example.init_state.copy()
  if not state.apply_move_test(move_up,i,displ):
    return True
  else:
    return False
def blocked_south(i,example,displ):
  state = example.init_state.copy()
  if not state.apply_move_test(move_down,i,displ):
    return True
  else:
    return False 
   
def move_left(obj,amount):
    obj.x -= amount #+ uniform(-0.25,0.25)
def move_right(obj,amount):
  obj.x += amount #+ uniform(-0.25,0.25)
def move_up(obj,amount):
  obj.y += amount #+ uniform(-0.25,0.25)
def move_down(obj,amount):
  obj.y -= amount #+ uniform(-0.25,0.25) 
  
  
  
  
  
'''
triangle abs(x)<0.1:

if not (constr_rels[0] or constr_rels[1]): # no south, right obj
  gzx += 1
elif constr_rels[0] and constr_rels[1]: 
  if blocked_right(i, example, 0.75): # south, right obj and blocked r
    gzx += 1
  else: # south, right obj and not blocked right
    bzx += 1
elif constr_rels[0]:
  if blocked_right(i, example, 0.75): # right obj and blocked right
    gzx += 1
  else: # right obj and not blocked right
    bzx += 1
elif constr_rels[1]:
  if blocked_south(i, example, 0.75): # south obj and blocked south
    if blocked_right(i, example, 0.75): # blocked right
      gzx += 1
    else: # not blocked right
      bzx += 1
  else: # south obj and not blocked south
    gzx += 1
    
tirangle abs(y)<0.1:

if not (constr_rels[0] or constr_rels[1]):
  gzy += 1
elif constr_rels[0] and constr_rels[1]:
  if blocked_south(i, example, 0.75):
    gzy += 1
  else:
    bzy += 1
elif constr_rels[0]:
  if blocked_right(i, example, 0.75):
    if blocked_south(i, example, 0.75):
      gzy += 1
    else:
      bzy += 1
  else:
    gzy += 1
elif constr_rels[1]:
  if blocked_south(i, example, 0.75):
    gzy += 1
  else:
    bzy += 1
'''  
  
'''gm = {'all':(mgood_moves,stdgood_moves),'circ':(mgmc,stdgmc),'tri':(mgmt,stdgmt),'sqr':(mgms,stdgms)}
  gz = {'all':(mgood_zeros,stdgood_zeros),'circ':(mgzc,stdgzc),'tri':(mgzt,stdgzt),'sqr':(mgzs,stdgzs)}
  
  goal = {'all':(msubgoal,stdsubgoal),'circ':(msgc,stdsgc),'tri':(msgt,stdsgt),'sqr':(msgs,stdsgs)}
  
  constr = {'all':(mconstr,stdconstr),'circ':(mcc,stdcc),'tri':(mct,stdct),'sqr':(mcs,stdcs)}
  
  nrm = {'subg':(mrmseg,stdrmseg) ,'constr':(mrmsed,stdrmsed)}
  
  roc = {'all':(mg_auc,stdg_auc),'circ':(mg_aucc,stdg_aucc),'tri':(mg_auct,stdg_auct),'sqr':(mg_aucs,stdg_aucs)}
  
  pr = {'all':(mg_pr,stdg_pr),'circ':(mg_prc,stdg_prc),'tri':(mg_prt,stdg_prt),'sqr':(mg_prs,stdg_prs)}
  
  return {'move':gm,'zero':gz,'goal':goal,'constr':constr,'roc':roc,'pr':pr,'nrm':nrm}'''  
  
  
  
if __name__ == '__main__':

  res_dir = 'results/' 
  #scens = ['flat_noise_%s/simple/','hier_noise_%s/simple/','flat_noise_%s/constrained/','hier_noise_%s/constrained/']
  res_name = 'results.pkl'
  
  '''out = {}
  for noise in [0.05,0.1,0.2,0.5,0.7]:
    tmp = {}
    for scen in scens:
      with open(res_dir+scen % (str(noise)) +res_name,'rb') as f:
        acc_res = pickle.load(f)
      
      twoD = True if '2D' in scen else False
      name = '{}|{}'.format('flat' if 'flat' in scen else 'hier', 'simple' if 'simple' in scen else 'constrained')
      tmp[name] = overall_accuracy_test(acc_res,twoD)
    out[str(noise)] = tmp  
  print('######################################')
  print('######################################')
  print('######################################')
  print('######################################')
  print('######################################')  
  pprint(out)  
  for table in ['goal','constr']:
    print('Table ',table)
    for noise in [0.05,0.1,0.2,0.5,0.7]:
      print('Noise ',str(noise))
      for row in ['all','circ','tri','sqr']:
        #for name in ['flat|simple','hier|simple','flat|constrained','hier|constrained']:
        print('{} & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$'.format(row, round(out[str(noise)]['flat|simple'][table][row][0],2) , round(out[str(noise)]['flat|simple'][table][row][1],2), round(out[str(noise)]['hier|simple'][table][row][0],2), round(out[str(noise)]['hier|simple'][table][row][1],2), round(out[str(noise)]['flat|constrained'][table][row][0],2), round(out[str(noise)]['flat|constrained'][table][row][1],2), round(out[str(noise)]['hier|constrained'][table][row][0],2), round(out[str(noise)]['hier|constrained'][table][row][1],2)))
      print()
    print()'''          
    
  '''for table in ['nrm']:
    print('Table ',table)
    for noise in [0.05,0.1,0.2,0.5,0.7]:
      print('Noise ',str(noise))
      for row in ['subg','constr']:
        #for name in ['flat|simple','hier|simple','flat|constrained','hier|constrained']:
        print('{} & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$'.format(row, round(out[str(noise)]['flat|simple'][table][row][0],2) , round(out[str(noise)]['flat|simple'][table][row][1],2), round(out[str(noise)]['hier|simple'][table][row][0],2), round(out[str(noise)]['hier|simple'][table][row][1],2), round(out[str(noise)]['flat|constrained'][table][row][0],2), round(out[str(noise)]['flat|constrained'][table][row][1],2), round(out[str(noise)]['hier|constrained'][table][row][0],2), round(out[str(noise)]['hier|constrained'][table][row][1],2)))
      print()
    print()'''
    
    
  '''scens = ['hier/constrained/','reusehier/constrained/']  
  out = {}
  for scen in scens:
    with open(res_dir+scen +res_name,'rb') as f:
      acc_res = pickle.load(f)
    
    twoD = True if '2D' in scen else False
    name = '{}'.format('reuse' if 'reuse' in scen else 'hier')
    out[name] = overall_accuracy_test(acc_res,twoD)
  print('######################################')
  print('######################################')
  print('######################################')
  print('######################################')
  print('######################################')  
  pprint(out)  
  for table in ['move','zero','goal','constr','roc','pr']:
    print('Table ',table)
    for row in ['all','circ','tri','sqr']:
      #for name in ['flat|simple','hier|simple','flat|constrained','hier|constrained']:
      print('{} & ${} \pm {}$ & ${} \pm {}$'.format(row, round(out['hier'][table][row][0],2) , round(out['hier'][table][row][1],2), round(out['reuse'][table][row][0],2), round(out['reuse'][table][row][1],2)))
    print()
    
  for table in ['nrm']:
    print('Table ',table)
    for row in ['subg','constr']:
      #for name in ['flat|simple','hier|simple','flat|constrained','hier|constrained']:
      print('{} & ${} \pm {}$ & ${} \pm {}$'.format(row, round(out['hier'][table][row][0],2) , round(out['hier'][table][row][1],2), round(out['reuse'][table][row][0],2), round(out['reuse'][table][row][1],2)))
    print()'''
    
  times = []    
  scens = ['reuseflat/simple/','reusehier/simple/','reuseflat/constrained/','reusehier/constrained/']
  
  out = {}
  for scen in scens:
    with open(res_dir+scen +'times.pkl','rb') as f:
      tmp = pickle.load(f)
      print(tmp)
    mtmp, stdtmp = np.mean(tmp), np.std(tmp)
    out[scen] = (mtmp,stdtmp)  
  print("times & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$".format(round(out['reuseflat/simple/'][0],2),round(out['reuseflat/simple/'][1],2),round(out['reusehier/simple/'][0],2),round(out['reusehier/simple/'][1],2),round(out['reuseflat/constrained/'][0],2),round(out['reuseflat/constrained/'][1],2),round(out['reusehier/constrained/'][0],2),round(out['reusehier/constrained/'][1],2)))  
  
        
