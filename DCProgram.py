import numpy as np

#declare directories to store programs
#this class should: 
# get all rules
# devide them into posX_t1,posY_t1,displ,move_left_of rules
# depending on prediction step write to file the relevant rules

rules_dir = '../probabilistic-dc-learner/data/cross_validation/'

class DCProgram:
  def __init__(self, rules_file,scen):
    self.posX_t1 = []
    self.posY_t1 = []
    self.displX = []
    self.displY = []
    self.move_left_of = []
    self.move_north_of = []
    with open(rules_dir+scen+rules_file,'r') as f:
      rules = f.readlines()
    for rule in rules:
      if rule.find('posX_t1') == 0:
        self.posX_t1 += [rule]
      elif rule.find('posY_t1') == 0:
        self.posY_t1 += [rule]
      elif rule.find('displX') == 0:
        self.displX += [rule]
      elif rule.find('displY') == 0:
        self.displY += [rule]
    
    '''if 'hier' in scen and 'noise' in scen:
      constrained = 'simple'
      if 'constrained' in scen:
        constrained = 'constrained'
      twoD = ''
      if '2D' in scen:
        twoD = '2D_'
      with open(rules_dir+twoD+'hier/'+constrained+'/'+rules_file,'r') as f:
        rules = f.readlines()
      for rule in rules:
        if rule.find('move_left_of') == 0:
          self.move_left_of += [rule]
        elif rule.find('move_north_of') == 0:
          self.move_north_of += [rule]'''
          
    if 'reusehier' in scen:      
      with open(rules_dir+'hier/simple/'+rules_file,'r') as f:
        rules = f.readlines()
      for rule in rules:
        if rule.find('move_left_of') == 0:
          self.move_left_of += [rule]
        elif rule.find('move_north_of') == 0:
          self.move_north_of += [rule]      
          
    elif 'hier' in scen:    
      with open(rules_dir+scen+rules_file,'r') as f:
        rules = f.readlines()
      for rule in rules:
        if rule.find('move_left_of') == 0:
          self.move_left_of += [rule]
        elif rule.find('move_north_of') == 0:
          self.move_north_of += [rule]
    
  def get_initialization(self,flag):
    out = get_dc_initialization()
    out += get_priors()
    if flag:
      out += get_displ_prior()
    out += get_background_knowledge()
    return out
   
  def get_program(self,step):
    if step == 0:
      return self.get_program_for_step_0()
    elif step == 1:
      return self.get_program_for_step_1()
    elif step == 2:
      return self.get_program_for_step_2()
    else:
      print("Error at get_program in DCProgram")
      return None  
    
  def get_program_for_step_0(self):
    out = self.get_initialization(True)
    for rules in [self.move_left_of,self.posX_t1,self.posY_t1]:
      for rule in rules:
        out += rule
    return out
    
  def get_program_for_step_1(self):
    out = self.get_initialization(False)
    for rules in [self.move_left_of,self.move_north_of,self.displX,self.displY,self.posX_t1,self.posY_t1]:
      for rule in rules:
        out += rule
    return out
    
  #think if we should include a prior for posX_t1
  def get_program_for_step_2(self):
    out = self.get_initialization(False)
    for rules in [self.move_left_of,self.displX,self.posX_t1,self.posY_t1]:
      for rule in rules:
        out += rule
    return out
    
def get_priors():
  return '''n(W) ~ val(_) :=true.
posX_t0(W,I) ~ val(_) :=true.
posY_t0(W,I) ~ val(_) :=true. 
shape(W,I) ~ val(_) :=true.
size(W,I) ~ val(_) :=true.
heavy(W,I) ~ val(_) := true.
left_of(W,I,I1) ~ val(_) :=true.
north_of(W,I,I1) ~ val(_) :=true.\n'''

def get_displ_prior():
  return '''displX(W,I) ~ val(_) := true.
displY(W,I) ~ val(_) := true.\n\n'''

def get_background_knowledge():
  return '''l_o_shape(W,I1,I2) ~ val(Sh) :=
  left_of(W,I1,I2) ~= true,
  shape(W,I2) ~= Sh.
l_o_posX_t0(W,I1,I2) ~ val(X) :=
  left_of(W,I1,I2) ~= true,
  posX_t0(W,I2) ~= X. 
  
atleastOne(W,I) ~ val(Sh) :=
  findall_forward(S,(left_of(W,I,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).  
  
latleastOne(W,I) ~ val(Sh) :=
  findall_forward(S,(left_of(W,I,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).
  
ratleastOne(W,I) ~ val(Sh) :=
  findall_forward(S,(left_of(W,I1,I)~=true,heavy(W,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).
  
mmshl(W,I) ~ val(Sh) :=
  findall_forward(X,(left_of(W,I,Id1)~=true,shape(W,Id1)~=X),L),
  maxMod(L)~=Sh.
mmshr(W,I) ~ val(Sh) :=
  findall_forward(X,(left_of(W,Id1,I)~=true,shape(W,Id1)~=X),L),
  maxMod(L)~=Sh.
  
avglshpos(W,I,Sh) ~ val(XM) :=
  findall_forward(X, (l_o_shape(W,I,I1)~=Sh,l_o_posX_t0(W,I,I1)~=X),L),
  avg(L)~=XM.
  
avgrshpos(W,I,Sh) ~ val(XM) :=
  findall_forward(X, (l_o_shape(W,I1,I)~=Sh,heavy(W,I1)~=true,l_o_posX_t0(W,I1,I)~=X),L),
  avg(L)~=XM.
  
all_combined(W,I) ~ val(XM) :=
  findall_forward(X,(left_of(W,I,Id1)~=true,shape(W,Id1)~=X),L),
  maxMod(L)~=Sh,
  findall_forward(X, (l_o_shape(W,I,I1)~=Sh,l_o_posX_t0(W,I,I1)~=X),L1),
  avg(L1)~=XM.
  
almove_left_of(W,I) ~ val(X) :=
  findall_forward(S,(move_left_of(W,I1,I)~=true,posX_t0(W,I1)~=S),L),
  avg(L)~=X.
  
armove_left_of(W,I) ~ val(X) :=
  findall_forward(S,(move_left_of(W,I,I1)~=true,heavy(W,I1)~=true,posX_t0(W,I1)~=S),L),
  avg(L)~=X.
  
atleastOneLeft(W,I) ~ val(Sh) :=
  findall_forward(S,(left_of(W,I,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).
  
atleastOneRight(W,I) ~ val(Sh) :=
  findall_forward(S,(left_of(W,I1,I)~=true,heavy(W,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).
  
atleastOneNorth(W,I) ~ val(Sh) :=
  findall_forward(S,(north_of(W,I,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).
  
atleastOneSouth(W,I) ~ val(Sh) :=
  findall_forward(S,(north_of(W,I1,I)~=true,heavy(W,I1)~=true,shape(W,I1)~=S),L),member(Sh,L).

blocked_left(W,I) ~ val(B) :=
  posX_t0(W,I)~=X,posY_t0(W,I)~=Y,size(W,I)~=S,
  findall_forward(I1,(left_of(W,I,I1)~=true,posX_t0(W,I1)~=X1,posY_t0(W,I1)~=Y1,size(W,I1)~=S1,sqrt(((X-0.75) - X1)**2 + ((Y+0) - Y1)**2) =< S + S1),L),
  (member(Q,L) -> B = true ; B = false).

blocked_right(W,I,B) ~ val(B) :=
  posX_t0(W,I)~=X,posY_t0(W,I)~=Y,size(W,I)~=S,
  findall_forward(I1,(left_of(W,I1,I)~=true,posX_t0(W,I1)~=X1,posY_t0(W,I1)~=Y1,size(W,I1)~=S1,sqrt(((X+0.75) - X1)**2 + ((Y+0) - Y1)**2) =< S + S1),L),
  (member(Q,L) -> B = true ; B = false).

blocked_south(W,I) ~ val(B) :=
  posX_t0(W,I)~=X,posY_t0(W,I)~=Y,size(W,I)~=S,
  findall_forward(I1,(north_of(W,I1,I)~=true,posX_t0(W,I1)~=X1,posY_t0(W,I1)~=Y1,size(W,I1)~=S1,sqrt(((X+0) - X1)**2 + ((Y-0.75) - Y1)**2) =< S + S1),L),
  (member(Q,L) -> B = true ; B = false).

blocked_north(W,I) ~ val(B) :=
  posX_t0(W,I)~=X,posY_t0(W,I)~=Y,size(W,I)~=S,
  findall_forward(I1,(north_of(W,I,I1)~=true,posX_t0(W,I1)~=X1,posY_t0(W,I1)~=Y1,size(W,I1)~=S1,sqrt(((X+0) - X1)**2 + ((Y+0.75) - Y1)**2) =< S + S1),L),
  (member(Q,L) -> B = true ; B = false).
  
mlo(W,I) ~ val(S) := findall_forward(B,move_left_of(W,_,I)~=B,L), (member(true,L) -> S = true ; S = false).
mno(W,I) ~ val(S) := findall_forward(B,move_north_of(W,_,I)~=B,L), (member(true,L) -> S = true ; S = false).
mro(W,I) ~ val(S) := findall_forward(B,(move_left_of(W,I,I1)~=B,heavy(W,I1)~=true),L), (member(true,L) -> S = true ; S = false).
mso(W,I) ~ val(S) := findall_forward(B,(move_north_of(W,I,I1)~=B,heavy(W,I1)~=true),L), (member(true,L) -> S = true ; S = false).

/*case(W,I) ~ val(C) :=
  mlo(W,I)~=true,mno(W,I)~=true,
  C = left_north.
case(W,I) ~ val(C) :=
  mlo(W,I)~=true,mno(W,I)~=false,
  C = left.
case(W,I) ~ val(C) :=
  mno(W,I)~=true,mlo(W,I)~=false,
  C = north.*/\n\n\n'''

def get_dc_initialization():
  return '''%%% -*- Mode: Prolog; -*-

:- use_module('../../../../DC/dcpf.pl').
:- use_module('../../../../DC/random/sampling.pl').
:- use_module('../../../../DC/distributionalclause.pl').
:- use_module(library(lists)).
:- set_options(default).
:- initialization(init).

builtin(min_list(_,_)).
builtin(max_list(_,_)).
builtin(lmindc(_,_)).
builtin(lmaxdc(_,_)).
builtin(length(_,_)).
builtin(listavgdc(_,_,_)).
builtin(getMean(_,_,_)).
builtin(logistic(_,_,_)).
builtin(softmax(_,_,_)).
builtin(oneElementOfListDC(_,_)).
builtin((_->_;_)).

getMean(X,P,Mean) :- dotProd(X,P,Mean).
dotProd([H1|T1], [H2|T2], Prod) :- dotProd(T1, T2, PartProd), Prod is H1*H2 + PartProd.
dotProd([], [H2], Prod) :- Prod is H2.

logistic(P,X,Result) :- dotProd(X,P,Mean), Result is 1/(1 + exp(-1*Mean)).

softmax(P,X,Result) :- calExponent(P,X,Sum,List), normalization(Sum, List, Result).

normalization(Sum, [X|Xs], [Y|Ys]) :- Y is X/Sum, normalization(Sum, Xs, Ys).
normalization(_, [], []).

calExponent([H|T], X, Sum, [Exp|Exps]) :-
    calExponent([H|T], X, 0, Sum, [Exp|Exps]).
calExponent([H|T], X, Acc0, Sum, [Exp|Exps]) :-
    dotProd(X,H,Prod),
    Exp is exp(Prod),
    Acc is Acc0 + Exp,
    calExponent(T, X, Acc, Sum, Exps).
calExponent([], _, Sum, Sum, []).

cnt(P) ~ val(Count) := length(P,Count).

pickOneElement(List) ~ uniform(List).
%oneElementOfListDC([H|_], X) :- X = H.
%oneElementOfListDC([H|T], X) :- T=[] -> X = H; oneElementOfListDC(T, X).
%pickOneElement(List) ~ val(X) := oneElementOfListDC(List, X).

lmaxdc(L, M) :- lmaxdc(L, [], [], M).
lmaxdc([], Seen, MMax, Max) :- MMax=[] -> Max=Seen; lmaxdc(MMax, [], [], Max).
lmaxdc([H|T], Seen, MMax, Max) :- (member(H, Seen) -> lmaxdc(T, Seen, [H|MMax], Max); lmaxdc(T, [H|Seen], MMax, Max)).
maxMod(L) ~ val(Max) := lmaxdc(L, Max1), pickOneElement(Max1) ~= Max.

lmindc(L, M) :- lmindc(L, [], [], M).
lmindc([], Seen, MMin, Min) :- MMin=[] -> Min=Seen, !; leftoverdc(Seen, MMin, [], Min).
lmindc([H|T], Seen, Left, Min) :- (member(H, Seen) -> lmindc(T, Seen, [H|Left], Min); lmindc(T, [H|Seen], Left, Min)).
leftoverdc([], MMin, TMin, Min) :- TMin=[] -> lmindc(MMin, [], [], Min); Min=TMin, !.
leftoverdc([H|Seen], MMin, TMin, Min) :- (member(H, MMin)) -> leftoverdc(Seen, MMin, TMin, Min); leftoverdc(Seen, MMin, [H|TMin], Min).
minMod(L) ~ val(Min) := lmindc(L, Min1), pickOneElement(Min1) ~= Min.

max(L) ~ val(Max) := max_list(L, Max).

min(L) ~ val(Min) := min_list(L, Min).

listavgdc(L, C, A) :- C =:= 0 -> false; sum_list(L, Sum), A is Sum / C.
avg(L) ~ val(Avg) := length(L,Cnt), listavgdc(L, Cnt, Avg).

samplesGenerator(Num, Evidence, Query, Var, SampleList) :- findall(Var, genSample(Num, Evidence, Query, World), SampleList).
genSample(Num, Evidence, Query, World) :- between(1, Num, SID), generate_backward(Evidence, Query, World).
\n\n'''
