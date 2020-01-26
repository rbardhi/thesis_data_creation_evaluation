%%% -*- Mode: Prolog; -*-

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


n(W) ~ val(_) :=true.
posX_t0(W,I) ~ val(_) :=true.
posY_t0(W,I) ~ val(_) :=true. 
shape(W,I) ~ val(_) :=true.
heavy(W,I) ~ val(_) := true.
left_of(W,I,I1) ~ val(_) :=true.
l_o_shape(W,I1,I2) ~ val(Sh) :=
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


displX(W,Id) ~ gaussian(0.0,0.0) := atleastOne(W,Id)~=Sh_M,Sh_M==square,shape(W,Id)~=Sh_M_1,Sh_M_1==square.
displX(W,Id) ~ gaussian(Mean,1.0301193014161734) := atleastOne(W,Id)~=Sh_M,Sh_M==square,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M)~=X_M_1,getMean([X_M,X_M_1],[-0.963738960305701,0.8324080383906587,-0.06446325150632104],Mean).
displX(W,Id) ~ gaussian(Mean,0.626979705447725) := atleastOne(W,Id)~=Sh_M,Sh_M==square,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M)~=X_M_1,getMean([X_M,X_M_1],[-1.174178755153761,0.8697812365212658,0.26515508753766737],Mean).
displX(W,Id) ~ gaussian(0.0,0.0) := atleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==square.
displX(W,Id) ~ gaussian(Mean,1.028426396273128) := atleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M_1)~=X_M_1,getMean([X_M,X_M_1],[-0.5646883142626485,0.3213564249376087,0.39206608571860146],Mean).
displX(W,Id) ~ gaussian(Mean,0.7089460787046407) := atleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M)~=X_M_1,getMean([X_M,X_M_1],[-0.8217981763364106,0.4963684095940929,0.006273754850441682],Mean).
displX(W,Id) ~ gaussian(0.0,0.0) := atleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==square.
displX(W,Id) ~ gaussian(Mean,0.44768359179584605) := atleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M)~=X_M_1,getMean([X_M,X_M_1],[-0.58897719937849,0.5300004346175837,0.5709963857602371],Mean).
displX(W,Id) ~ gaussian(Mean,0.6753904524759826) := atleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,posX_t0(W,Id)~=X_M,avglshpos(W,Id,Sh_M_1)~=X_M_1,getMean([X_M,X_M_1],[-0.7402030624198808,0.11854569878902588,0.787928774865615],Mean).
displX(W,Id) ~ gaussian(0.0,0.0) := \+atleastOne(W,Id)~=Sh_M.
displX(W,Id) ~ gaussian(-0.455060642409,0.803698597222) := true.
posX_t1(W,Id) ~ gaussian(Mean,0.0024684103123171065) := posX_t0(W,Id)~=X_M,displX(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9986376618091957,0.9968673173771698,0.0024028439577334915],Mean).
posX_t1(W,Id) ~ gaussian(2.24613345628,1.2407163659) := true.
posY_t1(W,Id) ~ gaussian(Mean,7.898411595082004e-31) := posY_t0(W,Id)~=X_M,getMean([X_M],[1.0000000000000007,-1.7763568394002505e-15],Mean).
posY_t1(W,Id) ~ gaussian(2.49358807333,1.80277612471) := true.
