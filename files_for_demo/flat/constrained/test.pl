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


displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(Mean,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==square,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==square,getMean([X_M],[0.0,0.0],Mean).
displX(W,Id) ~ gaussian(Mean,0.18842481406769682) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==square,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==triangle,getMean([X_M],[-0.1393443514558752,0.5217359052188577],Mean).
displX(W,Id) ~ gaussian(Mean,0.4053126951059915) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==square,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==circle,getMean([X_M],[-0.3989891813006071,1.4445932453853922],Mean).
displX(W,Id) ~ gaussian(Mean,0.7207289748328949) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==square,\+latleastOne(W,Id)~=Sh_M_2,getMean([X_M],[-0.8621180908999413,2.3209163177949033],Mean).
displX(W,Id) ~ gaussian(Mean,0.6157510111760308) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,getMean([X_M],[-0.5436640478654818,0.969848474079859],Mean).
displX(W,Id) ~ gaussian(Mean,0.5502169777967825) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==square,getMean([X_M],[-0.7506375250826114,0.7100026673315276],Mean).
displX(W,Id) ~ gaussian(Mean,0.43977686218535605) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==triangle,getMean([X_M],[-0.46558119054454133,-0.057771806227776334],Mean).
displX(W,Id) ~ gaussian(Mean,0.4911664611224038) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==circle,getMean([X_M],[-0.6630634938037319,0.7403621644649961],Mean).
displX(W,Id) ~ gaussian(Mean,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==square,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,\+latleastOne(W,Id)~=Sh_M_2,getMean([X_M],[-0.0,0.0],Mean).
displX(W,Id) ~ gaussian(Mean,0.8589897087670557) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==square,posX_t0(W,Id)~=X_M,getMean([X_M],[-0.6064261037770422,2.448617045638411],Mean).
displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,heavy(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,heavy(W,Id)~=B_M,B_M==false,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==square.
displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,heavy(W,Id)~=B_M,B_M==false,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displX(W,Id) ~ gaussian(Mean,0.22776424843501994) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,heavy(W,Id)~=B_M,B_M==false,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==circle,posX_t0(W,Id)~=X_M,getMean([X_M],[-0.22532652134914236,0.7232877015837779],Mean).
displX(W,Id) ~ gaussian(0.473156315463,0.754443165262) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,heavy(W,Id)~=B_M,B_M==false,\+latleastOne(W,Id)~=Sh_M_2.
displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,heavy(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(Mean,0.4128601116189788) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==square,getMean([X_M],[-0.6509518965412341,0.3199637955884753],Mean).
displX(W,Id) ~ gaussian(Mean,0.4340197948576833) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==triangle,getMean([X_M],[-0.6866359567411798,0.41156439697675684],Mean).
displX(W,Id) ~ gaussian(Mean,0.5211628543658151) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==circle,getMean([X_M],[-0.6992469969186833,0.8121701471938607],Mean).
displX(W,Id) ~ gaussian(Mean,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,heavy(W,Id)~=B_M,B_M==false,posX_t0(W,Id)~=X_M,\+latleastOne(W,Id)~=Sh_M_2,getMean([X_M],[-0.0,0.0],Mean).
displX(W,Id) ~ gaussian(Mean,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==square,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==square,getMean([X_M],[-0.0,0.0],Mean).
displX(W,Id) ~ gaussian(Mean,0.6374536250596426) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==square,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==triangle,getMean([X_M],[-0.012796208366186309,1.4171525503284639],Mean).
displX(W,Id) ~ gaussian(Mean,0.6418123462089034) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==square,posX_t0(W,Id)~=X_M,latleastOne(W,Id)~=Sh_M_2,Sh_M_2==circle,getMean([X_M],[-0.46792160953939677,2.2926059405363324],Mean).
displX(W,Id) ~ gaussian(Mean,0.775517209179852) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==square,posX_t0(W,Id)~=X_M,\+latleastOne(W,Id)~=Sh_M_2,getMean([X_M],[-0.804459656090194,2.949448202119721],Mean).
displX(W,Id) ~ gaussian(Mean,0.8871301801274153) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,posX_t0(W,Id)~=X_M,getMean([X_M],[-0.6362222245331296,2.501518843432941],Mean).
displX(W,Id) ~ gaussian(0.0,0.0) := ratleastOne(W,Id)~=Sh_M,Sh_M==circle,shape(W,Id)~=Sh_M_1,Sh_M_1==circle.
displX(W,Id) ~ gaussian(0.0,0.0) := \+ratleastOne(W,Id)~=Sh_M,heavy(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(0.0,0.0) := \+ratleastOne(W,Id)~=Sh_M,heavy(W,Id)~=B_M,B_M==false,shape(W,Id)~=Sh_M_1,Sh_M_1==square.
displX(W,Id) ~ gaussian(Mean,1.0230073205734391) := \+ratleastOne(W,Id)~=Sh_M,heavy(W,Id)~=B_M,B_M==false,shape(W,Id)~=Sh_M_1,Sh_M_1==triangle,posX_t0(W,Id)~=X_M,getMean([X_M],[-0.521896406787657,0.8202886033455825],Mean).
displX(W,Id) ~ gaussian(Mean,0.7215451429829453) := \+ratleastOne(W,Id)~=Sh_M,heavy(W,Id)~=B_M,B_M==false,shape(W,Id)~=Sh_M_1,Sh_M_1==circle,posX_t0(W,Id)~=X_M,getMean([X_M],[-0.6760377431171408,0.6900101065781399],Mean).
displX(W,Id) ~ gaussian(-0.192171875929,1.02956965818) := true.
posX_t1(W,Id) ~ gaussian(Mean,0.0025511244650500055) := posX_t0(W,Id)~=X_M,displX(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9987881744989153,0.9966709507411197,0.0018947239151172468],Mean).
posX_t1(W,Id) ~ gaussian(2.3754573241,1.34899102631) := true.
posY_t1(W,Id) ~ gaussian(Mean,2.8442717664288795e-31) := posY_t0(W,Id)~=X_M,getMean([X_M],[0.9999999999999997,8.881784197001252e-16],Mean).
posY_t1(W,Id) ~ gaussian(2.50285062704,1.79660130616) := true.
