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


n(W) ~ val(_) :=true.
posX_t0(W,I) ~ val(_) :=true.
posY_t0(W,I) ~ val(_) :=true. 
shape(W,I) ~ val(_) :=true.
size(W,I) ~ val(_) :=true.
heavy(W,I) ~ val(_) := true.
left_of(W,I,I1) ~ val(_) :=true.
north_of(W,I,I1) ~ val(_) :=true.
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
  C = north.*/


move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_left_of(W,Id,Id1) ~ finite([0.502944640753828:true,0.49705535924617195:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_left_of(W,Id,Id1) ~ finite([0.6061162079510704:false,0.39388379204892965:true]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_left_of(W,Id,Id1) ~ finite([0.45209302325581396:false,0.5479069767441861:true]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==false.
move_left_of(W,Id,Id1) ~ finite([0.03762601763192306:true,0.9623739823680769:false]) := true.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_north_of(W,Id,Id1) ~ finite([0.5283564814814815:true,0.47164351851851855:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_north_of(W,Id,Id1) ~ finite([0.38523182598740696:true,0.614768174012593:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_north_of(W,Id,Id1) ~ finite([0.47119140625:false,0.52880859375:true]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==false.
move_north_of(W,Id,Id1) ~ finite([0.037527593818984545:true,0.9624724061810155:false]) := true.
displX(W,Id) ~ gaussian(-0.417454363651,0.166149030079) := mlo(W,Id)~=B_M,B_M==true,blocked_left(W,Id)~=B_M_1,B_M_1==false,mno(W,Id)~=B_M_2,B_M_2==true,blocked_north(W,Id)~=B_M_3,B_M_3==false.
displX(W,Id) ~ gaussian(-0.725155795817,0.0436110228092) := mlo(W,Id)~=B_M,B_M==true,blocked_left(W,Id)~=B_M_1,B_M_1==false,mno(W,Id)~=B_M_2,B_M_2==false.
displX(W,Id) ~ gaussian(0.0,0.0) := mlo(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==true,mno(W,Id)~=B_M_2,B_M_2==false.
displX(W,Id) ~ gaussian(-0.00324040333184,0.00245314810642) := mlo(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==false.
displX(W,Id) ~ gaussian(-0.0981579157246,0.067718641083) := true.
displY(W,Id) ~ gaussian(0.375377952621,0.164974299743) := mno(W,Id)~=B_M,B_M==true,blocked_north(W,Id)~=B_M_1,B_M_1==false,mlo(W,Id)~=B_M_2,B_M_2==true,blocked_left(W,Id)~=B_M_3,B_M_3==false.
displY(W,Id) ~ gaussian(0.720456207855,0.0402193676094) := mno(W,Id)~=B_M,B_M==true,blocked_north(W,Id)~=B_M_1,B_M_1==false,mlo(W,Id)~=B_M_2,B_M_2==false.
displY(W,Id) ~ gaussian(0.0,0.0) := mno(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==true,mlo(W,Id)~=B_M_2,B_M_2==false.
displY(W,Id) ~ gaussian(0.00390047445137,0.00301672512933) := mno(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false.
displY(W,Id) ~ gaussian(0.0961465783682,0.0658212874519) := true.
posX_t1(W,Id) ~ gaussian(Mean,3.0534612172358344e-18) := posX_t0(W,Id)~=X_M,displX(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9999999999916973,1.000000000032411,2.5429436334434286e-11],Mean).
posX_t1(W,Id) ~ gaussian(2.37795097428,1.40469450413) := true.
posY_t1(W,Id) ~ gaussian(Mean,3.3090725860771358e-18) := posY_t0(W,Id)~=X_M,displY(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9999999999968736,0.9999999998633439,6.203482172395525e-12],Mean).
posY_t1(W,Id) ~ gaussian(2.62723004236,1.42259479829) := true.
