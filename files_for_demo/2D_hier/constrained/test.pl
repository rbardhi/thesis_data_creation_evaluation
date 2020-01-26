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
move_left_of(W,Id,Id1) ~ finite([0.5254604550379198:true,0.4745395449620802:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_left_of(W,Id,Id1) ~ finite([0.5488194001276324:false,0.4511805998723676:true]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_left_of(W,Id,Id1) ~ finite([0.5488803932277444:true,0.4511196067722556:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==circle.
move_left_of(W,Id,Id1) ~ finite([1.0:false]) := left_of(W,Id1,Id)~=B_M,B_M==false.
move_left_of(W,Id,Id1) ~ finite([0.03973686549915548:true,0.9602631345008446:false]) := true.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_north_of(W,Id,Id1) ~ finite([0.5229621125143513:true,0.4770378874856487:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_north_of(W,Id,Id1) ~ finite([0.5514655760054533:false,0.4485344239945467:true]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==square,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==square.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==triangle.
move_north_of(W,Id,Id1) ~ finite([0.42706013363028955:false,0.5729398663697105:true]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==triangle,shape(W,Id1)~=Sh_M_1,Sh_M_1==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==true,shape(W,Id)~=Sh_M,Sh_M==circle.
move_north_of(W,Id,Id1) ~ finite([1.0:false]) := north_of(W,Id1,Id)~=B_M,B_M==false.
move_north_of(W,Id,Id1) ~ finite([0.038492310427593567:true,0.9615076895724064:false]) := true.
displX(W,Id) ~ gaussian(0.0,0.0) := heavy(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.00781057197183,0.00433135745143) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==true,mno(W,Id)~=B_M_3,B_M_3==true.
displX(W,Id) ~ gaussian(-0.0486370904505,0.0274104382499) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==true,mno(W,Id)~=B_M_3,B_M_3==false.
displX(W,Id) ~ gaussian(-0.759745625625,0.0399312132394) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==true,blocked_north(W,Id)~=B_M_4,B_M_4==true.
displX(W,Id) ~ gaussian(-0.401998543879,0.167693590647) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==true,blocked_north(W,Id)~=B_M_4,B_M_4==false.
displX(W,Id) ~ gaussian(-0.7900608725,0.0196554890325) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_north(W,Id)~=B_M_4,B_M_4==true.
displX(W,Id) ~ gaussian(-0.714394851746,0.0422299808894) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==true,blocked_left(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_north(W,Id)~=B_M_4,B_M_4==false.
displX(W,Id) ~ gaussian(0.0368522545745,0.0201818193631) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==true,blocked_right(W,Id)~=B_M_3,B_M_3==true.
displX(W,Id) ~ gaussian(0.75359979375,0.0215141559742) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==true,blocked_right(W,Id)~=B_M_3,B_M_3==false,mso(W,Id)~=B_M_4,B_M_4==true,blocked_south(W,Id)~=B_M_5,B_M_5==true.
displX(W,Id) ~ gaussian(0.455284265537,0.173292579775) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==true,blocked_right(W,Id)~=B_M_3,B_M_3==false,mso(W,Id)~=B_M_4,B_M_4==true,blocked_south(W,Id)~=B_M_5,B_M_5==false.
displX(W,Id) ~ gaussian(0.74275831134,0.0329630871128) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==true,blocked_right(W,Id)~=B_M_3,B_M_3==false,mso(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==true.
displX(W,Id) ~ gaussian(0.680854866582,0.063908687222) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==true,blocked_right(W,Id)~=B_M_3,B_M_3==false,mso(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==false.
displX(W,Id) ~ gaussian(-0.664881666818,0.0638328674212) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==true,blocked_north(W,Id)~=B_M_4,B_M_4==true.
displX(W,Id) ~ gaussian(-0.0203326306312,0.014023528221) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==true,blocked_north(W,Id)~=B_M_4,B_M_4==false.
displX(W,Id) ~ gaussian(0.0,0.0) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_south(W,Id)~=B_M_4,B_M_4==true,blocked_left(W,Id)~=B_M_5,B_M_5==true,mso(W,Id)~=B_M_6,B_M_6==false.
displX(W,Id) ~ gaussian(0.0551370579119,0.0400769611466) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_south(W,Id)~=B_M_4,B_M_4==true,blocked_left(W,Id)~=B_M_5,B_M_5==false.
displX(W,Id) ~ gaussian(0.0067726129003,0.00528898862437) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_south(W,Id)~=B_M_4,B_M_4==false,blocked_left(W,Id)~=B_M_5,B_M_5==true.
displX(W,Id) ~ gaussian(0.00216449874122,0.00166663630809) := heavy(W,Id)~=B_M,B_M==false,mlo(W,Id)~=B_M_1,B_M_1==false,mro(W,Id)~=B_M_2,B_M_2==false,mno(W,Id)~=B_M_3,B_M_3==false,blocked_south(W,Id)~=B_M_4,B_M_4==false,blocked_left(W,Id)~=B_M_5,B_M_5==false.
displX(W,Id) ~ gaussian(-0.0360757750164,0.0766537606586) := true.
displY(W,Id) ~ gaussian(0.0,0.0) := heavy(W,Id)~=B_M,B_M==true.
displY(W,Id) ~ gaussian(0.0161849032308,0.00838140399443) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==true,blocked_north(W,Id)~=B_M_2,B_M_2==true,mlo(W,Id)~=B_M_3,B_M_3==true.
displY(W,Id) ~ gaussian(0.0496349137273,0.0250247354507) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==true,blocked_north(W,Id)~=B_M_2,B_M_2==true,mlo(W,Id)~=B_M_3,B_M_3==false.
displY(W,Id) ~ gaussian(0.771347377429,0.0218279914164) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==true,blocked_north(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==true,blocked_left(W,Id)~=B_M_4,B_M_4==true.
displY(W,Id) ~ gaussian(0.385956181799,0.161749215663) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==true,blocked_north(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==true,blocked_left(W,Id)~=B_M_4,B_M_4==false.
displY(W,Id) ~ gaussian(0.724756812254,0.0358739977577) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==true,blocked_north(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false.
displY(W,Id) ~ gaussian(-0.0540577075904,0.0278886181288) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==true,blocked_south(W,Id)~=B_M_3,B_M_3==true,blocked_left(W,Id)~=B_M_4,B_M_4==false.
displY(W,Id) ~ gaussian(-0.357880655041,0.172276927918) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==true,blocked_south(W,Id)~=B_M_3,B_M_3==false,mro(W,Id)~=B_M_4,B_M_4==true,blocked_right(W,Id)~=B_M_5,B_M_5==false.
displY(W,Id) ~ gaussian(-0.718074300714,0.0202531437336) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==true,blocked_south(W,Id)~=B_M_3,B_M_3==false,mro(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==true.
displY(W,Id) ~ gaussian(-0.683187495893,0.0554466827348) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==true,blocked_south(W,Id)~=B_M_3,B_M_3==false,mro(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==false.
displY(W,Id) ~ gaussian(0.686433423333,0.0627890479048) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==true,blocked_left(W,Id)~=B_M_4,B_M_4==true.
displY(W,Id) ~ gaussian(0.0313018520581,0.0232266845961) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==true,blocked_left(W,Id)~=B_M_4,B_M_4==false.
displY(W,Id) ~ gaussian(-0.0149089993333,0.0100025217505) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==true,blocked_south(W,Id)~=B_M_5,B_M_5==true.
displY(W,Id) ~ gaussian(-0.0328333266667,0.0270604204132) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==true,blocked_south(W,Id)~=B_M_5,B_M_5==false,blocked_left(W,Id)~=B_M_6,B_M_6==true.
displY(W,Id) ~ gaussian(-0.0846817960035,0.0601786113446) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==true,blocked_south(W,Id)~=B_M_5,B_M_5==false,blocked_left(W,Id)~=B_M_6,B_M_6==false.
displY(W,Id) ~ gaussian(-0.000884752736536,0.000537774947103) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==true.
displY(W,Id) ~ gaussian(-0.001887670998,0.00178521420015) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==false,blocked_left(W,Id)~=B_M_6,B_M_6==true.
displY(W,Id) ~ gaussian(-0.00297089151115,0.0023217042828) := heavy(W,Id)~=B_M,B_M==false,mno(W,Id)~=B_M_1,B_M_1==false,mso(W,Id)~=B_M_2,B_M_2==false,mlo(W,Id)~=B_M_3,B_M_3==false,blocked_right(W,Id)~=B_M_4,B_M_4==false,blocked_north(W,Id)~=B_M_5,B_M_5==false,blocked_left(W,Id)~=B_M_6,B_M_6==false.
displY(W,Id) ~ gaussian(0.0357081960994,0.0749584379661) := true.
posX_t1(W,Id) ~ gaussian(Mean,3.3152360509219995e-18) := posX_t0(W,Id)~=X_M,displX(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9999999999981414,1.000000000315996,9.618084106932656e-12],Mean).
posX_t1(W,Id) ~ gaussian(2.45611381101,1.44914055114) := true.
posY_t1(W,Id) ~ gaussian(Mean,3.1930728563348206e-18) := posY_t0(W,Id)~=X_M,displY(W,Id)~=X_M_1,getMean([X_M,X_M_1],[1.0000000000019624,0.9999999998966745,-1.3529177778082158e-11],Mean).
posY_t1(W,Id) ~ gaussian(2.53839997678,1.46129767964) := true.
