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


displX(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==square.
displX(W,Id) ~ gaussian(-0.00555848118227,0.00313329473268) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_left(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.4024804719,0.155512012808) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_left(W,Id)~=B_M,B_M==false.
displX(W,Id) ~ gaussian(-0.00285527475676,0.00150414253442) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.184405594279,0.116909783318) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==true.
displX(W,Id) ~ gaussian(-0.0998033038521,0.0670592492802) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square.
displX(W,Id) ~ gaussian(-0.112066544807,0.0760905141023) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displX(W,Id) ~ gaussian(-0.0675986037113,0.047272630503) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle.
displX(W,Id) ~ gaussian(-0.144543574773,0.0931079801876) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_left(W,Id)~=B_M,B_M==false,blocked_north(W,Id)~=B_M_1,B_M_1==false,\+atleastOneNorth(W,Id)~=Sh_M_2.
displX(W,Id) ~ gaussian(-0.273532716462,0.137903365652) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square,blocked_north(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.0932101151803,0.0658877531702) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square,blocked_north(W,Id)~=B_M,B_M==false.
displX(W,Id) ~ gaussian(-0.0229922033854,0.0168154121293) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle,blocked_north(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.0415785505768,0.0304093236027) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle,blocked_north(W,Id)~=B_M,B_M==false.
displX(W,Id) ~ gaussian(-0.000862278520115,0.000517492875391) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle,blocked_left(W,Id)~=B_M,B_M==true.
displX(W,Id) ~ gaussian(-0.0606071648118,0.0431944969742) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle,blocked_left(W,Id)~=B_M,B_M==false.
displX(W,Id) ~ gaussian(-0.0488288223592,0.034595053473) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,\+atleastOneNorth(W,Id)~=Sh_M_2.
displX(W,Id) ~ gaussian(-0.590867040976,0.119057768335) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square.
displX(W,Id) ~ gaussian(-0.215881157593,0.109687765433) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displX(W,Id) ~ gaussian(-0.0219662395062,0.0193306534948) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle.
displX(W,Id) ~ gaussian(-0.00240879381884,0.0019995183486) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M,B_M==false.
displX(W,Id) ~ gaussian(-0.0101727585806,0.00527853493276) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_north(W,Id)~=B_M_1,B_M_1==false.
displX(W,Id) ~ gaussian(-0.0189847340566,0.010647679601) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle.
displX(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square.
displX(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displX(W,Id) ~ gaussian(-0.0151721855034,0.00840989911632) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle.
displX(W,Id) ~ gaussian(-0.0140657059459,0.00722039511584) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==false,\+atleastOneNorth(W,Id)~=Sh_M_2.
displX(W,Id) ~ gaussian(-0.414573605812,0.155160676579) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==square.
displX(W,Id) ~ gaussian(-0.437288628736,0.155894538712) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==triangle.
displX(W,Id) ~ gaussian(-0.321193131254,0.14909968354) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==true.
displX(W,Id) ~ gaussian(-0.232104941151,0.130373920215) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_north(W,Id)~=B_M_1,B_M_1==false.
displX(W,Id) ~ gaussian(-0.494036468913,0.144356801839) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M_1,B_M_1==true.
displX(W,Id) ~ gaussian(-0.00970815446779,0.00755116184486) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==square.
displX(W,Id) ~ gaussian(-0.0209257504707,0.0158660380019) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displX(W,Id) ~ gaussian(-0.00269320764602,0.00214938105552) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M_1,B_M_1==false,atleastOneNorth(W,Id)~=Sh_M_2,Sh_M_2==circle.
displX(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_1,blocked_north(W,Id)~=B_M_1,B_M_1==false,\+atleastOneNorth(W,Id)~=Sh_M_2.
displX(W,Id) ~ gaussian(-0.0981579157246,0.067718641083) := true.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==square.
displY(W,Id) ~ gaussian(0.0283284533514,0.015888382118) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_north(W,Id)~=B_M,B_M==true.
displY(W,Id) ~ gaussian(0.498199181724,0.138942364927) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==true.
displY(W,Id) ~ gaussian(0.392261532726,0.154515014853) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==square,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.00897270736842,0.00484191765323) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.00341924333333,0.00224471519473) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.012070722037,0.00786792584675) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==true,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.239841458414,0.126332910299) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==true.
displY(W,Id) ~ gaussian(0.108617447602,0.0785636657143) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.134099558912,0.087563532873) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.0523250940323,0.0386730092543) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.18961839842,0.110859182389) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.5002225765,0.160061245831) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square,blocked_left(W,Id)~=B_M,B_M==true.
displY(W,Id) ~ gaussian(0.0533093264139,0.0401244626496) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square,blocked_left(W,Id)~=B_M,B_M==false.
displY(W,Id) ~ gaussian(0.137367351616,0.0896647858088) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle,blocked_left(W,Id)~=B_M,B_M==true.
displY(W,Id) ~ gaussian(0.0290814393643,0.022782171307) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle,blocked_left(W,Id)~=B_M,B_M==false.
displY(W,Id) ~ gaussian(0.00986702950794,0.00551535503882) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle,blocked_north(W,Id)~=B_M,B_M==true.
displY(W,Id) ~ gaussian(0.0919331521944,0.0651095567608) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==true.
displY(W,Id) ~ gaussian(0.0494691466531,0.0365807184224) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle,blocked_north(W,Id)~=B_M,B_M==false,blocked_left(W,Id)~=B_M_1,B_M_1==false.
displY(W,Id) ~ gaussian(0.0408418317251,0.0303709219242) := shape(W,Id)~=Sh_M,Sh_M==triangle,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.50510461871,0.143446772657) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.109953750943,0.0716113752525) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.0409596008621,0.0317262859257) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==true,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.0170778335745,0.0118610031121) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.0038735884044,0.00293168659782) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.00326158670588,0.00274458298871) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==false,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==triangle,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M,B_M==false,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.0233569677596,0.0138971901223) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==square.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.00773872268817,0.00553946230911) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.02839007275,0.0155167204978) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.0411057387143,0.0224101785818) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==square.
displY(W,Id) ~ gaussian(0.0,0.0) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==triangle.
displY(W,Id) ~ gaussian(0.0258919910204,0.0135293443775) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,atleastOneLeft(W,Id)~=Sh_M_2,Sh_M_2==circle.
displY(W,Id) ~ gaussian(0.0513375692754,0.0282669215694) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==true,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,\+atleastOneLeft(W,Id)~=Sh_M_2.
displY(W,Id) ~ gaussian(0.406127133018,0.154015735691) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==square.
displY(W,Id) ~ gaussian(0.423759684261,0.152840752836) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==triangle.
displY(W,Id) ~ gaussian(0.421791700141,0.141923148214) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_left(W,Id)~=B_M_1,B_M_1==true.
displY(W,Id) ~ gaussian(0.233380752897,0.128852170388) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,atleastOneNorth(W,Id)~=Sh_M_1,Sh_M_1==circle,blocked_left(W,Id)~=B_M_1,B_M_1==false.
displY(W,Id) ~ gaussian(0.388671329651,0.149710427281) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M_1,B_M_1==true.
displY(W,Id) ~ gaussian(0.0114143437022,0.00875818382665) := shape(W,Id)~=Sh_M,Sh_M==circle,blocked_north(W,Id)~=B_M,B_M==false,\+atleastOneNorth(W,Id)~=Sh_M_1,blocked_left(W,Id)~=B_M_1,B_M_1==false.
displY(W,Id) ~ gaussian(0.0961465783682,0.0658212874519) := true.
posX_t1(W,Id) ~ gaussian(Mean,3.0534612172358344e-18) := posX_t0(W,Id)~=X_M,displX(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9999999999916973,1.000000000032411,2.5429436334434286e-11],Mean).
posX_t1(W,Id) ~ gaussian(2.37795097428,1.40469450413) := true.
posY_t1(W,Id) ~ gaussian(Mean,3.3090725860771358e-18) := posY_t0(W,Id)~=X_M,displY(W,Id)~=X_M_1,getMean([X_M,X_M_1],[0.9999999999968736,0.9999999998633439,6.203482172395525e-12],Mean).
posY_t1(W,Id) ~ gaussian(2.62723004236,1.42259479829) := true.
