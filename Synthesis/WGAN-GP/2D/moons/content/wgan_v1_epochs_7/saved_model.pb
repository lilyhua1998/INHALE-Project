´»
ªÿ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8²
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:
*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:
*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:
*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:
*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:
*
dtype0
¢
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:
*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:

*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:
*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:
*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:
*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:
*
dtype0
¢
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:
*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:<*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:<*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:<*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ìS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§S
valueSBS BS
°
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api

signatures
ã
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
trainable_variables
regularization_losses
	keras_api
Ê
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
 regularization_losses
!	keras_api
Ö
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
:24
;25
<26
=27
¦
"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513
614
715
816
917
:18
;19
<20
=21
 
­
>metrics

?layers
	variables
trainable_variables
@layer_metrics
regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables
 
h

"kernel
#bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api

Gaxis
	$gamma
%beta
&moving_mean
'moving_variance
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

(kernel
)bias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api

Paxis
	*gamma
+beta
,moving_mean
-moving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api

Yaxis
	0gamma
1beta
2moving_mean
3moving_variance
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
h

4kernel
5bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
f
"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513
 
­
bmetrics

clayers
	variables
trainable_variables
dlayer_metrics
regularization_losses
elayer_regularization_losses
fnon_trainable_variables
h

6kernel
7bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
R
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

8kernel
9bias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
k

:kernel
;bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

<kernel
=bias
	variables
trainable_variables
regularization_losses
	keras_api
8
60
71
82
93
:4
;5
<6
=7
8
60
71
82
93
:4
;5
<6
=7
 
²
metrics
layers
	variables
trainable_variables
layer_metrics
 regularization_losses
 layer_regularization_losses
non_trainable_variables
JH
VARIABLE_VALUEdense_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_8/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_3/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_3/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_3/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_9/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_9/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_4/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_5/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_5/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_11/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_11/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_12/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_12/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_13/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_13/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_14/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_14/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_15/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_15/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
*
&0
'1
,2
-3
24
35

"0
#1

"0
#1
 
²
metrics
layers
C	variables
Dtrainable_variables
layer_metrics
Eregularization_losses
 layer_regularization_losses
non_trainable_variables
 

$0
%1
&2
'3

$0
%1
 
²
metrics
layers
H	variables
Itrainable_variables
layer_metrics
Jregularization_losses
  layer_regularization_losses
¡non_trainable_variables

(0
)1

(0
)1
 
²
¢metrics
£layers
L	variables
Mtrainable_variables
¤layer_metrics
Nregularization_losses
 ¥layer_regularization_losses
¦non_trainable_variables
 

*0
+1
,2
-3

*0
+1
 
²
§metrics
¨layers
Q	variables
Rtrainable_variables
©layer_metrics
Sregularization_losses
 ªlayer_regularization_losses
«non_trainable_variables

.0
/1

.0
/1
 
²
¬metrics
­layers
U	variables
Vtrainable_variables
®layer_metrics
Wregularization_losses
 ¯layer_regularization_losses
°non_trainable_variables
 

00
11
22
33

00
11
 
²
±metrics
²layers
Z	variables
[trainable_variables
³layer_metrics
\regularization_losses
 ´layer_regularization_losses
µnon_trainable_variables

40
51

40
51
 
²
¶metrics
·layers
^	variables
_trainable_variables
¸layer_metrics
`regularization_losses
 ¹layer_regularization_losses
ºnon_trainable_variables
 
1
0
	1

2
3
4
5
6
 
 
*
&0
'1
,2
-3
24
35

60
71

60
71
 
²
»metrics
¼layers
g	variables
htrainable_variables
½layer_metrics
iregularization_losses
 ¾layer_regularization_losses
¿non_trainable_variables
 
 
 
²
Àmetrics
Álayers
k	variables
ltrainable_variables
Âlayer_metrics
mregularization_losses
 Ãlayer_regularization_losses
Änon_trainable_variables
 
 
 
²
Åmetrics
Ælayers
o	variables
ptrainable_variables
Çlayer_metrics
qregularization_losses
 Èlayer_regularization_losses
Énon_trainable_variables

80
91

80
91
 
²
Êmetrics
Ëlayers
s	variables
ttrainable_variables
Ìlayer_metrics
uregularization_losses
 Ílayer_regularization_losses
Înon_trainable_variables
 
 
 
²
Ïmetrics
Ðlayers
w	variables
xtrainable_variables
Ñlayer_metrics
yregularization_losses
 Òlayer_regularization_losses
Ónon_trainable_variables
 
 
 
²
Ômetrics
Õlayers
{	variables
|trainable_variables
Ölayer_metrics
}regularization_losses
 ×layer_regularization_losses
Ønon_trainable_variables

:0
;1

:0
;1
 
´
Ùmetrics
Úlayers
	variables
trainable_variables
Ûlayer_metrics
regularization_losses
 Ülayer_regularization_losses
Ýnon_trainable_variables
 
 
 
µ
Þmetrics
ßlayers
	variables
trainable_variables
àlayer_metrics
regularization_losses
 álayer_regularization_losses
ânon_trainable_variables
 
 
 
µ
ãmetrics
älayers
	variables
trainable_variables
ålayer_metrics
regularization_losses
 ælayer_regularization_losses
çnon_trainable_variables
 
 
 
µ
èmetrics
élayers
	variables
trainable_variables
êlayer_metrics
regularization_losses
 ëlayer_regularization_losses
ìnon_trainable_variables

<0
=1

<0
=1
 
µ
ímetrics
îlayers
	variables
trainable_variables
ïlayer_metrics
regularization_losses
 ðlayer_regularization_losses
ñnon_trainable_variables
 
N
0
1
2
3
4
5
6
7
8
9
10
 
 
 
 
 
 
 
 
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 

,0
-1
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

"serving_default_sequential_2_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ù
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_2_inputdense_8/kerneldense_8/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_9/kerneldense_9/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_10/kerneldense_10/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_51040
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ë
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_52519
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_9/kerneldense_9/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_10/kerneldense_10/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_52613¶æ

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_50278

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

C
'__inference_re_lu_5_layer_call_fn_52355

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_502582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
¨
5__inference_batch_normalization_4_layer_call_fn_52092

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_50145

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã	
Ü
C__inference_dense_11_layer_call_and_return_conditional_losses_49824

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52168

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ä

,__inference_sequential_2_layer_call_fn_49989
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_499462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
µ
¨
5__inference_batch_normalization_3_layer_call_fn_51990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_493322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_10_layer_call_and_return_conditional_losses_49762

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Þü

G__inference_sequential_4_layer_call_and_return_conditional_losses_51221

inputs7
3sequential_2_dense_8_matmul_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resource<
8sequential_2_batch_normalization_3_assignmovingavg_51058>
:sequential_2_batch_normalization_3_assignmovingavg_1_51064L
Hsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_3_batchnorm_readvariableop_resource7
3sequential_2_dense_9_matmul_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource<
8sequential_2_batch_normalization_4_assignmovingavg_51097>
:sequential_2_batch_normalization_4_assignmovingavg_1_51103L
Hsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_4_batchnorm_readvariableop_resource8
4sequential_2_dense_10_matmul_readvariableop_resource9
5sequential_2_dense_10_biasadd_readvariableop_resource<
8sequential_2_batch_normalization_5_assignmovingavg_51136>
:sequential_2_batch_normalization_5_assignmovingavg_1_51142L
Hsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_5_batchnorm_readvariableop_resource8
4sequential_2_dense_11_matmul_readvariableop_resource9
5sequential_2_dense_11_biasadd_readvariableop_resource8
4sequential_3_dense_12_matmul_readvariableop_resource9
5sequential_3_dense_12_biasadd_readvariableop_resource8
4sequential_3_dense_13_matmul_readvariableop_resource9
5sequential_3_dense_13_biasadd_readvariableop_resource8
4sequential_3_dense_14_matmul_readvariableop_resource9
5sequential_3_dense_14_biasadd_readvariableop_resource8
4sequential_3_dense_15_matmul_readvariableop_resource9
5sequential_3_dense_15_biasadd_readvariableop_resource
identity¢Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢,sequential_2/dense_10/BiasAdd/ReadVariableOp¢+sequential_2/dense_10/MatMul/ReadVariableOp¢,sequential_2/dense_11/BiasAdd/ReadVariableOp¢+sequential_2/dense_11/MatMul/ReadVariableOp¢+sequential_2/dense_8/BiasAdd/ReadVariableOp¢*sequential_2/dense_8/MatMul/ReadVariableOp¢+sequential_2/dense_9/BiasAdd/ReadVariableOp¢*sequential_2/dense_9/MatMul/ReadVariableOp¢,sequential_3/dense_12/BiasAdd/ReadVariableOp¢+sequential_3/dense_12/MatMul/ReadVariableOp¢,sequential_3/dense_13/BiasAdd/ReadVariableOp¢+sequential_3/dense_13/MatMul/ReadVariableOp¢,sequential_3/dense_14/BiasAdd/ReadVariableOp¢+sequential_3/dense_14/MatMul/ReadVariableOp¢,sequential_3/dense_15/BiasAdd/ReadVariableOp¢+sequential_3/dense_15/MatMul/ReadVariableOpÌ
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp²
sequential_2/dense_8/MatMulMatMulinputs2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/MatMulË
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpÕ
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/ReluÐ
Asequential_2/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/batch_normalization_3/moments/mean/reduction_indices
/sequential_2/batch_normalization_3/moments/meanMean'sequential_2/dense_8/Relu:activations:0Jsequential_2/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(21
/sequential_2/batch_normalization_3/moments/meanå
7sequential_2/batch_normalization_3/moments/StopGradientStopGradient8sequential_2/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:29
7sequential_2/batch_normalization_3/moments/StopGradient®
<sequential_2/batch_normalization_3/moments/SquaredDifferenceSquaredDifference'sequential_2/dense_8/Relu:activations:0@sequential_2/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<sequential_2/batch_normalization_3/moments/SquaredDifferenceØ
Esequential_2/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_2/batch_normalization_3/moments/variance/reduction_indices¾
3sequential_2/batch_normalization_3/moments/varianceMean@sequential_2/batch_normalization_3/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(25
3sequential_2/batch_normalization_3/moments/varianceé
2sequential_2/batch_normalization_3/moments/SqueezeSqueeze8sequential_2/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 24
2sequential_2/batch_normalization_3/moments/Squeezeñ
4sequential_2/batch_normalization_3/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_2/batch_normalization_3/moments/Squeeze_1´
8sequential_2/batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/51058*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_3/AssignMovingAvg/decayû
Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_3_assignmovingavg_51058*
_output_shapes
:*
dtype02C
Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_3/AssignMovingAvg/subSubIsequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/51058*
_output_shapes
:28
6sequential_2/batch_normalization_3/AssignMovingAvg/sub
6sequential_2/batch_normalization_3/AssignMovingAvg/mulMul:sequential_2/batch_normalization_3/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/51058*
_output_shapes
:28
6sequential_2/batch_normalization_3/AssignMovingAvg/mulÿ
Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_3_assignmovingavg_51058:sequential_2/batch_normalization_3/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/51058*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpº
:sequential_2/batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/51064*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_3/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_3_assignmovingavg_1_51064*
_output_shapes
:*
dtype02E
Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_3/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/51064*
_output_shapes
:2:
8sequential_2/batch_normalization_3/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_3/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_3/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/51064*
_output_shapes
:2:
8sequential_2/batch_normalization_3/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_3_assignmovingavg_1_51064<sequential_2/batch_normalization_3/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/51064*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp­
2sequential_2/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_3/batchnorm/add/y
0sequential_2/batch_normalization_3/batchnorm/addAddV2=sequential_2/batch_normalization_3/moments/Squeeze_1:output:0;sequential_2/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/addÌ
2sequential_2/batch_normalization_3/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/Rsqrt
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_3/batchnorm/mulMul6sequential_2/batch_normalization_3/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/mul
2sequential_2/batch_normalization_3/batchnorm/mul_1Mul'sequential_2/dense_8/Relu:activations:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_2/batch_normalization_3/batchnorm/mul_1
2sequential_2/batch_normalization_3/batchnorm/mul_2Mul;sequential_2/batch_normalization_3/moments/Squeeze:output:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/mul_2û
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp
0sequential_2/batch_normalization_3/batchnorm/subSubCsequential_2/batch_normalization_3/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/sub
2sequential_2/batch_normalization_3/batchnorm/add_1AddV26sequential_2/batch_normalization_3/batchnorm/mul_1:z:04sequential_2/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_2/batch_normalization_3/batchnorm/add_1Ì
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOpâ
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/MatMulË
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpÕ
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/BiasAdd
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/ReluÐ
Asequential_2/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/batch_normalization_4/moments/mean/reduction_indices
/sequential_2/batch_normalization_4/moments/meanMean'sequential_2/dense_9/Relu:activations:0Jsequential_2/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(21
/sequential_2/batch_normalization_4/moments/meanå
7sequential_2/batch_normalization_4/moments/StopGradientStopGradient8sequential_2/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:
29
7sequential_2/batch_normalization_4/moments/StopGradient®
<sequential_2/batch_normalization_4/moments/SquaredDifferenceSquaredDifference'sequential_2/dense_9/Relu:activations:0@sequential_2/batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2>
<sequential_2/batch_normalization_4/moments/SquaredDifferenceØ
Esequential_2/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_2/batch_normalization_4/moments/variance/reduction_indices¾
3sequential_2/batch_normalization_4/moments/varianceMean@sequential_2/batch_normalization_4/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(25
3sequential_2/batch_normalization_4/moments/varianceé
2sequential_2/batch_normalization_4/moments/SqueezeSqueeze8sequential_2/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 24
2sequential_2/batch_normalization_4/moments/Squeezeñ
4sequential_2/batch_normalization_4/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 26
4sequential_2/batch_normalization_4/moments/Squeeze_1´
8sequential_2/batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/51097*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_4/AssignMovingAvg/decayû
Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_4_assignmovingavg_51097*
_output_shapes
:
*
dtype02C
Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_4/AssignMovingAvg/subSubIsequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/51097*
_output_shapes
:
28
6sequential_2/batch_normalization_4/AssignMovingAvg/sub
6sequential_2/batch_normalization_4/AssignMovingAvg/mulMul:sequential_2/batch_normalization_4/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/51097*
_output_shapes
:
28
6sequential_2/batch_normalization_4/AssignMovingAvg/mulÿ
Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_4_assignmovingavg_51097:sequential_2/batch_normalization_4/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/51097*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpº
:sequential_2/batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/51103*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_4/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_4_assignmovingavg_1_51103*
_output_shapes
:
*
dtype02E
Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_4/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/51103*
_output_shapes
:
2:
8sequential_2/batch_normalization_4/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_4/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_4/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/51103*
_output_shapes
:
2:
8sequential_2/batch_normalization_4/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_4_assignmovingavg_1_51103<sequential_2/batch_normalization_4/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/51103*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp­
2sequential_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_4/batchnorm/add/y
0sequential_2/batch_normalization_4/batchnorm/addAddV2=sequential_2/batch_normalization_4/moments/Squeeze_1:output:0;sequential_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/addÌ
2sequential_2/batch_normalization_4/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/Rsqrt
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_4/batchnorm/mulMul6sequential_2/batch_normalization_4/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/mul
2sequential_2/batch_normalization_4/batchnorm/mul_1Mul'sequential_2/dense_9/Relu:activations:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_4/batchnorm/mul_1
2sequential_2/batch_normalization_4/batchnorm/mul_2Mul;sequential_2/batch_normalization_4/moments/Squeeze:output:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/mul_2û
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp
0sequential_2/batch_normalization_4/batchnorm/subSubCsequential_2/batch_normalization_4/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/sub
2sequential_2/batch_normalization_4/batchnorm/add_1AddV26sequential_2/batch_normalization_4/batchnorm/mul_1:z:04sequential_2/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_4/batchnorm/add_1Ï
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02-
+sequential_2/dense_10/MatMul/ReadVariableOpå
sequential_2/dense_10/MatMulMatMul6sequential_2/batch_normalization_4/batchnorm/add_1:z:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/MatMulÎ
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOpÙ
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/BiasAdd
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/ReluÐ
Asequential_2/batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_2/batch_normalization_5/moments/mean/reduction_indices
/sequential_2/batch_normalization_5/moments/meanMean(sequential_2/dense_10/Relu:activations:0Jsequential_2/batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(21
/sequential_2/batch_normalization_5/moments/meanå
7sequential_2/batch_normalization_5/moments/StopGradientStopGradient8sequential_2/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:
29
7sequential_2/batch_normalization_5/moments/StopGradient¯
<sequential_2/batch_normalization_5/moments/SquaredDifferenceSquaredDifference(sequential_2/dense_10/Relu:activations:0@sequential_2/batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2>
<sequential_2/batch_normalization_5/moments/SquaredDifferenceØ
Esequential_2/batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_2/batch_normalization_5/moments/variance/reduction_indices¾
3sequential_2/batch_normalization_5/moments/varianceMean@sequential_2/batch_normalization_5/moments/SquaredDifference:z:0Nsequential_2/batch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(25
3sequential_2/batch_normalization_5/moments/varianceé
2sequential_2/batch_normalization_5/moments/SqueezeSqueeze8sequential_2/batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 24
2sequential_2/batch_normalization_5/moments/Squeezeñ
4sequential_2/batch_normalization_5/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 26
4sequential_2/batch_normalization_5/moments/Squeeze_1´
8sequential_2/batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/51136*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_5/AssignMovingAvg/decayû
Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_5_assignmovingavg_51136*
_output_shapes
:
*
dtype02C
Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_5/AssignMovingAvg/subSubIsequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/51136*
_output_shapes
:
28
6sequential_2/batch_normalization_5/AssignMovingAvg/sub
6sequential_2/batch_normalization_5/AssignMovingAvg/mulMul:sequential_2/batch_normalization_5/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/51136*
_output_shapes
:
28
6sequential_2/batch_normalization_5/AssignMovingAvg/mulÿ
Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_5_assignmovingavg_51136:sequential_2/batch_normalization_5/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/51136*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpº
:sequential_2/batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/51142*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_5/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_5_assignmovingavg_1_51142*
_output_shapes
:
*
dtype02E
Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_5/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/51142*
_output_shapes
:
2:
8sequential_2/batch_normalization_5/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_5/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_5/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/51142*
_output_shapes
:
2:
8sequential_2/batch_normalization_5/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_5_assignmovingavg_1_51142<sequential_2/batch_normalization_5/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/51142*
_output_shapes
 *
dtype02J
Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp­
2sequential_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_5/batchnorm/add/y
0sequential_2/batch_normalization_5/batchnorm/addAddV2=sequential_2/batch_normalization_5/moments/Squeeze_1:output:0;sequential_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/addÌ
2sequential_2/batch_normalization_5/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/Rsqrt
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_5/batchnorm/mulMul6sequential_2/batch_normalization_5/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/mul
2sequential_2/batch_normalization_5/batchnorm/mul_1Mul(sequential_2/dense_10/Relu:activations:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_5/batchnorm/mul_1
2sequential_2/batch_normalization_5/batchnorm/mul_2Mul;sequential_2/batch_normalization_5/moments/Squeeze:output:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/mul_2û
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp
0sequential_2/batch_normalization_5/batchnorm/subSubCsequential_2/batch_normalization_5/batchnorm/ReadVariableOp:value:06sequential_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/sub
2sequential_2/batch_normalization_5/batchnorm/add_1AddV26sequential_2/batch_normalization_5/batchnorm/mul_1:z:04sequential_2/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_5/batchnorm/add_1Ï
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOpå
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/MatMulÎ
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/dense_11/BiasAdd/ReadVariableOpÙ
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/BiasAdd
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/TanhÏ
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOpÍ
sequential_3/dense_12/MatMulMatMulsequential_2/dense_11/Tanh:y:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_12/MatMulÎ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_12/BiasAdd/ReadVariableOpÙ
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_12/BiasAdd
sequential_3/re_lu_3/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/re_lu_3/Relu
$sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_3/dropout/ConstÙ
"sequential_3/dropout_3/dropout/MulMul'sequential_3/re_lu_3/Relu:activations:0-sequential_3/dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_3/dropout_3/dropout/Mul£
$sequential_3/dropout_3/dropout/ShapeShape'sequential_3/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_3/dropout/Shapeù
;sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02=
;sequential_3/dropout_3/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2/
-sequential_3/dropout_3/dropout/GreaterEqual/y
+sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_3/dropout_3/dropout/GreaterEqualÄ
#sequential_3/dropout_3/dropout/CastCast/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_3/dropout_3/dropout/CastÖ
$sequential_3/dropout_3/dropout/Mul_1Mul&sequential_3/dropout_3/dropout/Mul:z:0'sequential_3/dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential_3/dropout_3/dropout/Mul_1Ï
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp×
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_3/dropout/Mul_1:z:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/dense_13/MatMulÎ
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOpÙ
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/dense_13/BiasAdd
sequential_3/re_lu_4/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/re_lu_4/Relu
$sequential_3/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_4/dropout/ConstÙ
"sequential_3/dropout_4/dropout/MulMul'sequential_3/re_lu_4/Relu:activations:0-sequential_3/dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2$
"sequential_3/dropout_4/dropout/Mul£
$sequential_3/dropout_4/dropout/ShapeShape'sequential_3/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_4/dropout/Shapeù
;sequential_3/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype02=
;sequential_3/dropout_4/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2/
-sequential_3/dropout_4/dropout/GreaterEqual/y
+sequential_3/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2-
+sequential_3/dropout_4/dropout/GreaterEqualÄ
#sequential_3/dropout_4/dropout/CastCast/sequential_3/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2%
#sequential_3/dropout_4/dropout/CastÖ
$sequential_3/dropout_4/dropout/Mul_1Mul&sequential_3/dropout_4/dropout/Mul:z:0'sequential_3/dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2&
$sequential_3/dropout_4/dropout/Mul_1Ï
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp×
sequential_3/dense_14/MatMulMatMul(sequential_3/dropout_4/dropout/Mul_1:z:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_14/MatMulÎ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOpÙ
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_14/BiasAdd
sequential_3/re_lu_5/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/re_lu_5/Relu
$sequential_3/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_5/dropout/ConstÙ
"sequential_3/dropout_5/dropout/MulMul'sequential_3/re_lu_5/Relu:activations:0-sequential_3/dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_3/dropout_5/dropout/Mul£
$sequential_3/dropout_5/dropout/ShapeShape'sequential_3/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_5/dropout/Shapeù
;sequential_3/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02=
;sequential_3/dropout_5/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2/
-sequential_3/dropout_5/dropout/GreaterEqual/y
+sequential_3/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_3/dropout_5/dropout/GreaterEqualÄ
#sequential_3/dropout_5/dropout/CastCast/sequential_3/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_3/dropout_5/dropout/CastÖ
$sequential_3/dropout_5/dropout/Mul_1Mul&sequential_3/dropout_5/dropout/Mul:z:0'sequential_3/dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential_3/dropout_5/dropout/Mul_1
sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_3/flatten_1/ConstÎ
sequential_3/flatten_1/ReshapeReshape(sequential_3/dropout_5/dropout/Mul_1:z:0%sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_3/flatten_1/ReshapeÏ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOpÖ
sequential_3/dense_15/MatMulMatMul'sequential_3/flatten_1/Reshape:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_15/MatMulÎ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOpÙ
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_15/BiasAdd¸
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0G^sequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2
Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2
Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp2
Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2
Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp2
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp2
Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2
Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp2
Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2
Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp2
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2
Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpFsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2
Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOpAsequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp2
Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpHsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2
Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOpCsequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2z
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp2
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49332

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_12_layer_call_and_return_conditional_losses_50099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_15_layer_call_fn_52412

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_503202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì	
Û
B__inference_dense_8_layer_call_and_return_conditional_losses_51899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á{
®
G__inference_sequential_2_layer_call_and_return_conditional_losses_51663

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢.batch_normalization_3/batchnorm/ReadVariableOp¢0batch_normalization_3/batchnorm/ReadVariableOp_1¢0batch_normalization_3/batchnorm/ReadVariableOp_2¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/ReluÔ
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yà
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulÌ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Ú
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ý
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ú
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2Û
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subÝ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/ReluÔ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yà
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulÌ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/mul_1Ú
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ý
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ú
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Û
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/add_1¨
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_10/MatMul/ReadVariableOp±
dense_10/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/ReluÔ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yà
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulÍ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/mul_1Ú
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ý
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ú
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Û
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/add_1¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/TanhÑ
IdentityIdentitydense_11/Tanh:y:0/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_15_layer_call_and_return_conditional_losses_50320

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ+
¯
G__inference_sequential_2_layer_call_and_return_conditional_losses_49841
dense_8_input
dense_8_49649
dense_8_49651
batch_normalization_3_49680
batch_normalization_3_49682
batch_normalization_3_49684
batch_normalization_3_49686
dense_9_49711
dense_9_49713
batch_normalization_4_49742
batch_normalization_4_49744
batch_normalization_4_49746
batch_normalization_4_49748
dense_10_49773
dense_10_49775
batch_normalization_5_49804
batch_normalization_5_49806
batch_normalization_5_49808
batch_normalization_5_49810
dense_11_49835
dense_11_49837
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_49649dense_8_49651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_496382!
dense_8/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_49680batch_normalization_3_49682batch_normalization_3_49684batch_normalization_3_49686*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_492992/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_49711dense_9_49713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_497002!
dense_9/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_49742batch_normalization_4_49744batch_normalization_4_49746batch_normalization_4_49748*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494392/
-batch_normalization_4/StatefulPartitionedCallÁ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_49773dense_10_49775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_497622"
 dense_10/StatefulPartitionedCall±
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_49804batch_normalization_5_49806batch_normalization_5_49808batch_normalization_5_49810*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_495792/
-batch_normalization_5/StatefulPartitionedCallÁ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_49835dense_11_49837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_498242"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
Ø
|
'__inference_dense_9_layer_call_fn_52010

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_497002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_12_layer_call_fn_52233

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_500992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íÍ
þ
G__inference_sequential_4_layer_call_and_return_conditional_losses_51333

inputs7
3sequential_2_dense_8_matmul_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resourceH
Dsequential_2_batch_normalization_3_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_3_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_3_batchnorm_readvariableop_2_resource7
3sequential_2_dense_9_matmul_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resourceH
Dsequential_2_batch_normalization_4_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_4_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_4_batchnorm_readvariableop_2_resource8
4sequential_2_dense_10_matmul_readvariableop_resource9
5sequential_2_dense_10_biasadd_readvariableop_resourceH
Dsequential_2_batch_normalization_5_batchnorm_readvariableop_resourceL
Hsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceJ
Fsequential_2_batch_normalization_5_batchnorm_readvariableop_1_resourceJ
Fsequential_2_batch_normalization_5_batchnorm_readvariableop_2_resource8
4sequential_2_dense_11_matmul_readvariableop_resource9
5sequential_2_dense_11_biasadd_readvariableop_resource8
4sequential_3_dense_12_matmul_readvariableop_resource9
5sequential_3_dense_12_biasadd_readvariableop_resource8
4sequential_3_dense_13_matmul_readvariableop_resource9
5sequential_3_dense_13_biasadd_readvariableop_resource8
4sequential_3_dense_14_matmul_readvariableop_resource9
5sequential_3_dense_14_biasadd_readvariableop_resource8
4sequential_3_dense_15_matmul_readvariableop_resource9
5sequential_3_dense_15_biasadd_readvariableop_resource
identity¢;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢,sequential_2/dense_10/BiasAdd/ReadVariableOp¢+sequential_2/dense_10/MatMul/ReadVariableOp¢,sequential_2/dense_11/BiasAdd/ReadVariableOp¢+sequential_2/dense_11/MatMul/ReadVariableOp¢+sequential_2/dense_8/BiasAdd/ReadVariableOp¢*sequential_2/dense_8/MatMul/ReadVariableOp¢+sequential_2/dense_9/BiasAdd/ReadVariableOp¢*sequential_2/dense_9/MatMul/ReadVariableOp¢,sequential_3/dense_12/BiasAdd/ReadVariableOp¢+sequential_3/dense_12/MatMul/ReadVariableOp¢,sequential_3/dense_13/BiasAdd/ReadVariableOp¢+sequential_3/dense_13/MatMul/ReadVariableOp¢,sequential_3/dense_14/BiasAdd/ReadVariableOp¢+sequential_3/dense_14/MatMul/ReadVariableOp¢,sequential_3/dense_15/BiasAdd/ReadVariableOp¢+sequential_3/dense_15/MatMul/ReadVariableOpÌ
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp²
sequential_2/dense_8/MatMulMatMulinputs2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/MatMulË
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpÕ
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/Reluû
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp­
2sequential_2/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_3/batchnorm/add/y
0sequential_2/batch_normalization_3/batchnorm/addAddV2Csequential_2/batch_normalization_3/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/addÌ
2sequential_2/batch_normalization_3/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/Rsqrt
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_3/batchnorm/mulMul6sequential_2/batch_normalization_3/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/mul
2sequential_2/batch_normalization_3/batchnorm/mul_1Mul'sequential_2/dense_8/Relu:activations:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_2/batch_normalization_3/batchnorm/mul_1
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1
2sequential_2/batch_normalization_3/batchnorm/mul_2MulEsequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/mul_2
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2
0sequential_2/batch_normalization_3/batchnorm/subSubEsequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/sub
2sequential_2/batch_normalization_3/batchnorm/add_1AddV26sequential_2/batch_normalization_3/batchnorm/mul_1:z:04sequential_2/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_2/batch_normalization_3/batchnorm/add_1Ì
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOpâ
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/MatMulË
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpÕ
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/BiasAdd
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/Reluû
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp­
2sequential_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_4/batchnorm/add/y
0sequential_2/batch_normalization_4/batchnorm/addAddV2Csequential_2/batch_normalization_4/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/addÌ
2sequential_2/batch_normalization_4/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/Rsqrt
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_4/batchnorm/mulMul6sequential_2/batch_normalization_4/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/mul
2sequential_2/batch_normalization_4/batchnorm/mul_1Mul'sequential_2/dense_9/Relu:activations:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_4/batchnorm/mul_1
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1
2sequential_2/batch_normalization_4/batchnorm/mul_2MulEsequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/mul_2
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2
0sequential_2/batch_normalization_4/batchnorm/subSubEsequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/sub
2sequential_2/batch_normalization_4/batchnorm/add_1AddV26sequential_2/batch_normalization_4/batchnorm/mul_1:z:04sequential_2/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_4/batchnorm/add_1Ï
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02-
+sequential_2/dense_10/MatMul/ReadVariableOpå
sequential_2/dense_10/MatMulMatMul6sequential_2/batch_normalization_4/batchnorm/add_1:z:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/MatMulÎ
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOpÙ
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/BiasAdd
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/Reluû
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp­
2sequential_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2sequential_2/batch_normalization_5/batchnorm/add/y
0sequential_2/batch_normalization_5/batchnorm/addAddV2Csequential_2/batch_normalization_5/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/addÌ
2sequential_2/batch_normalization_5/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/Rsqrt
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp
0sequential_2/batch_normalization_5/batchnorm/mulMul6sequential_2/batch_normalization_5/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/mul
2sequential_2/batch_normalization_5/batchnorm/mul_1Mul(sequential_2/dense_10/Relu:activations:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_5/batchnorm/mul_1
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1
2sequential_2/batch_normalization_5/batchnorm/mul_2MulEsequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/mul_2
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2
0sequential_2/batch_normalization_5/batchnorm/subSubEsequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/sub
2sequential_2/batch_normalization_5/batchnorm/add_1AddV26sequential_2/batch_normalization_5/batchnorm/mul_1:z:04sequential_2/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_5/batchnorm/add_1Ï
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOpå
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/MatMulÎ
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/dense_11/BiasAdd/ReadVariableOpÙ
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/BiasAdd
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/TanhÏ
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOpÍ
sequential_3/dense_12/MatMulMatMulsequential_2/dense_11/Tanh:y:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_12/MatMulÎ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_12/BiasAdd/ReadVariableOpÙ
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_12/BiasAdd
sequential_3/re_lu_3/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/re_lu_3/Relu©
sequential_3/dropout_3/IdentityIdentity'sequential_3/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_3/dropout_3/IdentityÏ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp×
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_3/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/dense_13/MatMulÎ
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOpÙ
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/dense_13/BiasAdd
sequential_3/re_lu_4/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
sequential_3/re_lu_4/Relu©
sequential_3/dropout_4/IdentityIdentity'sequential_3/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2!
sequential_3/dropout_4/IdentityÏ
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp×
sequential_3/dense_14/MatMulMatMul(sequential_3/dropout_4/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_14/MatMulÎ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOpÙ
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_14/BiasAdd
sequential_3/re_lu_5/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/re_lu_5/Relu©
sequential_3/dropout_5/IdentityIdentity'sequential_3/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_3/dropout_5/Identity
sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_3/flatten_1/ConstÎ
sequential_3/flatten_1/ReshapeReshape(sequential_3/dropout_5/Identity:output:0%sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_3/flatten_1/ReshapeÏ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOpÖ
sequential_3/dense_15/MatMulMatMul'sequential_3/flatten_1/Reshape:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_15/MatMulÎ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOpÙ
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_15/BiasAddÞ
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2z
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_22
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_22
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_22
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52066

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52148

inputs
assignmovingavg_52123
assignmovingavg_1_52129)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52123*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_52123*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52123*
_output_shapes
:
2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52123*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_52123AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52123*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52129*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_52129*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52129*
_output_shapes
:
2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52129*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_52129AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52129*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ª
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_52388

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_52350

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£<
Ô
__inference__traced_save_52519
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¥	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÏ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*á
_input_shapesÏ
Ì: :::::::
:
:
:
:
:
:

:
:
:
:
:
:
::::<:<:<:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
: 	

_output_shapes
:
: 


_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<: 

_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
Ú
}
(__inference_dense_10_layer_call_fn_52112

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_497622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_49579

inputs
assignmovingavg_49554
assignmovingavg_1_49560)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49554*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49554*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49554*
_output_shapes
:
2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49554*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49554AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49554*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49560*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49560*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49560*
_output_shapes
:
2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49560*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49560AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49560*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

E
)__inference_dropout_5_layer_call_fn_52382

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49472

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

b
)__inference_dropout_5_layer_call_fn_52377

inputs
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
¨
5__inference_batch_normalization_5_layer_call_fn_52194

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_496122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
×+
¨
G__inference_sequential_2_layer_call_and_return_conditional_losses_50042

inputs
dense_8_49994
dense_8_49996
batch_normalization_3_49999
batch_normalization_3_50001
batch_normalization_3_50003
batch_normalization_3_50005
dense_9_50008
dense_9_50010
batch_normalization_4_50013
batch_normalization_4_50015
batch_normalization_4_50017
batch_normalization_4_50019
dense_10_50022
dense_10_50024
batch_normalization_5_50027
batch_normalization_5_50029
batch_normalization_5_50031
batch_normalization_5_50033
dense_11_50036
dense_11_50038
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_49994dense_8_49996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_496382!
dense_8/StatefulPartitionedCall²
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_49999batch_normalization_3_50001batch_normalization_3_50003batch_normalization_3_50005*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_493322/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_50008dense_9_50010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_497002!
dense_9/StatefulPartitionedCall²
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_50013batch_normalization_4_50015batch_normalization_4_50017batch_normalization_4_50019*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494722/
-batch_normalization_4/StatefulPartitionedCallÁ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_50022dense_10_50024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_497622"
 dense_10/StatefulPartitionedCall³
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_50027batch_normalization_5_50029batch_normalization_5_50031batch_normalization_5_50033*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_496122/
-batch_normalization_5/StatefulPartitionedCallÁ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_50036dense_11_50038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_498242"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_52260

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_10_layer_call_and_return_conditional_losses_52103

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
³
¨
5__inference_batch_normalization_4_layer_call_fn_52079

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Ü
C__inference_dense_13_layer_call_and_return_conditional_losses_50168

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_50214

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_49612

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

C
'__inference_re_lu_4_layer_call_fn_52299

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_501892
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
°
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_52294

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
°
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_50120

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_50258

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_52316

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs

E
)__inference_dropout_3_layer_call_fn_52270

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_52255

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_11_layer_call_fn_52214

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_498242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52046

inputs
assignmovingavg_52021
assignmovingavg_1_52027)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52021*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_52021*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52021*
_output_shapes
:
2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52021*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_52021AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52021*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52027*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_52027*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52027*
_output_shapes
:
2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52027*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_52027AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52027*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Â

,__inference_sequential_4_layer_call_fn_51394

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_507952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì+
¯
G__inference_sequential_2_layer_call_and_return_conditional_losses_49892
dense_8_input
dense_8_49844
dense_8_49846
batch_normalization_3_49849
batch_normalization_3_49851
batch_normalization_3_49853
batch_normalization_3_49855
dense_9_49858
dense_9_49860
batch_normalization_4_49863
batch_normalization_4_49865
batch_normalization_4_49867
batch_normalization_4_49869
dense_10_49872
dense_10_49874
batch_normalization_5_49877
batch_normalization_5_49879
batch_normalization_5_49881
batch_normalization_5_49883
dense_11_49886
dense_11_49888
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_49844dense_8_49846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_496382!
dense_8/StatefulPartitionedCall²
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_49849batch_normalization_3_49851batch_normalization_3_49853batch_normalization_3_49855*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_493322/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_49858dense_9_49860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_497002!
dense_9/StatefulPartitionedCall²
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_49863batch_normalization_4_49865batch_normalization_4_49867batch_normalization_4_49869*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494722/
-batch_normalization_4/StatefulPartitionedCallÁ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_49872dense_10_49874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_497622"
 dense_10/StatefulPartitionedCall³
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_49877batch_normalization_5_49879batch_normalization_5_49881batch_normalization_5_49883*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_496122/
-batch_normalization_5/StatefulPartitionedCallÁ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_49886dense_11_49888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_498242"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
Ñ+
¨
G__inference_sequential_2_layer_call_and_return_conditional_losses_49946

inputs
dense_8_49898
dense_8_49900
batch_normalization_3_49903
batch_normalization_3_49905
batch_normalization_3_49907
batch_normalization_3_49909
dense_9_49912
dense_9_49914
batch_normalization_4_49917
batch_normalization_4_49919
batch_normalization_4_49921
batch_normalization_4_49923
dense_10_49926
dense_10_49928
batch_normalization_5_49931
batch_normalization_5_49933
batch_normalization_5_49935
batch_normalization_5_49937
dense_11_49940
dense_11_49942
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_49898dense_8_49900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_496382!
dense_8/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_49903batch_normalization_3_49905batch_normalization_3_49907batch_normalization_3_49909*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_492992/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_49912dense_9_49914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_497002!
dense_9/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_49917batch_normalization_4_49919batch_normalization_4_49921batch_normalization_4_49923*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_494392/
-batch_normalization_4/StatefulPartitionedCallÁ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_49926dense_10_49928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_497622"
 dense_10/StatefulPartitionedCall±
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_49931batch_normalization_5_49933batch_normalization_5_49935batch_normalization_5_49937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_495792/
-batch_normalization_5/StatefulPartitionedCallÁ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_49940dense_11_49942*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_498242"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_13_layer_call_and_return_conditional_losses_52280

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Ô
G__inference_sequential_4_layer_call_and_return_conditional_losses_50795

inputs
sequential_2_50736
sequential_2_50738
sequential_2_50740
sequential_2_50742
sequential_2_50744
sequential_2_50746
sequential_2_50748
sequential_2_50750
sequential_2_50752
sequential_2_50754
sequential_2_50756
sequential_2_50758
sequential_2_50760
sequential_2_50762
sequential_2_50764
sequential_2_50766
sequential_2_50768
sequential_2_50770
sequential_2_50772
sequential_2_50774
sequential_3_50777
sequential_3_50779
sequential_3_50781
sequential_3_50783
sequential_3_50785
sequential_3_50787
sequential_3_50789
sequential_3_50791
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall«
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_50736sequential_2_50738sequential_2_50740sequential_2_50742sequential_2_50744sequential_2_50746sequential_2_50748sequential_2_50750sequential_2_50752sequential_2_50754sequential_2_50756sequential_2_50758sequential_2_50760sequential_2_50762sequential_2_50764sequential_2_50766sequential_2_50768sequential_2_50770sequential_2_50772sequential_2_50774* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_499462&
$sequential_2/StatefulPartitionedCallÐ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_50777sequential_3_50779sequential_3_50781sequential_3_50783sequential_3_50785sequential_3_50787sequential_3_50789sequential_3_50791*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504022&
$sequential_3/StatefulPartitionedCallÏ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ/

G__inference_sequential_3_layer_call_and_return_conditional_losses_50337
dense_12_input
dense_12_50110
dense_12_50112
dense_13_50179
dense_13_50181
dense_14_50248
dense_14_50250
dense_15_50331
dense_15_50333
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_50110dense_12_50112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_500992"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_501202
re_lu_3/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501402#
!dropout_3/StatefulPartitionedCallµ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_50179dense_13_50181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_501682"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_501892
re_lu_4/PartitionedCall¬
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502092#
!dropout_4/StatefulPartitionedCallµ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_14_50248dense_14_50250*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_502372"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_502582
re_lu_5/PartitionedCall¬
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502782#
!dropout_5/StatefulPartitionedCallú
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_503022
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_50331dense_15_50333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_503202"
 dense_15/StatefulPartitionedCallõ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_12_input
Ú
}
(__inference_dense_14_layer_call_fn_52345

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_502372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
×
Ô
G__inference_sequential_4_layer_call_and_return_conditional_losses_50918

inputs
sequential_2_50859
sequential_2_50861
sequential_2_50863
sequential_2_50865
sequential_2_50867
sequential_2_50869
sequential_2_50871
sequential_2_50873
sequential_2_50875
sequential_2_50877
sequential_2_50879
sequential_2_50881
sequential_2_50883
sequential_2_50885
sequential_2_50887
sequential_2_50889
sequential_2_50891
sequential_2_50893
sequential_2_50895
sequential_2_50897
sequential_3_50900
sequential_3_50902
sequential_3_50904
sequential_3_50906
sequential_3_50908
sequential_3_50910
sequential_3_50912
sequential_3_50914
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall±
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_50859sequential_2_50861sequential_2_50863sequential_2_50865sequential_2_50867sequential_2_50869sequential_2_50871sequential_2_50873sequential_2_50875sequential_2_50877sequential_2_50879sequential_2_50881sequential_2_50883sequential_2_50885sequential_2_50887sequential_2_50889sequential_2_50891sequential_2_50893sequential_2_50895sequential_2_50897* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_500422&
$sequential_2/StatefulPartitionedCallÐ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_50900sequential_3_50902sequential_3_50904sequential_3_50906sequential_3_50908sequential_3_50910sequential_3_50912sequential_3_50914*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504542&
$sequential_3/StatefulPartitionedCallÏ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
G
Þ
G__inference_sequential_3_layer_call_and_return_conditional_losses_51810

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAddq
re_lu_3/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_3/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/dropout/Const¥
dropout_3/dropout/MulMulre_lu_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÒ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_3/dropout/GreaterEqual/yæ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Cast¢
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Mul_1¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dense_13/BiasAddq
re_lu_4/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
re_lu_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/dropout/Const¥
dropout_4/dropout/MulMulre_lu_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeÒ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_4/dropout/GreaterEqual/yæ
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2 
dropout_4/dropout/GreaterEqual
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_4/dropout/Cast¢
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_4/dropout/Mul_1¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldropout_4/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/BiasAddq
re_lu_5/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_5/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/dropout/Const¥
dropout_5/dropout/MulMulre_lu_5/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÒ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_5/dropout/GreaterEqual/yæ
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Cast¢
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp¢
dense_15/MatMulMatMulflatten_1/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddù
IdentityIdentitydense_15/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_52311

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs

E
)__inference_flatten_1_layer_call_fn_52393

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_503022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
¨
5__inference_batch_normalization_5_layer_call_fn_52181

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_495792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
+
Þ
G__inference_sequential_3_layer_call_and_return_conditional_losses_51846

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¨
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/MatMul§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp¥
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAddq
re_lu_3/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_3/Relu
dropout_3/IdentityIdentityre_lu_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Identity¨
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldropout_3/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_13/BiasAdd/ReadVariableOp¥
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dense_13/BiasAddq
re_lu_4/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
re_lu_4/Relu
dropout_4/IdentityIdentityre_lu_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_4/Identity¨
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldropout_4/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp¥
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/BiasAddq
re_lu_5/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_5/Relu
dropout_5/IdentityIdentityre_lu_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp¢
dense_15/MatMulMatMulflatten_1/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¥
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddù
IdentityIdentitydense_15/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
¤
,__inference_sequential_4_layer_call_fn_50854
sequential_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_507952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input
Ç
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_50283

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_sequential_4_layer_call_fn_51455

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_509182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_52372

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_52321

inputs
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
û
à
G__inference_sequential_4_layer_call_and_return_conditional_losses_50730
sequential_2_input
sequential_2_50671
sequential_2_50673
sequential_2_50675
sequential_2_50677
sequential_2_50679
sequential_2_50681
sequential_2_50683
sequential_2_50685
sequential_2_50687
sequential_2_50689
sequential_2_50691
sequential_2_50693
sequential_2_50695
sequential_2_50697
sequential_2_50699
sequential_2_50701
sequential_2_50703
sequential_2_50705
sequential_2_50707
sequential_2_50709
sequential_3_50712
sequential_3_50714
sequential_3_50716
sequential_3_50718
sequential_3_50720
sequential_3_50722
sequential_3_50724
sequential_3_50726
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall½
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputsequential_2_50671sequential_2_50673sequential_2_50675sequential_2_50677sequential_2_50679sequential_2_50681sequential_2_50683sequential_2_50685sequential_2_50687sequential_2_50689sequential_2_50691sequential_2_50693sequential_2_50695sequential_2_50697sequential_2_50699sequential_2_50701sequential_2_50703sequential_2_50705sequential_2_50707sequential_2_50709* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_500422&
$sequential_2/StatefulPartitionedCallÐ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_50712sequential_3_50714sequential_3_50716sequential_3_50718sequential_3_50720sequential_3_50722sequential_3_50724sequential_3_50726*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504542&
$sequential_3/StatefulPartitionedCallÏ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_52367

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_50302

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
}
(__inference_dense_13_layer_call_fn_52289

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_501682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51964

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì	
Û
B__inference_dense_8_layer_call_and_return_conditional_losses_49638

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý*

G__inference_sequential_3_layer_call_and_return_conditional_losses_50368
dense_12_input
dense_12_50340
dense_12_50342
dense_13_50347
dense_13_50349
dense_14_50354
dense_14_50356
dense_15_50362
dense_15_50364
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_50340dense_12_50342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_500992"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_501202
re_lu_3/PartitionedCallð
dropout_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501452
dropout_3/PartitionedCall­
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_50347dense_13_50349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_501682"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_501892
re_lu_4/PartitionedCallð
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502142
dropout_4/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_14_50354dense_14_50356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_502372"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_502582
re_lu_5/PartitionedCallð
dropout_5/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502832
dropout_5/PartitionedCallò
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_503022
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_50362dense_15_50364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_503202"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_12_input
ì	
Û
B__inference_dense_9_layer_call_and_return_conditional_losses_49700

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã	
Ü
C__inference_dense_11_layer_call_and_return_conditional_losses_52205

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
£
Û
,__inference_sequential_3_layer_call_fn_51888

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

,__inference_sequential_2_layer_call_fn_50085
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_500422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
£
Û
,__inference_sequential_3_layer_call_fn_51867

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
¨
5__inference_batch_normalization_3_layer_call_fn_51977

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_492992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_50189

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
	
Ü
C__inference_dense_14_layer_call_and_return_conditional_losses_50237

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
ì
¤
,__inference_sequential_4_layer_call_fn_50977
sequential_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_509182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input
Ø
|
'__inference_dense_8_layer_call_fn_51908

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_496382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

C
'__inference_re_lu_3_layer_call_fn_52243

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_501202
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

,__inference_sequential_2_layer_call_fn_51753

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_500422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_52238

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_15_layer_call_and_return_conditional_losses_52403

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_49439

inputs
assignmovingavg_49414
assignmovingavg_1_49420)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49414*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49414*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49414*
_output_shapes
:
2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49414*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49414AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49414*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49420*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49420*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49420*
_output_shapes
:
2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49420*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49420AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49420*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
/
þ
G__inference_sequential_3_layer_call_and_return_conditional_losses_50402

inputs
dense_12_50374
dense_12_50376
dense_13_50381
dense_13_50383
dense_14_50388
dense_14_50390
dense_15_50396
dense_15_50398
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_50374dense_12_50376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_500992"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_501202
re_lu_3/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501402#
!dropout_3/StatefulPartitionedCallµ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_50381dense_13_50383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_501682"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_501892
re_lu_4/PartitionedCall¬
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502092#
!dropout_4/StatefulPartitionedCallµ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_14_50388dense_14_50390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_502372"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_502582
re_lu_5/PartitionedCall¬
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502782#
!dropout_5/StatefulPartitionedCallú
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_503022
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_50396dense_15_50398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_503202"
 dense_15/StatefulPartitionedCallõ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
ã
,__inference_sequential_3_layer_call_fn_50421
dense_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_12_input
Å*

G__inference_sequential_3_layer_call_and_return_conditional_losses_50454

inputs
dense_12_50426
dense_12_50428
dense_13_50433
dense_13_50435
dense_14_50440
dense_14_50442
dense_15_50448
dense_15_50450
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_50426dense_12_50428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_500992"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_501202
re_lu_3/PartitionedCallð
dropout_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501452
dropout_3/PartitionedCall­
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_50433dense_13_50435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_501682"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_501892
re_lu_4/PartitionedCallð
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502142
dropout_4/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_14_50440dense_14_50442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_502372"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_502582
re_lu_5/PartitionedCallð
dropout_5/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_502832
dropout_5/PartitionedCallò
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_503022
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_50448dense_15_50450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_503202"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò÷
» 
 __inference__wrapped_model_49203
sequential_2_inputD
@sequential_4_sequential_2_dense_8_matmul_readvariableop_resourceE
Asequential_4_sequential_2_dense_8_biasadd_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_2_resourceD
@sequential_4_sequential_2_dense_9_matmul_readvariableop_resourceE
Asequential_4_sequential_2_dense_9_biasadd_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_2_resourceE
Asequential_4_sequential_2_dense_10_matmul_readvariableop_resourceF
Bsequential_4_sequential_2_dense_10_biasadd_readvariableop_resourceU
Qsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_resourceY
Usequential_4_sequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceW
Ssequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_1_resourceW
Ssequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_2_resourceE
Asequential_4_sequential_2_dense_11_matmul_readvariableop_resourceF
Bsequential_4_sequential_2_dense_11_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_12_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_12_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_13_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_13_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_14_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_14_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_15_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_15_biasadd_readvariableop_resource
identity¢Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp¢8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp¢9sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp¢8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp¢8sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp¢7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp¢8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp¢7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOpó
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOpå
(sequential_4/sequential_2/dense_8/MatMulMatMulsequential_2_input?sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_4/sequential_2/dense_8/MatMulò
8sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp
)sequential_4/sequential_2/dense_8/BiasAddBiasAdd2sequential_4/sequential_2/dense_8/MatMul:product:0@sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/sequential_2/dense_8/BiasAdd¾
&sequential_4/sequential_2/dense_8/ReluRelu2sequential_4/sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_4/sequential_2/dense_8/Relu¢
Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpÇ
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add/yÈ
=sequential_4/sequential_2/batch_normalization_3/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_3/batchnorm/addó
?sequential_4/sequential_2/batch_normalization_3/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/Rsqrt®
Lsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpÅ
=sequential_4/sequential_2/batch_normalization_3/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_3/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_3/batchnorm/mul´
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1Mul4sequential_4/sequential_2/dense_8/Relu:activations:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1¨
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1Å
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2¨
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2Ã
=sequential_4/sequential_2/batch_normalization_3/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_3/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add_1ó
7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype029
7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp
(sequential_4/sequential_2/dense_9/MatMulMatMulCsequential_4/sequential_2/batch_normalization_3/batchnorm/add_1:z:0?sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(sequential_4/sequential_2/dense_9/MatMulò
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02:
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp
)sequential_4/sequential_2/dense_9/BiasAddBiasAdd2sequential_4/sequential_2/dense_9/MatMul:product:0@sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2+
)sequential_4/sequential_2/dense_9/BiasAdd¾
&sequential_4/sequential_2/dense_9/ReluRelu2sequential_4/sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&sequential_4/sequential_2/dense_9/Relu¢
Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02J
Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpÇ
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add/yÈ
=sequential_4/sequential_2/batch_normalization_4/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_4/batchnorm/addó
?sequential_4/sequential_2/batch_normalization_4/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/Rsqrt®
Lsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02N
Lsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpÅ
=sequential_4/sequential_2/batch_normalization_4/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_4/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_4/batchnorm/mul´
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1Mul4sequential_4/sequential_2/dense_9/Relu:activations:0Asequential_4/sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1¨
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1Å
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_2¨
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2Ã
=sequential_4/sequential_2/batch_normalization_4/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_4/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add_1ö
8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02:
8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp
)sequential_4/sequential_2/dense_10/MatMulMatMulCsequential_4/sequential_2/batch_normalization_4/batchnorm/add_1:z:0@sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2+
)sequential_4/sequential_2/dense_10/MatMulõ
9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp
*sequential_4/sequential_2/dense_10/BiasAddBiasAdd3sequential_4/sequential_2/dense_10/MatMul:product:0Asequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2,
*sequential_4/sequential_2/dense_10/BiasAddÁ
'sequential_4/sequential_2/dense_10/ReluRelu3sequential_4/sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2)
'sequential_4/sequential_2/dense_10/Relu¢
Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02J
Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpÇ
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add/yÈ
=sequential_4/sequential_2/batch_normalization_5/batchnorm/addAddV2Psequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp:value:0Hsequential_4/sequential_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_5/batchnorm/addó
?sequential_4/sequential_2/batch_normalization_5/batchnorm/RsqrtRsqrtAsequential_4/sequential_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/Rsqrt®
Lsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpUsequential_4_sequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02N
Lsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpÅ
=sequential_4/sequential_2/batch_normalization_5/batchnorm/mulMulCsequential_4/sequential_2/batch_normalization_5/batchnorm/Rsqrt:y:0Tsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_5/batchnorm/mulµ
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_1Mul5sequential_4/sequential_2/dense_10/Relu:activations:0Asequential_4/sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_1¨
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1Å
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_2¨
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2Ã
=sequential_4/sequential_2/batch_normalization_5/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_5/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_5/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add_1ö
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02:
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp
)sequential_4/sequential_2/dense_11/MatMulMatMulCsequential_4/sequential_2/batch_normalization_5/batchnorm/add_1:z:0@sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/sequential_2/dense_11/MatMulõ
9sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp
*sequential_4/sequential_2/dense_11/BiasAddBiasAdd3sequential_4/sequential_2/dense_11/MatMul:product:0Asequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_4/sequential_2/dense_11/BiasAddÁ
'sequential_4/sequential_2/dense_11/TanhTanh3sequential_4/sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_4/sequential_2/dense_11/Tanhö
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_12/MatMulMatMul+sequential_4/sequential_2/dense_11/Tanh:y:0@sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/sequential_3/dense_12/MatMulõ
9sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_12/BiasAddBiasAdd3sequential_4/sequential_3/dense_12/MatMul:product:0Asequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_4/sequential_3/dense_12/BiasAdd¿
&sequential_4/sequential_3/re_lu_3/ReluRelu3sequential_4/sequential_3/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_4/sequential_3/re_lu_3/ReluÐ
,sequential_4/sequential_3/dropout_3/IdentityIdentity4sequential_4/sequential_3/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/sequential_3/dropout_3/Identityö
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02:
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_13/MatMulMatMul5sequential_4/sequential_3/dropout_3/Identity:output:0@sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2+
)sequential_4/sequential_3/dense_13/MatMulõ
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_13/BiasAddBiasAdd3sequential_4/sequential_3/dense_13/MatMul:product:0Asequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2,
*sequential_4/sequential_3/dense_13/BiasAdd¿
&sequential_4/sequential_3/re_lu_4/ReluRelu3sequential_4/sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2(
&sequential_4/sequential_3/re_lu_4/ReluÐ
,sequential_4/sequential_3/dropout_4/IdentityIdentity4sequential_4/sequential_3/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2.
,sequential_4/sequential_3/dropout_4/Identityö
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02:
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_14/MatMulMatMul5sequential_4/sequential_3/dropout_4/Identity:output:0@sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/sequential_3/dense_14/MatMulõ
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_14/BiasAddBiasAdd3sequential_4/sequential_3/dense_14/MatMul:product:0Asequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_4/sequential_3/dense_14/BiasAdd¿
&sequential_4/sequential_3/re_lu_5/ReluRelu3sequential_4/sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_4/sequential_3/re_lu_5/ReluÐ
,sequential_4/sequential_3/dropout_5/IdentityIdentity4sequential_4/sequential_3/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/sequential_3/dropout_5/Identity§
)sequential_4/sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2+
)sequential_4/sequential_3/flatten_1/Const
+sequential_4/sequential_3/flatten_1/ReshapeReshape5sequential_4/sequential_3/dropout_5/Identity:output:02sequential_4/sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_4/sequential_3/flatten_1/Reshapeö
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_15/MatMulMatMul4sequential_4/sequential_3/flatten_1/Reshape:output:0@sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/sequential_3/dense_15/MatMulõ
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_15/BiasAddBiasAdd3sequential_4/sequential_3/dense_15/MatMul:product:0Asequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_4/sequential_3/dense_15/BiasAdd×
IdentityIdentity3sequential_4/sequential_3/dense_15/BiasAdd:output:0I^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:^sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp9^sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp:^sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp9^sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp9^sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp8^sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp9^sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp8^sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2
Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp2
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_12
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_22
Lsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp2
Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp2
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_12
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_22
Lsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2
Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpHsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp2
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_12
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_22
Lsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpLsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2v
9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp2v
9sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp9sequential_4/sequential_2/dense_11/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp2t
8sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp8sequential_4/sequential_2/dense_8/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp2t
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_12/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_50140

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

#__inference_signature_wrapper_51040
sequential_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_492032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_50209

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
»
ã
,__inference_sequential_3_layer_call_fn_50473
dense_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_12_input
õ
à
G__inference_sequential_4_layer_call_and_return_conditional_losses_50668
sequential_2_input
sequential_2_50567
sequential_2_50569
sequential_2_50571
sequential_2_50573
sequential_2_50575
sequential_2_50577
sequential_2_50579
sequential_2_50581
sequential_2_50583
sequential_2_50585
sequential_2_50587
sequential_2_50589
sequential_2_50591
sequential_2_50593
sequential_2_50595
sequential_2_50597
sequential_2_50599
sequential_2_50601
sequential_2_50603
sequential_2_50605
sequential_3_50650
sequential_3_50652
sequential_3_50654
sequential_3_50656
sequential_3_50658
sequential_3_50660
sequential_3_50662
sequential_3_50664
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall·
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputsequential_2_50567sequential_2_50569sequential_2_50571sequential_2_50573sequential_2_50575sequential_2_50577sequential_2_50579sequential_2_50581sequential_2_50583sequential_2_50585sequential_2_50587sequential_2_50589sequential_2_50591sequential_2_50593sequential_2_50595sequential_2_50597sequential_2_50599sequential_2_50601sequential_2_50603sequential_2_50605* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_499462&
$sequential_2/StatefulPartitionedCallÐ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_50650sequential_3_50652sequential_3_50654sequential_3_50656sequential_3_50658sequential_3_50660sequential_3_50662sequential_3_50664*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_504022&
$sequential_3/StatefulPartitionedCallÏ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:[ W
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namesequential_2_input
ì	
Û
B__inference_dense_9_layer_call_and_return_conditional_losses_52001

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ü
C__inference_dense_12_layer_call_and_return_conditional_losses_52224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ðs
½
!__inference__traced_restore_52613
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias2
.assignvariableop_2_batch_normalization_3_gamma1
-assignvariableop_3_batch_normalization_3_beta8
4assignvariableop_4_batch_normalization_3_moving_mean<
8assignvariableop_5_batch_normalization_3_moving_variance%
!assignvariableop_6_dense_9_kernel#
assignvariableop_7_dense_9_bias2
.assignvariableop_8_batch_normalization_4_gamma1
-assignvariableop_9_batch_normalization_4_beta9
5assignvariableop_10_batch_normalization_4_moving_mean=
9assignvariableop_11_batch_normalization_4_moving_variance'
#assignvariableop_12_dense_10_kernel%
!assignvariableop_13_dense_10_bias3
/assignvariableop_14_batch_normalization_5_gamma2
.assignvariableop_15_batch_normalization_5_beta9
5assignvariableop_16_batch_normalization_5_moving_mean=
9assignvariableop_17_batch_normalization_5_moving_variance'
#assignvariableop_18_dense_11_kernel%
!assignvariableop_19_dense_11_bias'
#assignvariableop_20_dense_12_kernel%
!assignvariableop_21_dense_12_bias'
#assignvariableop_22_dense_13_kernel%
!assignvariableop_23_dense_13_bias'
#assignvariableop_24_dense_14_kernel%
!assignvariableop_25_dense_14_bias'
#assignvariableop_26_dense_15_kernel%
!assignvariableop_27_dense_15_bias
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¥	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices½
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16½
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Á
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_11_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_11_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_12_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_12_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22«
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_13_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23©
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_13_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_14_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25©
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_14_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26«
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_15_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27©
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_15_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÆ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28¹
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¯

,__inference_sequential_2_layer_call_fn_51708

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_499462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51944

inputs
assignmovingavg_51919
assignmovingavg_1_51925)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51919*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_51919*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51919*
_output_shapes
:2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51919*
_output_shapes
:2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_51919AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51919*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51925*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_51925*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51925*
_output_shapes
:2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51925*
_output_shapes
:2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_51925AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51925*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø/
Å
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_49299

inputs
assignmovingavg_49274
assignmovingavg_1_49280)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ë
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49274*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49274*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpð
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49274*
_output_shapes
:2
AssignMovingAvg/subç
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/49274*
_output_shapes
:2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49274AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/49274*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÑ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49280*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49280*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpú
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49280*
_output_shapes
:2
AssignMovingAvg_1/subñ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/49280*
_output_shapes
:2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49280AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/49280*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
úñ
ì
G__inference_sequential_2_layer_call_and_return_conditional_losses_51583

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource/
+batch_normalization_3_assignmovingavg_514731
-batch_normalization_3_assignmovingavg_1_51479?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource/
+batch_normalization_4_assignmovingavg_515121
-batch_normalization_4_assignmovingavg_1_51518?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource/
+batch_normalization_5_assignmovingavg_515511
-batch_normalization_5_assignmovingavg_1_51557?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/Relu¶
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indiceså
"batch_normalization_3/moments/meanMeandense_8/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_3/moments/mean¾
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_3/moments/StopGradientú
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_3/moments/SquaredDifference¾
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_3/moments/varianceÂ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeÊ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/51473*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_3/AssignMovingAvg/decayÔ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_51473*
_output_shapes
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpÞ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/51473*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/subÕ
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/51473*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/mul±
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_51473-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/51473*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/51479*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_3/AssignMovingAvg_1/decayÚ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_51479*
_output_shapes
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpè
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/51479*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/subß
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/51479*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/mul½
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_51479/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/51479*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yÚ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulÌ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Ó
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ô
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpÙ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subÝ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/Relu¶
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indiceså
"batch_normalization_4/moments/meanMeandense_9/Relu:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_4/moments/mean¾
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_4/moments/StopGradientú
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
21
/batch_normalization_4/moments/SquaredDifference¾
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_4/moments/varianceÂ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeÊ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/51512*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayÔ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_51512*
_output_shapes
:
*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpÞ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/51512*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/subÕ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/51512*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/mul±
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_51512-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/51512*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/51518*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayÚ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_51518*
_output_shapes
:
*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpè
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/51518*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/subß
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/51518*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/mul½
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_51518/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/51518*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yÚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulÌ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ô
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpÙ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/add_1¨
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_10/MatMul/ReadVariableOp±
dense_10/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/Relu¶
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesæ
"batch_normalization_5/moments/meanMeandense_10/Relu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_5/moments/mean¾
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_5/moments/StopGradientû
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
21
/batch_normalization_5/moments/SquaredDifference¾
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indices
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_5/moments/varianceÂ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeÊ
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1
+batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/51551*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_5/AssignMovingAvg/decayÔ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_5_assignmovingavg_51551*
_output_shapes
:
*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpÞ
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/51551*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/subÕ
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/51551*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/mul±
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_5_assignmovingavg_51551-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/51551*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/51557*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_5/AssignMovingAvg_1/decayÚ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_5_assignmovingavg_1_51557*
_output_shapes
:
*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpè
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/51557*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/subß
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/51557*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/mul½
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_5_assignmovingavg_1_51557/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/51557*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yÚ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulÍ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/mul_1Ó
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ô
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpÙ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/add_1¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/TanhÝ

IdentityIdentitydense_11/Tanh:y:0:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

E
)__inference_dropout_4_layer_call_fn_52326

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_502142
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
	
Ü
C__inference_dense_14_layer_call_and_return_conditional_losses_52336

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs

b
)__inference_dropout_3_layer_call_fn_52265

inputs
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_501402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Å
serving_default±
Q
sequential_2_input;
$serving_default_sequential_2_input:0ÿÿÿÿÿÿÿÿÿ@
sequential_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ö
¹t
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+ò&call_and_return_all_conditional_losses
ó_default_save_signature
ô__call__"¬r
_tf_keras_sequentialr{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_2_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_2_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}}
A
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
trainable_variables
regularization_losses
	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"ö=
_tf_keras_sequential×={"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
7
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
 regularization_losses
!	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"4
_tf_keras_sequentialç3{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ö
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
:24
;25
<26
=27"
trackable_list_wrapper
Æ
"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513
614
715
816
917
:18
;19
<20
=21"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
>metrics

?layers
	variables
trainable_variables
@layer_metrics
regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables
ô__call__
ó_default_save_signature
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
-
ùserving_default"
signature_map
á

"kernel
#bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
+ú&call_and_return_all_conditional_losses
û__call__"º
_tf_keras_layer {"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
´	
Gaxis
	$gamma
%beta
&moving_mean
'moving_variance
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
+ü&call_and_return_all_conditional_losses
ý__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
ò

(kernel
)bias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+þ&call_and_return_all_conditional_losses
ÿ__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
´	
Paxis
	*gamma
+beta
,moving_mean
-moving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
+&call_and_return_all_conditional_losses
__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ô

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
´	
Yaxis
	0gamma
1beta
2moving_mean
3moving_variance
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
+&call_and_return_all_conditional_losses
__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ó

4kernel
5bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
+&call_and_return_all_conditional_losses
__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
¶
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519"
trackable_list_wrapper

"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513"
trackable_list_wrapper
 "
trackable_list_wrapper
°
bmetrics

clayers
	variables
trainable_variables
dlayer_metrics
regularization_losses
elayer_regularization_losses
fnon_trainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
å

6kernel
7bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
+&call_and_return_all_conditional_losses
__call__"¾
_tf_keras_layer¤{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
í
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
+&call_and_return_all_conditional_losses
__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ç
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ö

8kernel
9bias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
+&call_and_return_all_conditional_losses
__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
í
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
+&call_and_return_all_conditional_losses
__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ç
{	variables
|trainable_variables
}regularization_losses
~	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ù

:kernel
;bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
ñ
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ë
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ì
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ù

<kernel
=bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
X
60
71
82
93
:4
;5
<6
=7"
trackable_list_wrapper
X
60
71
82
93
:4
;5
<6
=7"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
layers
	variables
trainable_variables
layer_metrics
 regularization_losses
 layer_regularization_losses
non_trainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 :2dense_8/kernel
:2dense_8/bias
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
 :
2dense_9/kernel
:
2dense_9/bias
):'
2batch_normalization_4/gamma
(:&
2batch_normalization_4/beta
1:/
 (2!batch_normalization_4/moving_mean
5:3
 (2%batch_normalization_4/moving_variance
!:

2dense_10/kernel
:
2dense_10/bias
):'
2batch_normalization_5/gamma
(:&
2batch_normalization_5/beta
1:/
 (2!batch_normalization_5/moving_mean
5:3
 (2%batch_normalization_5/moving_variance
!:
2dense_11/kernel
:2dense_11/bias
!:2dense_12/kernel
:2dense_12/bias
!:<2dense_13/kernel
:<2dense_13/bias
!:<2dense_14/kernel
:2dense_14/bias
!:2dense_15/kernel
:2dense_15/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
&0
'1
,2
-3
24
35"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
layers
C	variables
Dtrainable_variables
layer_metrics
Eregularization_losses
 layer_regularization_losses
non_trainable_variables
û__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
layers
H	variables
Itrainable_variables
layer_metrics
Jregularization_losses
  layer_regularization_losses
¡non_trainable_variables
ý__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢metrics
£layers
L	variables
Mtrainable_variables
¤layer_metrics
Nregularization_losses
 ¥layer_regularization_losses
¦non_trainable_variables
ÿ__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§metrics
¨layers
Q	variables
Rtrainable_variables
©layer_metrics
Sregularization_losses
 ªlayer_regularization_losses
«non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬metrics
­layers
U	variables
Vtrainable_variables
®layer_metrics
Wregularization_losses
 ¯layer_regularization_losses
°non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±metrics
²layers
Z	variables
[trainable_variables
³layer_metrics
\regularization_losses
 ´layer_regularization_losses
µnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶metrics
·layers
^	variables
_trainable_variables
¸layer_metrics
`regularization_losses
 ¹layer_regularization_losses
ºnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
&0
'1
,2
-3
24
35"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»metrics
¼layers
g	variables
htrainable_variables
½layer_metrics
iregularization_losses
 ¾layer_regularization_losses
¿non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Àmetrics
Álayers
k	variables
ltrainable_variables
Âlayer_metrics
mregularization_losses
 Ãlayer_regularization_losses
Änon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Åmetrics
Ælayers
o	variables
ptrainable_variables
Çlayer_metrics
qregularization_losses
 Èlayer_regularization_losses
Énon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Êmetrics
Ëlayers
s	variables
ttrainable_variables
Ìlayer_metrics
uregularization_losses
 Ílayer_regularization_losses
Înon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ïmetrics
Ðlayers
w	variables
xtrainable_variables
Ñlayer_metrics
yregularization_losses
 Òlayer_regularization_losses
Ónon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ômetrics
Õlayers
{	variables
|trainable_variables
Ölayer_metrics
}regularization_losses
 ×layer_regularization_losses
Ønon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
Ùmetrics
Úlayers
	variables
trainable_variables
Ûlayer_metrics
regularization_losses
 Ülayer_regularization_losses
Ýnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þmetrics
ßlayers
	variables
trainable_variables
àlayer_metrics
regularization_losses
 álayer_regularization_losses
ânon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãmetrics
älayers
	variables
trainable_variables
ålayer_metrics
regularization_losses
 ælayer_regularization_losses
çnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
èmetrics
élayers
	variables
trainable_variables
êlayer_metrics
regularization_losses
 ëlayer_regularization_losses
ìnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ímetrics
îlayers
	variables
trainable_variables
ïlayer_metrics
regularization_losses
 ðlayer_regularization_losses
ñnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ê2ç
G__inference_sequential_4_layer_call_and_return_conditional_losses_51221
G__inference_sequential_4_layer_call_and_return_conditional_losses_51333
G__inference_sequential_4_layer_call_and_return_conditional_losses_50730
G__inference_sequential_4_layer_call_and_return_conditional_losses_50668À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
é2æ
 __inference__wrapped_model_49203Á
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *1¢.
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_4_layer_call_fn_50977
,__inference_sequential_4_layer_call_fn_51394
,__inference_sequential_4_layer_call_fn_51455
,__inference_sequential_4_layer_call_fn_50854À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_49841
G__inference_sequential_2_layer_call_and_return_conditional_losses_51663
G__inference_sequential_2_layer_call_and_return_conditional_losses_49892
G__inference_sequential_2_layer_call_and_return_conditional_losses_51583À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
,__inference_sequential_2_layer_call_fn_51753
,__inference_sequential_2_layer_call_fn_51708
,__inference_sequential_2_layer_call_fn_50085
,__inference_sequential_2_layer_call_fn_49989À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_3_layer_call_and_return_conditional_losses_50337
G__inference_sequential_3_layer_call_and_return_conditional_losses_50368
G__inference_sequential_3_layer_call_and_return_conditional_losses_51846
G__inference_sequential_3_layer_call_and_return_conditional_losses_51810À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
,__inference_sequential_3_layer_call_fn_50473
,__inference_sequential_3_layer_call_fn_51888
,__inference_sequential_3_layer_call_fn_51867
,__inference_sequential_3_layer_call_fn_50421À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÕBÒ
#__inference_signature_wrapper_51040sequential_2_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_51899¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_8_layer_call_fn_51908¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51964
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51944´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_batch_normalization_3_layer_call_fn_51977
5__inference_batch_normalization_3_layer_call_fn_51990´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_dense_9_layer_call_and_return_conditional_losses_52001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_9_layer_call_fn_52010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52046
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52066´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_batch_normalization_4_layer_call_fn_52092
5__inference_batch_normalization_4_layer_call_fn_52079´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_10_layer_call_and_return_conditional_losses_52103¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_10_layer_call_fn_52112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52148
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52168´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_batch_normalization_5_layer_call_fn_52194
5__inference_batch_normalization_5_layer_call_fn_52181´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_11_layer_call_and_return_conditional_losses_52205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_11_layer_call_fn_52214¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_12_layer_call_and_return_conditional_losses_52224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_12_layer_call_fn_52233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_re_lu_3_layer_call_and_return_conditional_losses_52238¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_re_lu_3_layer_call_fn_52243¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Æ2Ã
D__inference_dropout_3_layer_call_and_return_conditional_losses_52255
D__inference_dropout_3_layer_call_and_return_conditional_losses_52260´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_3_layer_call_fn_52265
)__inference_dropout_3_layer_call_fn_52270´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_13_layer_call_and_return_conditional_losses_52280¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_13_layer_call_fn_52289¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_re_lu_4_layer_call_and_return_conditional_losses_52294¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_re_lu_4_layer_call_fn_52299¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Æ2Ã
D__inference_dropout_4_layer_call_and_return_conditional_losses_52316
D__inference_dropout_4_layer_call_and_return_conditional_losses_52311´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_4_layer_call_fn_52321
)__inference_dropout_4_layer_call_fn_52326´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_14_layer_call_and_return_conditional_losses_52336¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_14_layer_call_fn_52345¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_re_lu_5_layer_call_and_return_conditional_losses_52350¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_re_lu_5_layer_call_fn_52355¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Æ2Ã
D__inference_dropout_5_layer_call_and_return_conditional_losses_52372
D__inference_dropout_5_layer_call_and_return_conditional_losses_52367´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_5_layer_call_fn_52377
)__inference_dropout_5_layer_call_fn_52382´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_flatten_1_layer_call_and_return_conditional_losses_52388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_flatten_1_layer_call_fn_52393¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_15_layer_call_and_return_conditional_losses_52403¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_15_layer_call_fn_52412¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ½
 __inference__wrapped_model_49203"#'$&%()-*,+./3021456789:;<=;¢8
1¢.
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
sequential_3&#
sequential_3ÿÿÿÿÿÿÿÿÿ¶
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51944b&'$%3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_51964b'$&%3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_batch_normalization_3_layer_call_fn_51977U&'$%3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
5__inference_batch_normalization_3_layer_call_fn_51990U'$&%3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52046b,-*+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_52066b-*,+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
5__inference_batch_normalization_4_layer_call_fn_52079U,-*+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ

5__inference_batch_normalization_4_layer_call_fn_52092U-*,+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¶
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52148b23013¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¶
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_52168b30213¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
5__inference_batch_normalization_5_layer_call_fn_52181U23013¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ

5__inference_batch_normalization_5_layer_call_fn_52194U30213¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
£
C__inference_dense_10_layer_call_and_return_conditional_losses_52103\.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 {
(__inference_dense_10_layer_call_fn_52112O.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
£
C__inference_dense_11_layer_call_and_return_conditional_losses_52205\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_11_layer_call_fn_52214O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_12_layer_call_and_return_conditional_losses_52224\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_12_layer_call_fn_52233O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_13_layer_call_and_return_conditional_losses_52280\89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ<
 {
(__inference_dense_13_layer_call_fn_52289O89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ<£
C__inference_dense_14_layer_call_and_return_conditional_losses_52336\:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ<
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_14_layer_call_fn_52345O:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_15_layer_call_and_return_conditional_losses_52403\<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_15_layer_call_fn_52412O<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_8_layer_call_and_return_conditional_losses_51899\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_8_layer_call_fn_51908O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_9_layer_call_and_return_conditional_losses_52001\()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_dense_9_layer_call_fn_52010O()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_52255\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_52260\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dropout_3_layer_call_fn_52265O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ|
)__inference_dropout_3_layer_call_fn_52270O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dropout_4_layer_call_and_return_conditional_losses_52311\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ<
 ¤
D__inference_dropout_4_layer_call_and_return_conditional_losses_52316\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ<
 |
)__inference_dropout_4_layer_call_fn_52321O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "ÿÿÿÿÿÿÿÿÿ<|
)__inference_dropout_4_layer_call_fn_52326O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "ÿÿÿÿÿÿÿÿÿ<¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_52367\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_52372\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dropout_5_layer_call_fn_52377O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ|
)__inference_dropout_5_layer_call_fn_52382O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ 
D__inference_flatten_1_layer_call_and_return_conditional_losses_52388X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
)__inference_flatten_1_layer_call_fn_52393K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
B__inference_re_lu_3_layer_call_and_return_conditional_losses_52238X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
'__inference_re_lu_3_layer_call_fn_52243K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
B__inference_re_lu_4_layer_call_and_return_conditional_losses_52294X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ<
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ<
 v
'__inference_re_lu_4_layer_call_fn_52299K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿ<
B__inference_re_lu_5_layer_call_and_return_conditional_losses_52350X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
'__inference_re_lu_5_layer_call_fn_52355K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈ
G__inference_sequential_2_layer_call_and_return_conditional_losses_49841}"#&'$%(),-*+./230145>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
G__inference_sequential_2_layer_call_and_return_conditional_losses_49892}"#'$&%()-*,+./302145>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
G__inference_sequential_2_layer_call_and_return_conditional_losses_51583v"#&'$%(),-*+./2301457¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
G__inference_sequential_2_layer_call_and_return_conditional_losses_51663v"#'$&%()-*,+./3021457¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
,__inference_sequential_2_layer_call_fn_49989p"#&'$%(),-*+./230145>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_sequential_2_layer_call_fn_50085p"#'$&%()-*,+./302145>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_2_layer_call_fn_51708i"#&'$%(),-*+./2301457¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_2_layer_call_fn_51753i"#'$&%()-*,+./3021457¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ½
G__inference_sequential_3_layer_call_and_return_conditional_losses_50337r6789:;<=?¢<
5¢2
(%
dense_12_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
G__inference_sequential_3_layer_call_and_return_conditional_losses_50368r6789:;<=?¢<
5¢2
(%
dense_12_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_3_layer_call_and_return_conditional_losses_51810j6789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_3_layer_call_and_return_conditional_losses_51846j6789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_3_layer_call_fn_50421e6789:;<=?¢<
5¢2
(%
dense_12_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_3_layer_call_fn_50473e6789:;<=?¢<
5¢2
(%
dense_12_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_3_layer_call_fn_51867]6789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_3_layer_call_fn_51888]6789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÖ
G__inference_sequential_4_layer_call_and_return_conditional_losses_50668"#&'$%(),-*+./2301456789:;<=C¢@
9¢6
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ö
G__inference_sequential_4_layer_call_and_return_conditional_losses_50730"#'$&%()-*,+./3021456789:;<=C¢@
9¢6
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_4_layer_call_and_return_conditional_losses_51221~"#&'$%(),-*+./2301456789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_4_layer_call_and_return_conditional_losses_51333~"#'$&%()-*,+./3021456789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
,__inference_sequential_4_layer_call_fn_50854}"#&'$%(),-*+./2301456789:;<=C¢@
9¢6
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ­
,__inference_sequential_4_layer_call_fn_50977}"#'$&%()-*,+./3021456789:;<=C¢@
9¢6
,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_4_layer_call_fn_51394q"#&'$%(),-*+./2301456789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_4_layer_call_fn_51455q"#'$&%()-*,+./3021456789:;<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÖ
#__inference_signature_wrapper_51040®"#'$&%()-*,+./3021456789:;<=Q¢N
¢ 
GªD
B
sequential_2_input,)
sequential_2_inputÿÿÿÿÿÿÿÿÿ";ª8
6
sequential_3&#
sequential_3ÿÿÿÿÿÿÿÿÿ