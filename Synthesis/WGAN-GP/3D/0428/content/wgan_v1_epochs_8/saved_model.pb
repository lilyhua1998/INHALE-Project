¼µ
Ŗ’
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	č*
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
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:
*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
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

NoOpNoOp
R
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ŌQ
valueŹQBĒQ BĄQ
°
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ć
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
regularization_losses
trainable_variables
	variables
	keras_api
Ź
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
regularization_losses
trainable_variables
 	variables
!	keras_api
 

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
¾
"0
#1
$2
53
64
%5
&6
'7
(8
79
810
)11
*12
+13
,14
915
:16
-17
.18
/19
020
121
222
323
424
­
;metrics
<layer_metrics
regularization_losses
trainable_variables

=layers
>layer_regularization_losses
	variables
?non_trainable_variables
 
^

"kernel
@regularization_losses
Atrainable_variables
B	variables
C	keras_api

Daxis
	#gamma
$beta
5moving_mean
6moving_variance
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

%kernel
&bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api

Maxis
	'gamma
(beta
7moving_mean
8moving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

)kernel
*bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api

Vaxis
	+gamma
,beta
9moving_mean
:moving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
^

-kernel
[regularization_losses
\trainable_variables
]	variables
^	keras_api
 
V
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

"0
#1
$2
53
64
%5
&6
'7
(8
79
810
)11
*12
+13
,14
915
:16
-17
­
_metrics
`layer_metrics
regularization_losses
trainable_variables

alayers
blayer_regularization_losses
	variables
cnon_trainable_variables
^

.kernel
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
R
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
h

/kernel
0bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
R
tregularization_losses
utrainable_variables
v	variables
w	keras_api
R
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
h

1kernel
2bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
l

3kernel
4bias
regularization_losses
trainable_variables
	variables
	keras_api
 
1
.0
/1
02
13
24
35
46
1
.0
/1
02
13
24
35
46
²
metrics
layer_metrics
regularization_losses
trainable_variables
layers
 layer_regularization_losses
 	variables
non_trainable_variables
TR
VARIABLE_VALUEdense_8/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_3/gamma0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_3/beta0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_9/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_9/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_4/gamma0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_4/beta0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_10/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_10/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_5/gamma0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_5/beta1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_11/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_12/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_13/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_13/bias1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_14/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_14/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_15/kernel1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_15/bias1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_3/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_3/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_4/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
*
50
61
72
83
94
:5
 

"0

"0
²
metrics
layer_metrics
@regularization_losses
Atrainable_variables
layers
 layer_regularization_losses
B	variables
non_trainable_variables
 
 

#0
$1

#0
$1
52
63
²
metrics
layer_metrics
Eregularization_losses
Ftrainable_variables
layers
 layer_regularization_losses
G	variables
non_trainable_variables
 

%0
&1

%0
&1
²
metrics
 layer_metrics
Iregularization_losses
Jtrainable_variables
”layers
 ¢layer_regularization_losses
K	variables
£non_trainable_variables
 
 

'0
(1

'0
(1
72
83
²
¤metrics
„layer_metrics
Nregularization_losses
Otrainable_variables
¦layers
 §layer_regularization_losses
P	variables
Ønon_trainable_variables
 

)0
*1

)0
*1
²
©metrics
Ŗlayer_metrics
Rregularization_losses
Strainable_variables
«layers
 ¬layer_regularization_losses
T	variables
­non_trainable_variables
 
 

+0
,1

+0
,1
92
:3
²
®metrics
Ælayer_metrics
Wregularization_losses
Xtrainable_variables
°layers
 ±layer_regularization_losses
Y	variables
²non_trainable_variables
 

-0

-0
²
³metrics
“layer_metrics
[regularization_losses
\trainable_variables
µlayers
 ¶layer_regularization_losses
]	variables
·non_trainable_variables
 
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
*
50
61
72
83
94
:5
 

.0

.0
²
ømetrics
¹layer_metrics
dregularization_losses
etrainable_variables
ŗlayers
 »layer_regularization_losses
f	variables
¼non_trainable_variables
 
 
 
²
½metrics
¾layer_metrics
hregularization_losses
itrainable_variables
ælayers
 Ąlayer_regularization_losses
j	variables
Įnon_trainable_variables
 
 
 
²
Āmetrics
Ćlayer_metrics
lregularization_losses
mtrainable_variables
Älayers
 Ålayer_regularization_losses
n	variables
Ęnon_trainable_variables
 

/0
01

/0
01
²
Ēmetrics
Člayer_metrics
pregularization_losses
qtrainable_variables
Élayers
 Źlayer_regularization_losses
r	variables
Ėnon_trainable_variables
 
 
 
²
Ģmetrics
Ķlayer_metrics
tregularization_losses
utrainable_variables
Īlayers
 Ļlayer_regularization_losses
v	variables
Šnon_trainable_variables
 
 
 
²
Ńmetrics
Ņlayer_metrics
xregularization_losses
ytrainable_variables
Ólayers
 Ōlayer_regularization_losses
z	variables
Õnon_trainable_variables
 

10
21

10
21
²
Ömetrics
×layer_metrics
|regularization_losses
}trainable_variables
Ųlayers
 Łlayer_regularization_losses
~	variables
Śnon_trainable_variables
 
 
 
µ
Ūmetrics
Ülayer_metrics
regularization_losses
trainable_variables
Żlayers
 Žlayer_regularization_losses
	variables
ßnon_trainable_variables
 
 
 
µ
ąmetrics
įlayer_metrics
regularization_losses
trainable_variables
ālayers
 ćlayer_regularization_losses
	variables
änon_trainable_variables
 
 
 
µ
åmetrics
ęlayer_metrics
regularization_losses
trainable_variables
ēlayers
 člayer_regularization_losses
	variables
énon_trainable_variables
 

30
41

30
41
µ
źmetrics
ėlayer_metrics
regularization_losses
trainable_variables
ģlayers
 ķlayer_regularization_losses
	variables
īnon_trainable_variables
 
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

50
61
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
70
81
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
90
:1
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

"serving_default_sequential_2_inputPlaceholder*(
_output_shapes
:’’’’’’’’’č*
dtype0*
shape:’’’’’’’’’č
Ē
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_2_inputdense_8/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_9/kerneldense_9/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_10/kerneldense_10/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_11/kerneldense_12/kerneldense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_55246
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOpConst*&
Tin
2*
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
__inference__traced_save_56641

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kernelbatch_normalization_3/gammabatch_normalization_3/betadense_9/kerneldense_9/biasbatch_normalization_4/gammabatch_normalization_4/betadense_10/kerneldense_10/biasbatch_normalization_5/gammabatch_normalization_5/betadense_11/kerneldense_12/kerneldense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance*%
Tin
2*
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
!__inference__traced_restore_56726Śź
+

G__inference_sequential_2_layer_call_and_return_conditional_losses_54235

inputs
dense_8_54191
batch_normalization_3_54194
batch_normalization_3_54196
batch_normalization_3_54198
batch_normalization_3_54200
dense_9_54203
dense_9_54205
batch_normalization_4_54208
batch_normalization_4_54210
batch_normalization_4_54212
batch_normalization_4_54214
dense_10_54217
dense_10_54219
batch_normalization_5_54222
batch_normalization_5_54224
batch_normalization_5_54226
batch_normalization_5_54228
dense_11_54231
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCallū
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_54191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_539462!
dense_8/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_54194batch_normalization_3_54196batch_normalization_3_54198batch_normalization_3_54200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536102/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_54203dense_9_54205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_540042!
dense_9/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_54208batch_normalization_4_54210batch_normalization_4_54212batch_normalization_4_54214*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537502/
-batch_normalization_4/StatefulPartitionedCallĮ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_54217dense_10_54219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_540662"
 dense_10/StatefulPartitionedCall±
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_54222batch_normalization_5_54224batch_normalization_5_54226batch_normalization_5_54228*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_538902/
-batch_normalization_5/StatefulPartitionedCallÆ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_54231*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_541252"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_54415

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56105

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
:’’’’’’’’’2
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
:’’’’’’’’’2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ńé
Ī
G__inference_sequential_2_layer_call_and_return_conditional_losses_55753

inputs*
&dense_8_matmul_readvariableop_resource/
+batch_normalization_3_assignmovingavg_556461
-batch_normalization_3_assignmovingavg_1_55652?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource/
+batch_normalization_4_assignmovingavg_556851
-batch_normalization_4_assignmovingavg_1_55691?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource/
+batch_normalization_5_assignmovingavg_557241
-batch_normalization_5_assignmovingavg_1_55730?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource
identity¢9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	č*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_8/MatMulp
dense_8/ReluReludense_8/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
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
*batch_normalization_3/moments/StopGradientś
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’21
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
&batch_normalization_3/moments/varianceĀ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeŹ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/55646*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_3/AssignMovingAvg/decayŌ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_55646*
_output_shapes
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpŽ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/55646*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/subÕ
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/55646*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/mul±
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_55646-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/55646*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/55652*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_3/AssignMovingAvg_1/decayŚ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_55652*
_output_shapes
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpč
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/55652*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/subß
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/55652*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/mul½
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_55652/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/55652*
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
%batch_normalization_3/batchnorm/add/yŚ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add„
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtą
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulĢ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%batch_normalization_3/batchnorm/mul_1Ó
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ō
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpŁ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subŻ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%batch_normalization_3/batchnorm/add_1„
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp”
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
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
*batch_normalization_4/moments/StopGradientś
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
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
&batch_normalization_4/moments/varianceĀ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeŹ
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
20loc:@batch_normalization_4/AssignMovingAvg/55685*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayŌ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_55685*
_output_shapes
:
*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpŽ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/55685*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/subÕ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/55685*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/mul±
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_55685-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/55685*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/55691*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayŚ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_55691*
_output_shapes
:
*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpč
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/55691*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/subß
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/55691*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/mul½
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_55691/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/55691*
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
%batch_normalization_4/batchnorm/add/yŚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulĢ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ō
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpŁ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_4/batchnorm/add_1Ø
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
:’’’’’’’’’
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp„
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_10/Relu¶
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesę
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
*batch_normalization_5/moments/StopGradientū
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
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
&batch_normalization_5/moments/varianceĀ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeŹ
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
20loc:@batch_normalization_5/AssignMovingAvg/55724*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_5/AssignMovingAvg/decayŌ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_5_assignmovingavg_55724*
_output_shapes
:
*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpŽ
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/55724*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/subÕ
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/55724*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/mul±
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_5_assignmovingavg_55724-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/55724*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/55730*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_5/AssignMovingAvg_1/decayŚ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_5_assignmovingavg_1_55730*
_output_shapes
:
*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpč
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/55730*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/subß
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/55730*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/mul½
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_5_assignmovingavg_1_55730/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/55730*
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
%batch_normalization_5/batchnorm/add/yŚ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add„
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtą
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulĶ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_5/batchnorm/mul_1Ó
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ō
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpŁ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subŻ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_5/batchnorm/add_1Ø
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_11/Tanh

IdentityIdentitydense_11/Tanh:y:0:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2v
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
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56207

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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
³
Ø
5__inference_batch_normalization_5_layer_call_fn_56322

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
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_538902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
ģ	
Ū
B__inference_dense_9_layer_call_and_return_conditional_losses_56142

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
:’’’’’’’’’
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
:’’’’’’’’’
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų/
Å
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56085

inputs
assignmovingavg_56060
assignmovingavg_1_56066)
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
:’’’’’’’’’2
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56060*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_56060*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56060*
_output_shapes
:2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56060*
_output_shapes
:2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_56060AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56060*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56066*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_56066*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56066*
_output_shapes
:2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56066*
_output_shapes
:2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_56066AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56066*
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
:’’’’’’’’’2
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
:’’’’’’’’’2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
µ
Ø
5__inference_batch_normalization_3_layer_call_fn_56131

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
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
å

G__inference_sequential_4_layer_call_and_return_conditional_losses_54966
sequential_2_input
sequential_2_54913
sequential_2_54915
sequential_2_54917
sequential_2_54919
sequential_2_54921
sequential_2_54923
sequential_2_54925
sequential_2_54927
sequential_2_54929
sequential_2_54931
sequential_2_54933
sequential_2_54935
sequential_2_54937
sequential_2_54939
sequential_2_54941
sequential_2_54943
sequential_2_54945
sequential_2_54947
sequential_3_54950
sequential_3_54952
sequential_3_54954
sequential_3_54956
sequential_3_54958
sequential_3_54960
sequential_3_54962
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputsequential_2_54913sequential_2_54915sequential_2_54917sequential_2_54919sequential_2_54921sequential_2_54923sequential_2_54925sequential_2_54927sequential_2_54929sequential_2_54931sequential_2_54933sequential_2_54935sequential_2_54937sequential_2_54939sequential_2_54941sequential_2_54943sequential_2_54945sequential_2_54947*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_543232&
$sequential_2/StatefulPartitionedCallŗ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_54950sequential_3_54952sequential_3_54954sequential_3_54956sequential_3_54958sequential_3_54960sequential_3_54962*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_547162&
$sequential_3/StatefulPartitionedCallĻ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input
ķ

ų
,__inference_sequential_2_layer_call_fn_55868

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

unknown_16
identity¢StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_542352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_56452

inputs
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs

Ģ
,__inference_sequential_3_layer_call_fn_56015

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_546682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_54390

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾
Ā
G__inference_sequential_4_layer_call_and_return_conditional_losses_55521

inputs7
3sequential_2_dense_8_matmul_readvariableop_resourceH
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
4sequential_2_dense_11_matmul_readvariableop_resource8
4sequential_3_dense_12_matmul_readvariableop_resource8
4sequential_3_dense_13_matmul_readvariableop_resource9
5sequential_3_dense_13_biasadd_readvariableop_resource8
4sequential_3_dense_14_matmul_readvariableop_resource9
5sequential_3_dense_14_biasadd_readvariableop_resource8
4sequential_3_dense_15_matmul_readvariableop_resource9
5sequential_3_dense_15_biasadd_readvariableop_resource
identity¢;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢,sequential_2/dense_10/BiasAdd/ReadVariableOp¢+sequential_2/dense_10/MatMul/ReadVariableOp¢+sequential_2/dense_11/MatMul/ReadVariableOp¢*sequential_2/dense_8/MatMul/ReadVariableOp¢+sequential_2/dense_9/BiasAdd/ReadVariableOp¢*sequential_2/dense_9/MatMul/ReadVariableOp¢+sequential_3/dense_12/MatMul/ReadVariableOp¢,sequential_3/dense_13/BiasAdd/ReadVariableOp¢+sequential_3/dense_13/MatMul/ReadVariableOp¢,sequential_3/dense_14/BiasAdd/ReadVariableOp¢+sequential_3/dense_14/MatMul/ReadVariableOp¢,sequential_3/dense_15/BiasAdd/ReadVariableOp¢+sequential_3/dense_15/MatMul/ReadVariableOpĶ
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	č*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp²
sequential_2/dense_8/MatMulMatMulinputs2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_8/MatMul
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_8/Reluū
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
0sequential_2/batch_normalization_3/batchnorm/addĢ
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
:’’’’’’’’’24
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
:’’’’’’’’’24
2sequential_2/batch_normalization_3/batchnorm/add_1Ģ
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOpā
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/MatMulĖ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpÕ
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/BiasAdd
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/Reluū
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
0sequential_2/batch_normalization_4/batchnorm/addĢ
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
:’’’’’’’’’
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_4/batchnorm/add_1Ļ
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
:’’’’’’’’’
2
sequential_2/dense_10/MatMulĪ
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOpŁ
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_10/BiasAdd
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_10/Reluū
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
0sequential_2/batch_normalization_5/batchnorm/addĢ
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
:’’’’’’’’’
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_5/batchnorm/add_1Ļ
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOpå
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_11/MatMul
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_11/TanhĻ
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOpĶ
sequential_3/dense_12/MatMulMatMulsequential_2/dense_11/Tanh:y:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_12/MatMul
sequential_3/re_lu_3/ReluRelu&sequential_3/dense_12/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/re_lu_3/Relu©
sequential_3/dropout_3/IdentityIdentity'sequential_3/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_3/dropout_3/IdentityĻ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp×
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_3/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/dense_13/MatMulĪ
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOpŁ
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/dense_13/BiasAdd
sequential_3/re_lu_4/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/re_lu_4/Relu©
sequential_3/dropout_4/IdentityIdentity'sequential_3/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’<2!
sequential_3/dropout_4/IdentityĻ
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp×
sequential_3/dense_14/MatMulMatMul(sequential_3/dropout_4/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_14/MatMulĪ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOpŁ
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_14/BiasAdd
sequential_3/re_lu_5/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/re_lu_5/Relu©
sequential_3/dropout_5/IdentityIdentity'sequential_3/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2!
sequential_3/dropout_5/Identity
sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
sequential_3/flatten_1/ConstĪ
sequential_3/flatten_1/ReshapeReshape(sequential_3/dropout_5/Identity:output:0%sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2 
sequential_3/flatten_1/ReshapeĻ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOpÖ
sequential_3/dense_15/MatMulMatMul'sequential_3/flatten_1/Reshape:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_15/MatMulĪ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOpŁ
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_15/BiasAddŅ
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2z
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
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ŗ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_54572

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ģ
,__inference_sequential_3_layer_call_fn_56034

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_547162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_56386

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
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų/
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_53890

inputs
assignmovingavg_53865
assignmovingavg_1_53871)
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
:’’’’’’’’’
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53865*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53865*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53865*
_output_shapes
:
2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53865*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53865AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53865*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53871*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53871*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53871*
_output_shapes
:
2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53871*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53871AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53871*
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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs

ō
,__inference_sequential_4_layer_call_fn_55078
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

unknown_23
identity¢StatefulPartitionedCall¶
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*5
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_550252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input

E
)__inference_dropout_3_layer_call_fn_56401

inputs
identityĀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_56369

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
/
ņ
G__inference_sequential_3_layer_call_and_return_conditional_losses_54607
dense_12_input
dense_12_54382
dense_13_54449
dense_13_54451
dense_14_54518
dense_14_54520
dense_15_54601
dense_15_54603
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_54382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_543732"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_543902
re_lu_3/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544102#
!dropout_3/StatefulPartitionedCallµ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_54449dense_13_54451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_544382"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_544592
re_lu_4/PartitionedCall¬
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544792#
!dropout_4/StatefulPartitionedCallµ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_14_54518dense_14_54520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_545072"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_545282
re_lu_5/PartitionedCall¬
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545482#
!dropout_5/StatefulPartitionedCallś
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_545722
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_54601dense_15_54603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_545902"
 dense_15/StatefulPartitionedCallõ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namedense_12_input
°
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_56481

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_53923

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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
ģ	
Ū
B__inference_dense_9_layer_call_and_return_conditional_losses_54004

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
:’’’’’’’’’
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
:’’’’’’’’’
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_54528

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

E
)__inference_flatten_1_layer_call_fn_56524

inputs
identityĀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_545722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

E
)__inference_dropout_5_layer_call_fn_56513

inputs
identityĀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ŗ
n
(__inference_dense_12_layer_call_fn_56364

inputs
unknown
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_543732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ō
,__inference_sequential_3_layer_call_fn_54733
dense_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_547162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namedense_12_input

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_54410

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
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
*
ž
G__inference_sequential_3_layer_call_and_return_conditional_losses_54716

inputs
dense_12_54690
dense_13_54695
dense_13_54697
dense_14_54702
dense_14_54704
dense_15_54710
dense_15_54712
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall’
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_54690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_543732"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_543902
re_lu_3/PartitionedCallš
dropout_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544152
dropout_3/PartitionedCall­
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_54695dense_13_54697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_544382"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_544592
re_lu_4/PartitionedCallš
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544842
dropout_4/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_14_54702dense_14_54704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_545072"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_545282
re_lu_5/PartitionedCallš
dropout_5/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545532
dropout_5/PartitionedCallņ
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_545722
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_54710dense_15_54712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_545902"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ü
C__inference_dense_13_layer_call_and_return_conditional_losses_54438

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
:’’’’’’’’’<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
µ
Ø
5__inference_batch_normalization_4_layer_call_fn_56233

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
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
Ś
}
(__inference_dense_14_layer_call_fn_56476

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
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_545072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
Ž9
Ė
__inference__traced_save_56641
file_prefix-
)savev2_dense_8_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop
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
ShardedFilenameÓ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*å	
valueŪ	BŲ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¼
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesĻ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop*savev2_dense_11_kernel_read_readvariableop*savev2_dense_12_kernel_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
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

identity_1Identity_1:output:0*Š
_input_shapes¾
»: :	č:::
:
:
:
:

:
:
:
:
::<:<:<::::::
:
:
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	č: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:$ 

_output_shapes

:

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
:$ 

_output_shapes

:
:$ 

_output_shapes

::$ 

_output_shapes

:<: 

_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:

_output_shapes
: 
ó.
ź
G__inference_sequential_3_layer_call_and_return_conditional_losses_54668

inputs
dense_12_54642
dense_13_54647
dense_13_54649
dense_14_54654
dense_14_54656
dense_15_54662
dense_15_54664
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall’
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_54642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_543732"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_543902
re_lu_3/PartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544102#
!dropout_3/StatefulPartitionedCallµ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_54647dense_13_54649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_544382"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_544592
re_lu_4/PartitionedCall¬
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544792#
!dropout_4/StatefulPartitionedCallµ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_14_54654dense_14_54656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_545072"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_545282
re_lu_5/PartitionedCall¬
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545482#
!dropout_5/StatefulPartitionedCallś
flatten_1/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_545722
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_54662dense_15_54664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_545902"
 dense_15/StatefulPartitionedCallõ
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

C
'__inference_re_lu_3_layer_call_fn_56374

inputs
identityĄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_543902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų/
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56187

inputs
assignmovingavg_56162
assignmovingavg_1_56168)
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
:’’’’’’’’’
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56162*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_56162*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56162*
_output_shapes
:
2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56162*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_56162AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56162*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56168*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_56168*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56168*
_output_shapes
:
2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56168*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_56168AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56168*
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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_56498

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
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
óģ
Ī
G__inference_sequential_4_layer_call_and_return_conditional_losses_55418

inputs7
3sequential_2_dense_8_matmul_readvariableop_resource<
8sequential_2_batch_normalization_3_assignmovingavg_55261>
:sequential_2_batch_normalization_3_assignmovingavg_1_55267L
Hsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_3_batchnorm_readvariableop_resource7
3sequential_2_dense_9_matmul_readvariableop_resource8
4sequential_2_dense_9_biasadd_readvariableop_resource<
8sequential_2_batch_normalization_4_assignmovingavg_55300>
:sequential_2_batch_normalization_4_assignmovingavg_1_55306L
Hsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_4_batchnorm_readvariableop_resource8
4sequential_2_dense_10_matmul_readvariableop_resource9
5sequential_2_dense_10_biasadd_readvariableop_resource<
8sequential_2_batch_normalization_5_assignmovingavg_55339>
:sequential_2_batch_normalization_5_assignmovingavg_1_55345L
Hsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceH
Dsequential_2_batch_normalization_5_batchnorm_readvariableop_resource8
4sequential_2_dense_11_matmul_readvariableop_resource8
4sequential_3_dense_12_matmul_readvariableop_resource8
4sequential_3_dense_13_matmul_readvariableop_resource9
5sequential_3_dense_13_biasadd_readvariableop_resource8
4sequential_3_dense_14_matmul_readvariableop_resource9
5sequential_3_dense_14_biasadd_readvariableop_resource8
4sequential_3_dense_15_matmul_readvariableop_resource9
5sequential_3_dense_15_biasadd_readvariableop_resource
identity¢Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp¢Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢,sequential_2/dense_10/BiasAdd/ReadVariableOp¢+sequential_2/dense_10/MatMul/ReadVariableOp¢+sequential_2/dense_11/MatMul/ReadVariableOp¢*sequential_2/dense_8/MatMul/ReadVariableOp¢+sequential_2/dense_9/BiasAdd/ReadVariableOp¢*sequential_2/dense_9/MatMul/ReadVariableOp¢+sequential_3/dense_12/MatMul/ReadVariableOp¢,sequential_3/dense_13/BiasAdd/ReadVariableOp¢+sequential_3/dense_13/MatMul/ReadVariableOp¢,sequential_3/dense_14/BiasAdd/ReadVariableOp¢+sequential_3/dense_14/MatMul/ReadVariableOp¢,sequential_3/dense_15/BiasAdd/ReadVariableOp¢+sequential_3/dense_15/MatMul/ReadVariableOpĶ
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	č*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp²
sequential_2/dense_8/MatMulMatMulinputs2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_8/MatMul
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_8/ReluŠ
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
:’’’’’’’’’2>
<sequential_2/batch_normalization_3/moments/SquaredDifferenceŲ
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
2sequential_2/batch_normalization_3/moments/Squeezeń
4sequential_2/batch_normalization_3/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 26
4sequential_2/batch_normalization_3/moments/Squeeze_1“
8sequential_2/batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/55261*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_3/AssignMovingAvg/decayū
Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_3_assignmovingavg_55261*
_output_shapes
:*
dtype02C
Asequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_3/AssignMovingAvg/subSubIsequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/55261*
_output_shapes
:28
6sequential_2/batch_normalization_3/AssignMovingAvg/sub
6sequential_2/batch_normalization_3/AssignMovingAvg/mulMul:sequential_2/batch_normalization_3/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/55261*
_output_shapes
:28
6sequential_2/batch_normalization_3/AssignMovingAvg/mul’
Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_3_assignmovingavg_55261:sequential_2/batch_normalization_3/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_3/AssignMovingAvg/55261*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpŗ
:sequential_2/batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/55267*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_3/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_3_assignmovingavg_1_55267*
_output_shapes
:*
dtype02E
Csequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_3/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/55267*
_output_shapes
:2:
8sequential_2/batch_normalization_3/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_3/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_3/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/55267*
_output_shapes
:2:
8sequential_2/batch_normalization_3/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_3_assignmovingavg_1_55267<sequential_2/batch_normalization_3/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_3/AssignMovingAvg_1/55267*
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
0sequential_2/batch_normalization_3/batchnorm/addĢ
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
:’’’’’’’’’24
2sequential_2/batch_normalization_3/batchnorm/mul_1
2sequential_2/batch_normalization_3/batchnorm/mul_2Mul;sequential_2/batch_normalization_3/moments/Squeeze:output:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/mul_2ū
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
:’’’’’’’’’24
2sequential_2/batch_normalization_3/batchnorm/add_1Ģ
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOpā
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/MatMulĖ
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpÕ
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/BiasAdd
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_9/ReluŠ
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
:’’’’’’’’’
2>
<sequential_2/batch_normalization_4/moments/SquaredDifferenceŲ
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
2sequential_2/batch_normalization_4/moments/Squeezeń
4sequential_2/batch_normalization_4/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 26
4sequential_2/batch_normalization_4/moments/Squeeze_1“
8sequential_2/batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/55300*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_4/AssignMovingAvg/decayū
Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_4_assignmovingavg_55300*
_output_shapes
:
*
dtype02C
Asequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_4/AssignMovingAvg/subSubIsequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/55300*
_output_shapes
:
28
6sequential_2/batch_normalization_4/AssignMovingAvg/sub
6sequential_2/batch_normalization_4/AssignMovingAvg/mulMul:sequential_2/batch_normalization_4/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/55300*
_output_shapes
:
28
6sequential_2/batch_normalization_4/AssignMovingAvg/mul’
Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_4_assignmovingavg_55300:sequential_2/batch_normalization_4/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_4/AssignMovingAvg/55300*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpŗ
:sequential_2/batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/55306*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_4/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_4_assignmovingavg_1_55306*
_output_shapes
:
*
dtype02E
Csequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_4/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/55306*
_output_shapes
:
2:
8sequential_2/batch_normalization_4/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_4/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_4/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/55306*
_output_shapes
:
2:
8sequential_2/batch_normalization_4/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_4_assignmovingavg_1_55306<sequential_2/batch_normalization_4/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_4/AssignMovingAvg_1/55306*
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
0sequential_2/batch_normalization_4/batchnorm/addĢ
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_4/batchnorm/mul_1
2sequential_2/batch_normalization_4/batchnorm/mul_2Mul;sequential_2/batch_normalization_4/moments/Squeeze:output:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/mul_2ū
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_4/batchnorm/add_1Ļ
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
:’’’’’’’’’
2
sequential_2/dense_10/MatMulĪ
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOpŁ
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_10/BiasAdd
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential_2/dense_10/ReluŠ
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
7sequential_2/batch_normalization_5/moments/StopGradientÆ
<sequential_2/batch_normalization_5/moments/SquaredDifferenceSquaredDifference(sequential_2/dense_10/Relu:activations:0@sequential_2/batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2>
<sequential_2/batch_normalization_5/moments/SquaredDifferenceŲ
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
2sequential_2/batch_normalization_5/moments/Squeezeń
4sequential_2/batch_normalization_5/moments/Squeeze_1Squeeze<sequential_2/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 26
4sequential_2/batch_normalization_5/moments/Squeeze_1“
8sequential_2/batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/55339*
_output_shapes
: *
dtype0*
valueB
 *
×#<2:
8sequential_2/batch_normalization_5/AssignMovingAvg/decayū
Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp8sequential_2_batch_normalization_5_assignmovingavg_55339*
_output_shapes
:
*
dtype02C
Asequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp
6sequential_2/batch_normalization_5/AssignMovingAvg/subSubIsequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0;sequential_2/batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/55339*
_output_shapes
:
28
6sequential_2/batch_normalization_5/AssignMovingAvg/sub
6sequential_2/batch_normalization_5/AssignMovingAvg/mulMul:sequential_2/batch_normalization_5/AssignMovingAvg/sub:z:0Asequential_2/batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/55339*
_output_shapes
:
28
6sequential_2/batch_normalization_5/AssignMovingAvg/mul’
Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp8sequential_2_batch_normalization_5_assignmovingavg_55339:sequential_2/batch_normalization_5/AssignMovingAvg/mul:z:0B^sequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*K
_classA
?=loc:@sequential_2/batch_normalization_5/AssignMovingAvg/55339*
_output_shapes
 *
dtype02H
Fsequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpŗ
:sequential_2/batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/55345*
_output_shapes
: *
dtype0*
valueB
 *
×#<2<
:sequential_2/batch_normalization_5/AssignMovingAvg_1/decay
Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_5_assignmovingavg_1_55345*
_output_shapes
:
*
dtype02E
Csequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp©
8sequential_2/batch_normalization_5/AssignMovingAvg_1/subSubKsequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:0=sequential_2/batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/55345*
_output_shapes
:
2:
8sequential_2/batch_normalization_5/AssignMovingAvg_1/sub 
8sequential_2/batch_normalization_5/AssignMovingAvg_1/mulMul<sequential_2/batch_normalization_5/AssignMovingAvg_1/sub:z:0Csequential_2/batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/55345*
_output_shapes
:
2:
8sequential_2/batch_normalization_5/AssignMovingAvg_1/mul
Hsequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp:sequential_2_batch_normalization_5_assignmovingavg_1_55345<sequential_2/batch_normalization_5/AssignMovingAvg_1/mul:z:0D^sequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*M
_classC
A?loc:@sequential_2/batch_normalization_5/AssignMovingAvg_1/55345*
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
0sequential_2/batch_normalization_5/batchnorm/addĢ
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_5/batchnorm/mul_1
2sequential_2/batch_normalization_5/batchnorm/mul_2Mul;sequential_2/batch_normalization_5/moments/Squeeze:output:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/mul_2ū
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
:’’’’’’’’’
24
2sequential_2/batch_normalization_5/batchnorm/add_1Ļ
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOpå
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_11/MatMul
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_11/TanhĻ
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOpĶ
sequential_3/dense_12/MatMulMatMulsequential_2/dense_11/Tanh:y:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_12/MatMul
sequential_3/re_lu_3/ReluRelu&sequential_3/dense_12/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/re_lu_3/Relu
$sequential_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_3/dropout/ConstŁ
"sequential_3/dropout_3/dropout/MulMul'sequential_3/re_lu_3/Relu:activations:0-sequential_3/dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"sequential_3/dropout_3/dropout/Mul£
$sequential_3/dropout_3/dropout/ShapeShape'sequential_3/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_3/dropout/Shapeł
;sequential_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02=
;sequential_3/dropout_3/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2/
-sequential_3/dropout_3/dropout/GreaterEqual/y
+sequential_3/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+sequential_3/dropout_3/dropout/GreaterEqualÄ
#sequential_3/dropout_3/dropout/CastCast/sequential_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2%
#sequential_3/dropout_3/dropout/CastÖ
$sequential_3/dropout_3/dropout/Mul_1Mul&sequential_3/dropout_3/dropout/Mul:z:0'sequential_3/dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$sequential_3/dropout_3/dropout/Mul_1Ļ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp×
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_3/dropout/Mul_1:z:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/dense_13/MatMulĪ
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOpŁ
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/dense_13/BiasAdd
sequential_3/re_lu_4/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential_3/re_lu_4/Relu
$sequential_3/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_4/dropout/ConstŁ
"sequential_3/dropout_4/dropout/MulMul'sequential_3/re_lu_4/Relu:activations:0-sequential_3/dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2$
"sequential_3/dropout_4/dropout/Mul£
$sequential_3/dropout_4/dropout/ShapeShape'sequential_3/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_4/dropout/Shapeł
;sequential_3/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’<*
dtype02=
;sequential_3/dropout_4/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2/
-sequential_3/dropout_4/dropout/GreaterEqual/y
+sequential_3/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2-
+sequential_3/dropout_4/dropout/GreaterEqualÄ
#sequential_3/dropout_4/dropout/CastCast/sequential_3/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’<2%
#sequential_3/dropout_4/dropout/CastÖ
$sequential_3/dropout_4/dropout/Mul_1Mul&sequential_3/dropout_4/dropout/Mul:z:0'sequential_3/dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’<2&
$sequential_3/dropout_4/dropout/Mul_1Ļ
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp×
sequential_3/dense_14/MatMulMatMul(sequential_3/dropout_4/dropout/Mul_1:z:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_14/MatMulĪ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOpŁ
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_14/BiasAdd
sequential_3/re_lu_5/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/re_lu_5/Relu
$sequential_3/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$sequential_3/dropout_5/dropout/ConstŁ
"sequential_3/dropout_5/dropout/MulMul'sequential_3/re_lu_5/Relu:activations:0-sequential_3/dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"sequential_3/dropout_5/dropout/Mul£
$sequential_3/dropout_5/dropout/ShapeShape'sequential_3/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_3/dropout_5/dropout/Shapeł
;sequential_3/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_3/dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02=
;sequential_3/dropout_5/dropout/random_uniform/RandomUniform£
-sequential_3/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2/
-sequential_3/dropout_5/dropout/GreaterEqual/y
+sequential_3/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_3/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_3/dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+sequential_3/dropout_5/dropout/GreaterEqualÄ
#sequential_3/dropout_5/dropout/CastCast/sequential_3/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2%
#sequential_3/dropout_5/dropout/CastÖ
$sequential_3/dropout_5/dropout/Mul_1Mul&sequential_3/dropout_5/dropout/Mul:z:0'sequential_3/dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2&
$sequential_3/dropout_5/dropout/Mul_1
sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
sequential_3/flatten_1/ConstĪ
sequential_3/flatten_1/ReshapeReshape(sequential_3/dropout_5/dropout/Mul_1:z:0%sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2 
sequential_3/flatten_1/ReshapeĻ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOpÖ
sequential_3/dense_15/MatMulMatMul'sequential_3/flatten_1/Reshape:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_15/MatMulĪ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOpŁ
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_3/dense_15/BiasAdd¬
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0G^sequential_2/batch_normalization_3/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_3/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_4/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_4/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpG^sequential_2/batch_normalization_5/AssignMovingAvg/AssignSubVariableOpB^sequential_2/batch_normalization_5/AssignMovingAvg/ReadVariableOpI^sequential_2/batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpD^sequential_2/batch_normalization_5/AssignMovingAvg_1/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2
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
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
C

G__inference_sequential_3_layer_call_and_return_conditional_losses_55963

inputs+
'dense_12_matmul_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOpØ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_12/MatMulq
re_lu_3/ReluReludense_12/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
re_lu_3/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/dropout/Const„
dropout_3/dropout/MulMulre_lu_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeŅ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2"
 dropout_3/dropout/GreaterEqual/yę
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout_3/dropout/Cast¢
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_3/dropout/Mul_1Ø
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_13/BiasAdd/ReadVariableOp„
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense_13/BiasAddq
re_lu_4/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
re_lu_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/dropout/Const„
dropout_4/dropout/MulMulre_lu_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeŅ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’<*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2"
 dropout_4/dropout/GreaterEqual/yę
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2 
dropout_4/dropout/GreaterEqual
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’<2
dropout_4/dropout/Cast¢
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout_4/dropout/Mul_1Ø
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldropout_4/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp„
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/BiasAddq
re_lu_5/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
re_lu_5/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/dropout/Const„
dropout_5/dropout/MulMulre_lu_5/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeŅ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2"
 dropout_5/dropout/GreaterEqual/yę
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout_5/dropout/Cast¢
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_5/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
flatten_1/ReshapeØ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp¢
dense_15/MatMulMatMulflatten_1/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp„
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_15/BiasAdd×
IdentityIdentitydense_15/BiasAdd:output:0^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³
Ø
5__inference_batch_normalization_4_layer_call_fn_56220

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
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
ā
č
,__inference_sequential_4_layer_call_fn_55631

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

unknown_23
identity¢StatefulPartitionedCall°
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_551362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
š

C__inference_dense_12_layer_call_and_return_conditional_losses_56357

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
+

G__inference_sequential_2_layer_call_and_return_conditional_losses_54323

inputs
dense_8_54279
batch_normalization_3_54282
batch_normalization_3_54284
batch_normalization_3_54286
batch_normalization_3_54288
dense_9_54291
dense_9_54293
batch_normalization_4_54296
batch_normalization_4_54298
batch_normalization_4_54300
batch_normalization_4_54302
dense_10_54305
dense_10_54307
batch_normalization_5_54310
batch_normalization_5_54312
batch_normalization_5_54314
batch_normalization_5_54316
dense_11_54319
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCallū
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_54279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_539462!
dense_8/StatefulPartitionedCall²
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_54282batch_normalization_3_54284batch_normalization_3_54286batch_normalization_3_54288*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536432/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_54291dense_9_54293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_540042!
dense_9/StatefulPartitionedCall²
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_54296batch_normalization_4_54298batch_normalization_4_54300batch_normalization_4_54302*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537832/
-batch_normalization_4/StatefulPartitionedCallĮ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_54305dense_10_54307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_540662"
 dense_10/StatefulPartitionedCall³
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_54310batch_normalization_5_54312batch_normalization_5_54314batch_normalization_5_54316*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_539232/
-batch_normalization_5/StatefulPartitionedCallÆ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_54319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_541252"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ś
}
(__inference_dense_10_layer_call_fn_56253

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
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_540662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
	
Ü
C__inference_dense_15_layer_call_and_return_conditional_losses_54590

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

’
,__inference_sequential_2_layer_call_fn_54274
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

unknown_16
identity¢StatefulPartitionedCallĻ
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_542352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’č
'
_user_specified_namedense_8_input
Ų
|
'__inference_dense_9_layer_call_fn_56151

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_540042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_53643

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
:’’’’’’’’’2
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
:’’’’’’’’’2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

’
,__inference_sequential_2_layer_call_fn_54362
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

unknown_16
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_543232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’č
'
_user_specified_namedense_8_input
µ
Ø
5__inference_batch_normalization_5_layer_call_fn_56335

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
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_539232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56309

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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
Ö
ė
#__inference_signature_wrapper_55246
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

unknown_23
identity¢StatefulPartitionedCall
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_535142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input
ß

G__inference_sequential_4_layer_call_and_return_conditional_losses_54910
sequential_2_input
sequential_2_54819
sequential_2_54821
sequential_2_54823
sequential_2_54825
sequential_2_54827
sequential_2_54829
sequential_2_54831
sequential_2_54833
sequential_2_54835
sequential_2_54837
sequential_2_54839
sequential_2_54841
sequential_2_54843
sequential_2_54845
sequential_2_54847
sequential_2_54849
sequential_2_54851
sequential_2_54853
sequential_3_54894
sequential_3_54896
sequential_3_54898
sequential_3_54900
sequential_3_54902
sequential_3_54904
sequential_3_54906
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallsequential_2_inputsequential_2_54819sequential_2_54821sequential_2_54823sequential_2_54825sequential_2_54827sequential_2_54829sequential_2_54831sequential_2_54833sequential_2_54835sequential_2_54837sequential_2_54839sequential_2_54841sequential_2_54843sequential_2_54845sequential_2_54847sequential_2_54849sequential_2_54851sequential_2_54853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_542352&
$sequential_2/StatefulPartitionedCallŗ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_54894sequential_3_54896sequential_3_54898sequential_3_54900sequential_3_54902sequential_3_54904sequential_3_54906*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_546682&
$sequential_3/StatefulPartitionedCallĻ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_54479

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
:’’’’’’’’’<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs

ō
,__inference_sequential_4_layer_call_fn_55189
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

unknown_23
identity¢StatefulPartitionedCall¼
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_551362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input
øs

G__inference_sequential_2_layer_call_and_return_conditional_losses_55827

inputs*
&dense_8_matmul_readvariableop_resource;
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
'dense_11_matmul_readvariableop_resource
identity¢.batch_normalization_3/batchnorm/ReadVariableOp¢0batch_normalization_3/batchnorm/ReadVariableOp_1¢0batch_normalization_3/batchnorm/ReadVariableOp_2¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	č*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_8/MatMulp
dense_8/ReluReludense_8/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_8/ReluŌ
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
%batch_normalization_3/batchnorm/add/yą
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add„
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtą
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulĢ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%batch_normalization_3/batchnorm/mul_1Ś
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ż
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ś
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2Ū
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subŻ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%batch_normalization_3/batchnorm/add_1„
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp”
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_9/ReluŌ
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
%batch_normalization_4/batchnorm/add/yą
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulĢ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_4/batchnorm/mul_1Ś
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ż
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ś
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Ū
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_4/batchnorm/add_1Ø
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
:’’’’’’’’’
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp„
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_10/ReluŌ
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
%batch_normalization_5/batchnorm/add/yą
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add„
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtą
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulĶ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_5/batchnorm/mul_1Ś
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ż
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ś
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Ū
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subŻ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2'
%batch_normalization_5/batchnorm/add_1Ø
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_11/Tanh
IdentityIdentitydense_11/Tanh:y:0/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2`
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
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_56447

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
°
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_54459

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’<2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
ä
±
 __inference__wrapped_model_53514
sequential_2_inputD
@sequential_4_sequential_2_dense_8_matmul_readvariableop_resourceU
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
Asequential_4_sequential_2_dense_11_matmul_readvariableop_resourceE
Asequential_4_sequential_3_dense_12_matmul_readvariableop_resourceE
Asequential_4_sequential_3_dense_13_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_13_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_14_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_14_biasadd_readvariableop_resourceE
Asequential_4_sequential_3_dense_15_matmul_readvariableop_resourceF
Bsequential_4_sequential_3_dense_15_biasadd_readvariableop_resource
identity¢Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢Lsequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢9sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp¢8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp¢8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp¢7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp¢8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp¢7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp¢8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp¢9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp¢8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOpō
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp@sequential_4_sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	č*
dtype029
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOpå
(sequential_4/sequential_2/dense_8/MatMulMatMulsequential_2_input?sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2*
(sequential_4/sequential_2/dense_8/MatMul¾
&sequential_4/sequential_2/dense_8/ReluRelu2sequential_4/sequential_2/dense_8/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&sequential_4/sequential_2/dense_8/Relu¢
Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpĒ
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add/yČ
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
=sequential_4/sequential_2/batch_normalization_3/batchnorm/mul“
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1Mul4sequential_4/sequential_2/dense_8/Relu:activations:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1Ø
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpSsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1Å
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2MulRsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?sequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2Ø
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2Ć
=sequential_4/sequential_2/batch_normalization_3/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=sequential_4/sequential_2/batch_normalization_3/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_3/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_3/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2A
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
:’’’’’’’’’
2*
(sequential_4/sequential_2/dense_9/MatMulņ
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02:
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp
)sequential_4/sequential_2/dense_9/BiasAddBiasAdd2sequential_4/sequential_2/dense_9/MatMul:product:0@sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2+
)sequential_4/sequential_2/dense_9/BiasAdd¾
&sequential_4/sequential_2/dense_9/ReluRelu2sequential_4/sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&sequential_4/sequential_2/dense_9/Relu¢
Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02J
Hsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpĒ
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add/yČ
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
=sequential_4/sequential_2/batch_normalization_4/batchnorm/mul“
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1Mul4sequential_4/sequential_2/dense_9/Relu:activations:0Asequential_4/sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2A
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1Ø
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
?sequential_4/sequential_2/batch_normalization_4/batchnorm/mul_2Ø
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2Ć
=sequential_4/sequential_2/batch_normalization_4/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_4/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_4/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_4/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
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
:’’’’’’’’’
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
:’’’’’’’’’
2,
*sequential_4/sequential_2/dense_10/BiasAddĮ
'sequential_4/sequential_2/dense_10/ReluRelu3sequential_4/sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2)
'sequential_4/sequential_2/dense_10/Relu¢
Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpQsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02J
Hsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpĒ
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add/yČ
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
:’’’’’’’’’
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_1Ø
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
?sequential_4/sequential_2/batch_normalization_5/batchnorm/mul_2Ø
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpSsequential_4_sequential_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02L
Jsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2Ć
=sequential_4/sequential_2/batch_normalization_5/batchnorm/subSubRsequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:0Csequential_4/sequential_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2?
=sequential_4/sequential_2/batch_normalization_5/batchnorm/subÅ
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add_1AddV2Csequential_4/sequential_2/batch_normalization_5/batchnorm/mul_1:z:0Asequential_4/sequential_2/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2A
?sequential_4/sequential_2/batch_normalization_5/batchnorm/add_1ö
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02:
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp
)sequential_4/sequential_2/dense_11/MatMulMatMulCsequential_4/sequential_2/batch_normalization_5/batchnorm/add_1:z:0@sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2+
)sequential_4/sequential_2/dense_11/MatMulĮ
'sequential_4/sequential_2/dense_11/TanhTanh3sequential_4/sequential_2/dense_11/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2)
'sequential_4/sequential_2/dense_11/Tanhö
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_12/MatMulMatMul+sequential_4/sequential_2/dense_11/Tanh:y:0@sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2+
)sequential_4/sequential_3/dense_12/MatMulæ
&sequential_4/sequential_3/re_lu_3/ReluRelu3sequential_4/sequential_3/dense_12/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&sequential_4/sequential_3/re_lu_3/ReluŠ
,sequential_4/sequential_3/dropout_3/IdentityIdentity4sequential_4/sequential_3/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,sequential_4/sequential_3/dropout_3/Identityö
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02:
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_13/MatMulMatMul5sequential_4/sequential_3/dropout_3/Identity:output:0@sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2+
)sequential_4/sequential_3/dense_13/MatMulõ
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02;
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_13/BiasAddBiasAdd3sequential_4/sequential_3/dense_13/MatMul:product:0Asequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2,
*sequential_4/sequential_3/dense_13/BiasAddæ
&sequential_4/sequential_3/re_lu_4/ReluRelu3sequential_4/sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2(
&sequential_4/sequential_3/re_lu_4/ReluŠ
,sequential_4/sequential_3/dropout_4/IdentityIdentity4sequential_4/sequential_3/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’<2.
,sequential_4/sequential_3/dropout_4/Identityö
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02:
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_14/MatMulMatMul5sequential_4/sequential_3/dropout_4/Identity:output:0@sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2+
)sequential_4/sequential_3/dense_14/MatMulõ
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_14/BiasAddBiasAdd3sequential_4/sequential_3/dense_14/MatMul:product:0Asequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2,
*sequential_4/sequential_3/dense_14/BiasAddæ
&sequential_4/sequential_3/re_lu_5/ReluRelu3sequential_4/sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&sequential_4/sequential_3/re_lu_5/ReluŠ
,sequential_4/sequential_3/dropout_5/IdentityIdentity4sequential_4/sequential_3/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,sequential_4/sequential_3/dropout_5/Identity§
)sequential_4/sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2+
)sequential_4/sequential_3/flatten_1/Const
+sequential_4/sequential_3/flatten_1/ReshapeReshape5sequential_4/sequential_3/dropout_5/Identity:output:02sequential_4/sequential_3/flatten_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+sequential_4/sequential_3/flatten_1/Reshapeö
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOpAsequential_4_sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02:
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp
)sequential_4/sequential_3/dense_15/MatMulMatMul4sequential_4/sequential_3/flatten_1/Reshape:output:0@sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2+
)sequential_4/sequential_3/dense_15/MatMulõ
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOpBsequential_4_sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp
*sequential_4/sequential_3/dense_15/BiasAddBiasAdd3sequential_4/sequential_3/dense_15/MatMul:product:0Asequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2,
*sequential_4/sequential_3/dense_15/BiasAdd¤
IdentityIdentity3sequential_4/sequential_3/dense_15/BiasAdd:output:0I^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpI^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOpK^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1K^sequential_4/sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2M^sequential_4/sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:^sequential_4/sequential_2/dense_10/BiasAdd/ReadVariableOp9^sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp9^sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp8^sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp9^sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp8^sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp9^sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp:^sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp9^sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2
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
8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp8sequential_4/sequential_2/dense_10/MatMul/ReadVariableOp2t
8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp8sequential_4/sequential_2/dense_11/MatMul/ReadVariableOp2r
7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_8/MatMul/ReadVariableOp2t
8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp8sequential_4/sequential_2/dense_9/BiasAdd/ReadVariableOp2r
7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp7sequential_4/sequential_2/dense_9/MatMul/ReadVariableOp2t
8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_12/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_13/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_13/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_14/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_14/MatMul/ReadVariableOp2v
9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp9sequential_4/sequential_3/dense_15/BiasAdd/ReadVariableOp2t
8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp8sequential_4/sequential_3/dense_15/MatMul/ReadVariableOp:\ X
(
_output_shapes
:’’’’’’’’’č
,
_user_specified_namesequential_2_input
³*

G__inference_sequential_3_layer_call_and_return_conditional_losses_54636
dense_12_input
dense_12_54610
dense_13_54615
dense_13_54617
dense_14_54622
dense_14_54624
dense_15_54630
dense_15_54632
identity¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_54610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_543732"
 dense_12/StatefulPartitionedCalló
re_lu_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_543902
re_lu_3/PartitionedCallš
dropout_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544152
dropout_3/PartitionedCall­
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_54615dense_13_54617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_544382"
 dense_13/StatefulPartitionedCalló
re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_544592
re_lu_4/PartitionedCallš
dropout_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544842
dropout_4/PartitionedCall­
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_14_54622dense_14_54624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_545072"
 dense_14/StatefulPartitionedCalló
re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_545282
re_lu_5/PartitionedCallš
dropout_5/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545532
dropout_5/PartitionedCallņ
flatten_1/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_545722
flatten_1/PartitionedCall­
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_15_54630dense_15_54632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_545902"
 dense_15/StatefulPartitionedCall
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namedense_12_input
ķ	
Ü
C__inference_dense_10_layer_call_and_return_conditional_losses_54066

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
:’’’’’’’’’
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
:’’’’’’’’’
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
+

G__inference_sequential_2_layer_call_and_return_conditional_losses_54138
dense_8_input
dense_8_53955
batch_normalization_3_53984
batch_normalization_3_53986
batch_normalization_3_53988
batch_normalization_3_53990
dense_9_54015
dense_9_54017
batch_normalization_4_54046
batch_normalization_4_54048
batch_normalization_4_54050
batch_normalization_4_54052
dense_10_54077
dense_10_54079
batch_normalization_5_54108
batch_normalization_5_54110
batch_normalization_5_54112
batch_normalization_5_54114
dense_11_54134
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_53955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_539462!
dense_8/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_53984batch_normalization_3_53986batch_normalization_3_53988batch_normalization_3_53990*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536102/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_54015dense_9_54017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_540042!
dense_9/StatefulPartitionedCall°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_54046batch_normalization_4_54048batch_normalization_4_54050batch_normalization_4_54052*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537502/
-batch_normalization_4/StatefulPartitionedCallĮ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_54077dense_10_54079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_540662"
 dense_10/StatefulPartitionedCall±
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_54108batch_normalization_5_54110batch_normalization_5_54112batch_normalization_5_54114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_538902/
-batch_normalization_5/StatefulPartitionedCallÆ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_54134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_541252"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’č
'
_user_specified_namedense_8_input
Ī

B__inference_dense_8_layer_call_and_return_conditional_losses_53946

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	č*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*+
_input_shapes
:’’’’’’’’’č:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_54553

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_56442

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
:’’’’’’’’’<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
Ó

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_53783

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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1Ū
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
³
Ø
5__inference_batch_normalization_3_layer_call_fn_56118

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
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_56503

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
°
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_56425

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’<2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
Ā

C__inference_dense_11_layer_call_and_return_conditional_losses_56343

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’
:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs

C
'__inference_re_lu_4_layer_call_fn_56430

inputs
identityĄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_544592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
ŗ
n
(__inference_dense_11_layer_call_fn_56350

inputs
unknown
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_541252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’
:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
»

G__inference_sequential_4_layer_call_and_return_conditional_losses_55025

inputs
sequential_2_54972
sequential_2_54974
sequential_2_54976
sequential_2_54978
sequential_2_54980
sequential_2_54982
sequential_2_54984
sequential_2_54986
sequential_2_54988
sequential_2_54990
sequential_2_54992
sequential_2_54994
sequential_2_54996
sequential_2_54998
sequential_2_55000
sequential_2_55002
sequential_2_55004
sequential_2_55006
sequential_3_55009
sequential_3_55011
sequential_3_55013
sequential_3_55015
sequential_3_55017
sequential_3_55019
sequential_3_55021
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall’
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_54972sequential_2_54974sequential_2_54976sequential_2_54978sequential_2_54980sequential_2_54982sequential_2_54984sequential_2_54986sequential_2_54988sequential_2_54990sequential_2_54992sequential_2_54994sequential_2_54996sequential_2_54998sequential_2_55000sequential_2_55002sequential_2_55004sequential_2_55006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_542352&
$sequential_2/StatefulPartitionedCallŗ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_55009sequential_3_55011sequential_3_55013sequential_3_55015sequential_3_55017sequential_3_55019sequential_3_55021*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_546682&
$sequential_3/StatefulPartitionedCallĻ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
ų/
Å
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56289

inputs
assignmovingavg_56264
assignmovingavg_1_56270)
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
:’’’’’’’’’
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56264*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_56264*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56264*
_output_shapes
:
2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/56264*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_56264AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/56264*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56270*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_56270*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56270*
_output_shapes
:
2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/56270*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_56270AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/56270*
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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs

Ō
,__inference_sequential_3_layer_call_fn_54685
dense_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_546682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:’’’’’’’’’
(
_user_specified_namedense_12_input
ų/
Å
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_53750

inputs
assignmovingavg_53725
assignmovingavg_1_53731)
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
:’’’’’’’’’
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53725*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53725*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53725*
_output_shapes
:
2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53725*
_output_shapes
:
2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53725AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53725*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53731*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53731*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53731*
_output_shapes
:
2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53731*
_output_shapes
:
2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53731AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53731*
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
:’’’’’’’’’
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
:’’’’’’’’’
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
üi

!__inference__traced_restore_56726
file_prefix#
assignvariableop_dense_8_kernel2
.assignvariableop_1_batch_normalization_3_gamma1
-assignvariableop_2_batch_normalization_3_beta%
!assignvariableop_3_dense_9_kernel#
assignvariableop_4_dense_9_bias2
.assignvariableop_5_batch_normalization_4_gamma1
-assignvariableop_6_batch_normalization_4_beta&
"assignvariableop_7_dense_10_kernel$
 assignvariableop_8_dense_10_bias2
.assignvariableop_9_batch_normalization_5_gamma2
.assignvariableop_10_batch_normalization_5_beta'
#assignvariableop_11_dense_11_kernel'
#assignvariableop_12_dense_12_kernel'
#assignvariableop_13_dense_13_kernel%
!assignvariableop_14_dense_13_bias'
#assignvariableop_15_dense_14_kernel%
!assignvariableop_16_dense_14_bias'
#assignvariableop_17_dense_15_kernel%
!assignvariableop_18_dense_15_bias9
5assignvariableop_19_batch_normalization_3_moving_mean=
9assignvariableop_20_batch_normalization_3_moving_variance9
5assignvariableop_21_batch_normalization_4_moving_mean=
9assignvariableop_22_batch_normalization_4_moving_variance9
5assignvariableop_23_batch_normalization_5_moving_mean=
9assignvariableop_24_batch_normalization_5_moving_variance
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ł

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*å	
valueŪ	BŲ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesĀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
22
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

Identity_1³
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_3_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2²
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batch_normalization_3_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_9_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_9_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_4_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6²
AssignVariableOp_6AssignVariableOp-assignvariableop_6_batch_normalization_4_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_10_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8„
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_10_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_5_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¶
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batch_normalization_5_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_11_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13«
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_13_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_13_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_14_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16©
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_14_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_15_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18©
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_15_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19½
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_3_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Į
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_3_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21½
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_4_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Į
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_4_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23½
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_5_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Į
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_5_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25÷
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242(
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
ó

ų
,__inference_sequential_2_layer_call_fn_55909

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

unknown_16
identity¢StatefulPartitionedCallĪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_543232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ŗ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_56519

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ā

C__inference_dense_11_layer_call_and_return_conditional_losses_54125

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’
:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs
Į

G__inference_sequential_4_layer_call_and_return_conditional_losses_55136

inputs
sequential_2_55083
sequential_2_55085
sequential_2_55087
sequential_2_55089
sequential_2_55091
sequential_2_55093
sequential_2_55095
sequential_2_55097
sequential_2_55099
sequential_2_55101
sequential_2_55103
sequential_2_55105
sequential_2_55107
sequential_2_55109
sequential_2_55111
sequential_2_55113
sequential_2_55115
sequential_2_55117
sequential_3_55120
sequential_3_55122
sequential_3_55124
sequential_3_55126
sequential_3_55128
sequential_3_55130
sequential_3_55132
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_55083sequential_2_55085sequential_2_55087sequential_2_55089sequential_2_55091sequential_2_55093sequential_2_55095sequential_2_55097sequential_2_55099sequential_2_55101sequential_2_55103sequential_2_55105sequential_2_55107sequential_2_55109sequential_2_55111sequential_2_55113sequential_2_55115sequential_2_55117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_543232&
$sequential_2/StatefulPartitionedCallŗ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_55120sequential_3_55122sequential_3_55124sequential_3_55126sequential_3_55128sequential_3_55130sequential_3_55132*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_547162&
$sequential_3/StatefulPartitionedCallĻ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
Ś
}
(__inference_dense_15_layer_call_fn_56543

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
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_545902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_54548

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
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape“
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ĶĢL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ü
C__inference_dense_14_layer_call_and_return_conditional_losses_54507

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
ų/
Å
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_53610

inputs
assignmovingavg_53585
assignmovingavg_1_53591)
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
:’’’’’’’’’2
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
moments/Squeeze_1Ė
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53585*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53585*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpš
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53585*
_output_shapes
:2
AssignMovingAvg/subē
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53585*
_output_shapes
:2
AssignMovingAvg/mul­
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53585AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53585*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŃ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53591*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53591*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpś
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53591*
_output_shapes
:2
AssignMovingAvg_1/subń
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53591*
_output_shapes
:2
AssignMovingAvg_1/mul¹
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53591AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53591*
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
:’’’’’’’’’2
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
:’’’’’’’’’2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ	
Ü
C__inference_dense_10_layer_call_and_return_conditional_losses_56244

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
:’’’’’’’’’
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
:’’’’’’’’’
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’

 
_user_specified_nameinputs

b
)__inference_dropout_3_layer_call_fn_56396

inputs
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_544102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

C
'__inference_re_lu_5_layer_call_fn_56486

inputs
identityĄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_545282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š

C__inference_dense_12_layer_call_and_return_conditional_losses_54373

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_54484

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
ŗ
m
'__inference_dense_8_layer_call_fn_56049

inputs
unknown
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_539462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*+
_input_shapes
:’’’’’’’’’č:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs
'

G__inference_sequential_3_layer_call_and_return_conditional_losses_55996

inputs+
'dense_12_matmul_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOpØ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_12/MatMulq
re_lu_3/ReluReludense_12/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
re_lu_3/Relu
dropout_3/IdentityIdentityre_lu_3/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_3/IdentityØ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldropout_3/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense_13/MatMul§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02!
dense_13/BiasAdd/ReadVariableOp„
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense_13/BiasAddq
re_lu_4/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2
re_lu_4/Relu
dropout_4/IdentityIdentityre_lu_4/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dropout_4/IdentityØ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldropout_4/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/MatMul§
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp„
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/BiasAddq
re_lu_5/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
re_lu_5/Relu
dropout_5/IdentityIdentityre_lu_5/Relu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2
dropout_5/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_5/Identity:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
flatten_1/ReshapeØ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp¢
dense_15/MatMulMatMulflatten_1/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp„
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_15/BiasAdd×
IdentityIdentitydense_15/BiasAdd:output:0^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:’’’’’’’’’:::::::2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ü
C__inference_dense_13_layer_call_and_return_conditional_losses_56411

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
:’’’’’’’’’<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
+

G__inference_sequential_2_layer_call_and_return_conditional_losses_54185
dense_8_input
dense_8_54141
batch_normalization_3_54144
batch_normalization_3_54146
batch_normalization_3_54148
batch_normalization_3_54150
dense_9_54153
dense_9_54155
batch_normalization_4_54158
batch_normalization_4_54160
batch_normalization_4_54162
batch_normalization_4_54164
dense_10_54167
dense_10_54169
batch_normalization_5_54172
batch_normalization_5_54174
batch_normalization_5_54176
batch_normalization_5_54178
dense_11_54181
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_54141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_539462!
dense_8/StatefulPartitionedCall²
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_54144batch_normalization_3_54146batch_normalization_3_54148batch_normalization_3_54150*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_536432/
-batch_normalization_3/StatefulPartitionedCall¼
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_54153dense_9_54155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_540042!
dense_9/StatefulPartitionedCall²
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_54158batch_normalization_4_54160batch_normalization_4_54162batch_normalization_4_54164*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_537832/
-batch_normalization_4/StatefulPartitionedCallĮ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_54167dense_10_54169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_540662"
 dense_10/StatefulPartitionedCall³
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_54172batch_normalization_5_54174batch_normalization_5_54176batch_normalization_5_54178*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_539232/
-batch_normalization_5/StatefulPartitionedCallÆ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_54181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_541252"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:’’’’’’’’’č::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’č
'
_user_specified_namedense_8_input
Ś
}
(__inference_dense_13_layer_call_fn_56420

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
:’’’’’’’’’<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_544382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ē
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_56391

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ī

B__inference_dense_8_layer_call_and_return_conditional_losses_56042

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	č*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*+
_input_shapes
:’’’’’’’’’č:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs

E
)__inference_dropout_4_layer_call_fn_56457

inputs
identityĀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_544842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’<:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs

b
)__inference_dropout_5_layer_call_fn_56508

inputs
identity¢StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_545482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ü
C__inference_dense_14_layer_call_and_return_conditional_losses_56467

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’<
 
_user_specified_nameinputs
	
Ü
C__inference_dense_15_layer_call_and_return_conditional_losses_56534

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü
č
,__inference_sequential_4_layer_call_fn_55576

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

unknown_23
identity¢StatefulPartitionedCallŖ
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*5
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_550252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*
_input_shapesz
x:’’’’’’’’’č:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’č
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ę
serving_default²
R
sequential_2_input<
$serving_default_sequential_2_input:0’’’’’’’’’č@
sequential_30
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ó
×t
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ļ__call__
+š&call_and_return_all_conditional_losses
ń_default_save_signature"Źr
_tf_keras_sequential«r{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_2_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_2_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}}
®A
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
regularization_losses
trainable_variables
	variables
	keras_api
ņ__call__
+ó&call_and_return_all_conditional_losses">
_tf_keras_sequentialķ={"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
7
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
regularization_losses
trainable_variables
 	variables
!	keras_api
ō__call__
+õ&call_and_return_all_conditional_losses"4
_tf_keras_sequentialé3{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
 "
trackable_list_wrapper
®
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
418"
trackable_list_wrapper
Ž
"0
#1
$2
53
64
%5
&6
'7
(8
79
810
)11
*12
+13
,14
915
:16
-17
.18
/19
020
121
222
323
424"
trackable_list_wrapper
Ī
;metrics
<layer_metrics
regularization_losses
trainable_variables

=layers
>layer_regularization_losses
	variables
?non_trainable_variables
ļ__call__
ń_default_save_signature
+š&call_and_return_all_conditional_losses
'š"call_and_return_conditional_losses"
_generic_user_object
-
öserving_default"
signature_map
ä

"kernel
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
÷__call__
+ų&call_and_return_all_conditional_losses"Ē
_tf_keras_layer­{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
“	
Daxis
	#gamma
$beta
5moving_mean
6moving_variance
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
ł__call__
+ś&call_and_return_all_conditional_losses"Ž
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
ņ

%kernel
&bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
ū__call__
+ü&call_and_return_all_conditional_losses"Ė
_tf_keras_layer±{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
“	
Maxis
	'gamma
(beta
7moving_mean
8moving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
ż__call__
+ž&call_and_return_all_conditional_losses"Ž
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ō

)kernel
*bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
’__call__
+&call_and_return_all_conditional_losses"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
“	
Vaxis
	+gamma
,beta
9moving_mean
:moving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
__call__
+&call_and_return_all_conditional_losses"Ž
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ź

-kernel
[regularization_losses
\trainable_variables
]	variables
^	keras_api
__call__
+&call_and_return_all_conditional_losses"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
 "
trackable_list_wrapper
v
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
-11"
trackable_list_wrapper
¦
"0
#1
$2
53
64
%5
&6
'7
(8
79
810
)11
*12
+13
,14
915
:16
-17"
trackable_list_wrapper
°
_metrics
`layer_metrics
regularization_losses
trainable_variables

alayers
blayer_regularization_losses
	variables
cnon_trainable_variables
ņ__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
Ü

.kernel
dregularization_losses
etrainable_variables
f	variables
g	keras_api
__call__
+&call_and_return_all_conditional_losses"æ
_tf_keras_layer„{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
ķ
hregularization_losses
itrainable_variables
j	variables
k	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerĀ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ē
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ö

/kernel
0bias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
__call__
+&call_and_return_all_conditional_losses"Ļ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
ķ
tregularization_losses
utrainable_variables
v	variables
w	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerĀ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ē
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ö

1kernel
2bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ļ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
ń
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerĀ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ė
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ģ
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ł

3kernel
4bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
 "
trackable_list_wrapper
Q
.0
/1
02
13
24
35
46"
trackable_list_wrapper
Q
.0
/1
02
13
24
35
46"
trackable_list_wrapper
µ
metrics
layer_metrics
regularization_losses
trainable_variables
layers
 layer_regularization_losses
 	variables
non_trainable_variables
ō__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
!:	č2dense_8/kernel
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
 :
2dense_9/kernel
:
2dense_9/bias
):'
2batch_normalization_4/gamma
(:&
2batch_normalization_4/beta
!:

2dense_10/kernel
:
2dense_10/bias
):'
2batch_normalization_5/gamma
(:&
2batch_normalization_5/beta
!:
2dense_11/kernel
!:2dense_12/kernel
!:<2dense_13/kernel
:<2dense_13/bias
!:<2dense_14/kernel
:2dense_14/bias
!:2dense_15/kernel
:2dense_15/bias
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
1:/
 (2!batch_normalization_4/moving_mean
5:3
 (2%batch_normalization_4/moving_variance
1:/
 (2!batch_normalization_5/moving_mean
5:3
 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
50
61
72
83
94
:5"
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
µ
metrics
layer_metrics
@regularization_losses
Atrainable_variables
layers
 layer_regularization_losses
B	variables
non_trainable_variables
÷__call__
+ų&call_and_return_all_conditional_losses
'ų"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
<
#0
$1
52
63"
trackable_list_wrapper
µ
metrics
layer_metrics
Eregularization_losses
Ftrainable_variables
layers
 layer_regularization_losses
G	variables
non_trainable_variables
ł__call__
+ś&call_and_return_all_conditional_losses
'ś"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
µ
metrics
 layer_metrics
Iregularization_losses
Jtrainable_variables
”layers
 ¢layer_regularization_losses
K	variables
£non_trainable_variables
ū__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
<
'0
(1
72
83"
trackable_list_wrapper
µ
¤metrics
„layer_metrics
Nregularization_losses
Otrainable_variables
¦layers
 §layer_regularization_losses
P	variables
Ønon_trainable_variables
ż__call__
+ž&call_and_return_all_conditional_losses
'ž"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
µ
©metrics
Ŗlayer_metrics
Rregularization_losses
Strainable_variables
«layers
 ¬layer_regularization_losses
T	variables
­non_trainable_variables
’__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
<
+0
,1
92
:3"
trackable_list_wrapper
µ
®metrics
Ælayer_metrics
Wregularization_losses
Xtrainable_variables
°layers
 ±layer_regularization_losses
Y	variables
²non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
-0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
µ
³metrics
“layer_metrics
[regularization_losses
\trainable_variables
µlayers
 ¶layer_regularization_losses
]	variables
·non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
trackable_list_wrapper
J
50
61
72
83
94
:5"
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
µ
ømetrics
¹layer_metrics
dregularization_losses
etrainable_variables
ŗlayers
 »layer_regularization_losses
f	variables
¼non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
½metrics
¾layer_metrics
hregularization_losses
itrainable_variables
ælayers
 Ąlayer_regularization_losses
j	variables
Įnon_trainable_variables
__call__
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
Āmetrics
Ćlayer_metrics
lregularization_losses
mtrainable_variables
Älayers
 Ålayer_regularization_losses
n	variables
Ęnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
µ
Ēmetrics
Člayer_metrics
pregularization_losses
qtrainable_variables
Élayers
 Źlayer_regularization_losses
r	variables
Ėnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ģmetrics
Ķlayer_metrics
tregularization_losses
utrainable_variables
Īlayers
 Ļlayer_regularization_losses
v	variables
Šnon_trainable_variables
__call__
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
Ńmetrics
Ņlayer_metrics
xregularization_losses
ytrainable_variables
Ólayers
 Ōlayer_regularization_losses
z	variables
Õnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
µ
Ömetrics
×layer_metrics
|regularization_losses
}trainable_variables
Ųlayers
 Łlayer_regularization_losses
~	variables
Śnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ūmetrics
Ülayer_metrics
regularization_losses
trainable_variables
Żlayers
 Žlayer_regularization_losses
	variables
ßnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ąmetrics
įlayer_metrics
regularization_losses
trainable_variables
ālayers
 ćlayer_regularization_losses
	variables
änon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
åmetrics
ęlayer_metrics
regularization_losses
trainable_variables
ēlayers
 člayer_regularization_losses
	variables
énon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
ø
źmetrics
ėlayer_metrics
regularization_losses
trainable_variables
ģlayers
 ķlayer_regularization_losses
	variables
īnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
50
61"
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
.
70
81"
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
.
90
:1"
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
 "
trackable_list_wrapper
ž2ū
,__inference_sequential_4_layer_call_fn_55078
,__inference_sequential_4_layer_call_fn_55631
,__inference_sequential_4_layer_call_fn_55189
,__inference_sequential_4_layer_call_fn_55576Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
G__inference_sequential_4_layer_call_and_return_conditional_losses_55521
G__inference_sequential_4_layer_call_and_return_conditional_losses_54966
G__inference_sequential_4_layer_call_and_return_conditional_losses_55418
G__inference_sequential_4_layer_call_and_return_conditional_losses_54910Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
 __inference__wrapped_model_53514Ā
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
annotationsŖ *2¢/
-*
sequential_2_input’’’’’’’’’č
ž2ū
,__inference_sequential_2_layer_call_fn_54274
,__inference_sequential_2_layer_call_fn_55909
,__inference_sequential_2_layer_call_fn_54362
,__inference_sequential_2_layer_call_fn_55868Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
G__inference_sequential_2_layer_call_and_return_conditional_losses_54185
G__inference_sequential_2_layer_call_and_return_conditional_losses_55827
G__inference_sequential_2_layer_call_and_return_conditional_losses_54138
G__inference_sequential_2_layer_call_and_return_conditional_losses_55753Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ž2ū
,__inference_sequential_3_layer_call_fn_56034
,__inference_sequential_3_layer_call_fn_56015
,__inference_sequential_3_layer_call_fn_54733
,__inference_sequential_3_layer_call_fn_54685Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
G__inference_sequential_3_layer_call_and_return_conditional_losses_55996
G__inference_sequential_3_layer_call_and_return_conditional_losses_55963
G__inference_sequential_3_layer_call_and_return_conditional_losses_54607
G__inference_sequential_3_layer_call_and_return_conditional_losses_54636Ą
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
kwonlydefaultsŖ 
annotationsŖ *
 
ÕBŅ
#__inference_signature_wrapper_55246sequential_2_input"
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
annotationsŖ *
 
Ń2Ī
'__inference_dense_8_layer_call_fn_56049¢
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
annotationsŖ *
 
ģ2é
B__inference_dense_8_layer_call_and_return_conditional_losses_56042¢
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
annotationsŖ *
 
Ø2„
5__inference_batch_normalization_3_layer_call_fn_56118
5__inference_batch_normalization_3_layer_call_fn_56131“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ž2Ū
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56105
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56085“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ń2Ī
'__inference_dense_9_layer_call_fn_56151¢
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
annotationsŖ *
 
ģ2é
B__inference_dense_9_layer_call_and_return_conditional_losses_56142¢
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
annotationsŖ *
 
Ø2„
5__inference_batch_normalization_4_layer_call_fn_56233
5__inference_batch_normalization_4_layer_call_fn_56220“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ž2Ū
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56207
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56187“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_10_layer_call_fn_56253¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_10_layer_call_and_return_conditional_losses_56244¢
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
annotationsŖ *
 
Ø2„
5__inference_batch_normalization_5_layer_call_fn_56335
5__inference_batch_normalization_5_layer_call_fn_56322“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ž2Ū
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56309
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56289“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_11_layer_call_fn_56350¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_11_layer_call_and_return_conditional_losses_56343¢
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
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_12_layer_call_fn_56364¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_12_layer_call_and_return_conditional_losses_56357¢
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
annotationsŖ *
 
Ń2Ī
'__inference_re_lu_3_layer_call_fn_56374¢
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
annotationsŖ *
 
ģ2é
B__inference_re_lu_3_layer_call_and_return_conditional_losses_56369¢
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
annotationsŖ *
 
2
)__inference_dropout_3_layer_call_fn_56396
)__inference_dropout_3_layer_call_fn_56401“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ę2Ć
D__inference_dropout_3_layer_call_and_return_conditional_losses_56386
D__inference_dropout_3_layer_call_and_return_conditional_losses_56391“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_13_layer_call_fn_56420¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_13_layer_call_and_return_conditional_losses_56411¢
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
annotationsŖ *
 
Ń2Ī
'__inference_re_lu_4_layer_call_fn_56430¢
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
annotationsŖ *
 
ģ2é
B__inference_re_lu_4_layer_call_and_return_conditional_losses_56425¢
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
annotationsŖ *
 
2
)__inference_dropout_4_layer_call_fn_56457
)__inference_dropout_4_layer_call_fn_56452“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ę2Ć
D__inference_dropout_4_layer_call_and_return_conditional_losses_56442
D__inference_dropout_4_layer_call_and_return_conditional_losses_56447“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_14_layer_call_fn_56476¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_14_layer_call_and_return_conditional_losses_56467¢
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
annotationsŖ *
 
Ń2Ī
'__inference_re_lu_5_layer_call_fn_56486¢
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
annotationsŖ *
 
ģ2é
B__inference_re_lu_5_layer_call_and_return_conditional_losses_56481¢
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
annotationsŖ *
 
2
)__inference_dropout_5_layer_call_fn_56513
)__inference_dropout_5_layer_call_fn_56508“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ę2Ć
D__inference_dropout_5_layer_call_and_return_conditional_losses_56498
D__inference_dropout_5_layer_call_and_return_conditional_losses_56503“
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
kwonlydefaultsŖ 
annotationsŖ *
 
Ó2Š
)__inference_flatten_1_layer_call_fn_56524¢
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
annotationsŖ *
 
ī2ė
D__inference_flatten_1_layer_call_and_return_conditional_losses_56519¢
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
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_15_layer_call_fn_56543¢
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
annotationsŖ *
 
ķ2ź
C__inference_dense_15_layer_call_and_return_conditional_losses_56534¢
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
annotationsŖ *
 »
 __inference__wrapped_model_53514"6#5$%&8'7()*:+9,-./01234<¢9
2¢/
-*
sequential_2_input’’’’’’’’’č
Ŗ ";Ŗ8
6
sequential_3&#
sequential_3’’’’’’’’’¶
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56085b56#$3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ¶
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_56105b6#5$3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 
5__inference_batch_normalization_3_layer_call_fn_56118U56#$3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
5__inference_batch_normalization_3_layer_call_fn_56131U6#5$3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56187b78'(3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_56207b8'7(3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
5__inference_batch_normalization_4_layer_call_fn_56220U78'(3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

5__inference_batch_normalization_4_layer_call_fn_56233U8'7(3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
¶
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56289b9:+,3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ¶
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_56309b:+9,3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
5__inference_batch_normalization_5_layer_call_fn_56322U9:+,3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

5__inference_batch_normalization_5_layer_call_fn_56335U:+9,3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
£
C__inference_dense_10_layer_call_and_return_conditional_losses_56244\)*/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 {
(__inference_dense_10_layer_call_fn_56253O)*/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
¢
C__inference_dense_11_layer_call_and_return_conditional_losses_56343[-/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’
 z
(__inference_dense_11_layer_call_fn_56350N-/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’¢
C__inference_dense_12_layer_call_and_return_conditional_losses_56357[./¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 z
(__inference_dense_12_layer_call_fn_56364N./¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’£
C__inference_dense_13_layer_call_and_return_conditional_losses_56411\/0/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’<
 {
(__inference_dense_13_layer_call_fn_56420O/0/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’<£
C__inference_dense_14_layer_call_and_return_conditional_losses_56467\12/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "%¢"

0’’’’’’’’’
 {
(__inference_dense_14_layer_call_fn_56476O12/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "’’’’’’’’’£
C__inference_dense_15_layer_call_and_return_conditional_losses_56534\34/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 {
(__inference_dense_15_layer_call_fn_56543O34/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¢
B__inference_dense_8_layer_call_and_return_conditional_losses_56042\"0¢-
&¢#
!
inputs’’’’’’’’’č
Ŗ "%¢"

0’’’’’’’’’
 z
'__inference_dense_8_layer_call_fn_56049O"0¢-
&¢#
!
inputs’’’’’’’’’č
Ŗ "’’’’’’’’’¢
B__inference_dense_9_layer_call_and_return_conditional_losses_56142\%&/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’

 z
'__inference_dense_9_layer_call_fn_56151O%&/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_56386\3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ¤
D__inference_dropout_3_layer_call_and_return_conditional_losses_56391\3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 |
)__inference_dropout_3_layer_call_fn_56396O3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’|
)__inference_dropout_3_layer_call_fn_56401O3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’¤
D__inference_dropout_4_layer_call_and_return_conditional_losses_56442\3¢0
)¢&
 
inputs’’’’’’’’’<
p
Ŗ "%¢"

0’’’’’’’’’<
 ¤
D__inference_dropout_4_layer_call_and_return_conditional_losses_56447\3¢0
)¢&
 
inputs’’’’’’’’’<
p 
Ŗ "%¢"

0’’’’’’’’’<
 |
)__inference_dropout_4_layer_call_fn_56452O3¢0
)¢&
 
inputs’’’’’’’’’<
p
Ŗ "’’’’’’’’’<|
)__inference_dropout_4_layer_call_fn_56457O3¢0
)¢&
 
inputs’’’’’’’’’<
p 
Ŗ "’’’’’’’’’<¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_56498\3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_56503\3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 |
)__inference_dropout_5_layer_call_fn_56508O3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’|
)__inference_dropout_5_layer_call_fn_56513O3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’ 
D__inference_flatten_1_layer_call_and_return_conditional_losses_56519X/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 x
)__inference_flatten_1_layer_call_fn_56524K/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
B__inference_re_lu_3_layer_call_and_return_conditional_losses_56369X/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 v
'__inference_re_lu_3_layer_call_fn_56374K/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
B__inference_re_lu_4_layer_call_and_return_conditional_losses_56425X/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "%¢"

0’’’’’’’’’<
 v
'__inference_re_lu_4_layer_call_fn_56430K/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "’’’’’’’’’<
B__inference_re_lu_5_layer_call_and_return_conditional_losses_56481X/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 v
'__inference_re_lu_5_layer_call_fn_56486K/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Ē
G__inference_sequential_2_layer_call_and_return_conditional_losses_54138|"56#$%&78'()*9:+,-?¢<
5¢2
(%
dense_8_input’’’’’’’’’č
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ē
G__inference_sequential_2_layer_call_and_return_conditional_losses_54185|"6#5$%&8'7()*:+9,-?¢<
5¢2
(%
dense_8_input’’’’’’’’’č
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ą
G__inference_sequential_2_layer_call_and_return_conditional_losses_55753u"56#$%&78'()*9:+,-8¢5
.¢+
!
inputs’’’’’’’’’č
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ą
G__inference_sequential_2_layer_call_and_return_conditional_losses_55827u"6#5$%&8'7()*:+9,-8¢5
.¢+
!
inputs’’’’’’’’’č
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
,__inference_sequential_2_layer_call_fn_54274o"56#$%&78'()*9:+,-?¢<
5¢2
(%
dense_8_input’’’’’’’’’č
p

 
Ŗ "’’’’’’’’’
,__inference_sequential_2_layer_call_fn_54362o"6#5$%&8'7()*:+9,-?¢<
5¢2
(%
dense_8_input’’’’’’’’’č
p 

 
Ŗ "’’’’’’’’’
,__inference_sequential_2_layer_call_fn_55868h"56#$%&78'()*9:+,-8¢5
.¢+
!
inputs’’’’’’’’’č
p

 
Ŗ "’’’’’’’’’
,__inference_sequential_2_layer_call_fn_55909h"6#5$%&8'7()*:+9,-8¢5
.¢+
!
inputs’’’’’’’’’č
p 

 
Ŗ "’’’’’’’’’¼
G__inference_sequential_3_layer_call_and_return_conditional_losses_54607q./01234?¢<
5¢2
(%
dense_12_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ¼
G__inference_sequential_3_layer_call_and_return_conditional_losses_54636q./01234?¢<
5¢2
(%
dense_12_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 “
G__inference_sequential_3_layer_call_and_return_conditional_losses_55963i./012347¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 “
G__inference_sequential_3_layer_call_and_return_conditional_losses_55996i./012347¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
,__inference_sequential_3_layer_call_fn_54685d./01234?¢<
5¢2
(%
dense_12_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
,__inference_sequential_3_layer_call_fn_54733d./01234?¢<
5¢2
(%
dense_12_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
,__inference_sequential_3_layer_call_fn_56015\./012347¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
,__inference_sequential_3_layer_call_fn_56034\./012347¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’Ō
G__inference_sequential_4_layer_call_and_return_conditional_losses_54910"56#$%&78'()*9:+,-./01234D¢A
:¢7
-*
sequential_2_input’’’’’’’’’č
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ō
G__inference_sequential_4_layer_call_and_return_conditional_losses_54966"6#5$%&8'7()*:+9,-./01234D¢A
:¢7
-*
sequential_2_input’’’’’’’’’č
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ē
G__inference_sequential_4_layer_call_and_return_conditional_losses_55418|"56#$%&78'()*9:+,-./012348¢5
.¢+
!
inputs’’’’’’’’’č
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ē
G__inference_sequential_4_layer_call_and_return_conditional_losses_55521|"6#5$%&8'7()*:+9,-./012348¢5
.¢+
!
inputs’’’’’’’’’č
p 

 
Ŗ "%¢"

0’’’’’’’’’
 «
,__inference_sequential_4_layer_call_fn_55078{"56#$%&78'()*9:+,-./01234D¢A
:¢7
-*
sequential_2_input’’’’’’’’’č
p

 
Ŗ "’’’’’’’’’«
,__inference_sequential_4_layer_call_fn_55189{"6#5$%&8'7()*:+9,-./01234D¢A
:¢7
-*
sequential_2_input’’’’’’’’’č
p 

 
Ŗ "’’’’’’’’’
,__inference_sequential_4_layer_call_fn_55576o"56#$%&78'()*9:+,-./012348¢5
.¢+
!
inputs’’’’’’’’’č
p

 
Ŗ "’’’’’’’’’
,__inference_sequential_4_layer_call_fn_55631o"6#5$%&8'7()*:+9,-./012348¢5
.¢+
!
inputs’’’’’’’’’č
p 

 
Ŗ "’’’’’’’’’Ō
#__inference_signature_wrapper_55246¬"6#5$%&8'7()*:+9,-./01234R¢O
¢ 
HŖE
C
sequential_2_input-*
sequential_2_input’’’’’’’’’č";Ŗ8
6
sequential_3&#
sequential_3’’’’’’’’’