ӌ
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
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
list(type)(0?
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
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
dtype0
?
conv1d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_7/kernel
?
-conv1d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_7/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_8/kernel
?
-conv1d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_8/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_16/gamma
?
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_16/beta
?
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_16/moving_mean
?
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_16/moving_variance
?
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_9/kernel
?
-conv1d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_9/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_17/beta
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_17/moving_mean
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_17/moving_variance
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?7B?7 B?7
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+regularization_losses
,trainable_variables
-	variables
.	keras_api
^

/kernel
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
^

Akernel
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
?
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
 
V
0
1
'2
(3
/4
95
:6
A7
O8
P9
W10
X11
?
0
1
'2
(3
)4
*5
/6
97
:8
;9
<10
A11
O12
P13
Q14
R15
W16
X17
?
]non_trainable_variables
regularization_losses
^layer_metrics
_metrics
`layer_regularization_losses

alayers
trainable_variables
	variables
 
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
trainable_variables
blayer_metrics
cmetrics
dlayer_regularization_losses

elayers
fnon_trainable_variables
	variables
 
 
 
?
regularization_losses
trainable_variables
glayer_metrics
hmetrics
ilayer_regularization_losses

jlayers
knon_trainable_variables
	variables
ec
VARIABLE_VALUEconv1d_transpose_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
trainable_variables
llayer_metrics
mmetrics
nlayer_regularization_losses

olayers
pnon_trainable_variables
 	variables
 
 
 
?
"regularization_losses
#trainable_variables
qlayer_metrics
rmetrics
slayer_regularization_losses

tlayers
unon_trainable_variables
$	variables
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
)2
*3
?
+regularization_losses
,trainable_variables
vlayer_metrics
wmetrics
xlayer_regularization_losses

ylayers
znon_trainable_variables
-	variables
ec
VARIABLE_VALUEconv1d_transpose_8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

/0

/0
?
0regularization_losses
1trainable_variables
{layer_metrics
|metrics
}layer_regularization_losses

~layers
non_trainable_variables
2	variables
 
 
 
?
4regularization_losses
5trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
6	variables
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
;2
<3
?
=regularization_losses
>trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
?	variables
ec
VARIABLE_VALUEconv1d_transpose_9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

A0

A0
?
Bregularization_losses
Ctrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
D	variables
 
 
 
?
Fregularization_losses
Gtrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
H	variables
 
 
 
?
Jregularization_losses
Ktrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
L	variables
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

O0
P1
Q2
R3
?
Sregularization_losses
Ttrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
U	variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
?
Yregularization_losses
Ztrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
[	variables
*
)0
*1
;2
<3
Q4
R5
 
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
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

)0
*1
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

;0
<1
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

Q0
R1
 
 
 
 
 
z
serving_default_input_6Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6dense_12/kernelconv1d_transpose_7/kernel&batch_normalization_15/moving_variancebatch_normalization_15/gamma"batch_normalization_15/moving_meanbatch_normalization_15/betaconv1d_transpose_8/kernel&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betaconv1d_transpose_9/kernel&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betadense_13/kerneldense_13/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_52639
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp-conv1d_transpose_7/kernel/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp-conv1d_transpose_8/kernel/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp-conv1d_transpose_9/kernel/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_53617
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kernelconv1d_transpose_7/kernelbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv1d_transpose_8/kernelbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv1d_transpose_9/kernelbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_13/kerneldense_13/bias*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_53681??
?
^
B__inference_re_lu_9_layer_call_and_return_conditional_losses_52105

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_5_layer_call_fn_53145

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_520892
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53283

inputs
assignmovingavg_53258
assignmovingavg_1_53264)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53258*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53258*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53258*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53258*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53258AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53258*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53264*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53264*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53264*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53264*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53264AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53264*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53191

inputs
assignmovingavg_53166
assignmovingavg_1_53172)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53166*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53166*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53166*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53166*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53166AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53166*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53172*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53172*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53172*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53172*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53172AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53172*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53494

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_52214

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_11_layer_call_and_return_conditional_losses_53351

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_10_layer_call_and_return_conditional_losses_52156

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_51535

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?A
?
B__inference_Decoder_layer_call_and_return_conditional_losses_52354
input_6
dense_12_52073
conv1d_transpose_7_52097 
batch_normalization_15_52139 
batch_normalization_15_52141 
batch_normalization_15_52143 
batch_normalization_15_52145
conv1d_transpose_8_52148 
batch_normalization_16_52190 
batch_normalization_16_52192 
batch_normalization_16_52194 
batch_normalization_16_52196
conv1d_transpose_9_52199 
batch_normalization_17_52317 
batch_normalization_17_52319 
batch_normalization_17_52321 
batch_normalization_17_52323
dense_13_52348
dense_13_52350
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall?*conv1d_transpose_9/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_52073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_520642"
 dense_12/StatefulPartitionedCall?
reshape_5/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_520892
reshape_5/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_7_52097*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_515352,
*conv1d_transpose_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_521052
re_lu_9/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0batch_normalization_15_52139batch_normalization_15_52141batch_normalization_15_52143batch_normalization_15_52145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5163920
.batch_normalization_15/StatefulPartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv1d_transpose_8_52148*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_517202,
*conv1d_transpose_8/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_521562
re_lu_10/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0batch_normalization_16_52190batch_normalization_16_52192batch_normalization_16_52194batch_normalization_16_52196*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_5182420
.batch_normalization_16/StatefulPartitionedCall?
*conv1d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv1d_transpose_9_52199*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_519052,
*conv1d_transpose_9/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall3conv1d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_522142
flatten_5/PartitionedCall?
re_lu_11/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_11_layer_call_and_return_conditional_losses_522272
re_lu_11/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall!re_lu_11/PartitionedCall:output:0batch_normalization_17_52317batch_normalization_17_52319batch_normalization_17_52321batch_normalization_17_52323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_5227020
.batch_normalization_17/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_13_52348dense_13_52350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_523372"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall+^conv1d_transpose_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2X
*conv1d_transpose_9/StatefulPartitionedCall*conv1d_transpose_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
??
?
B__inference_Decoder_layer_call_and_return_conditional_losses_53031

inputs+
'dense_12_matmul_readvariableop_resourceL
Hconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resource@
<batch_normalization_15_batchnorm_mul_readvariableop_resource>
:batch_normalization_15_batchnorm_readvariableop_1_resource>
:batch_normalization_15_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource@
<batch_normalization_16_batchnorm_mul_readvariableop_resource>
:batch_normalization_16_batchnorm_readvariableop_1_resource>
:batch_normalization_16_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource@
<batch_normalization_17_batchnorm_mul_readvariableop_resource>
:batch_normalization_17_batchnorm_readvariableop_1_resource>
:batch_normalization_17_batchnorm_readvariableop_2_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??/batch_normalization_15/batchnorm/ReadVariableOp?1batch_normalization_15/batchnorm/ReadVariableOp_1?1batch_normalization_15/batchnorm/ReadVariableOp_2?3batch_normalization_15/batchnorm/mul/ReadVariableOp?/batch_normalization_16/batchnorm/ReadVariableOp?1batch_normalization_16/batchnorm/ReadVariableOp_1?1batch_normalization_16/batchnorm/ReadVariableOp_2?3batch_normalization_16/batchnorm/mul/ReadVariableOp?/batch_normalization_17/batchnorm/ReadVariableOp?1batch_normalization_17/batchnorm/ReadVariableOp_1?1batch_normalization_17/batchnorm/ReadVariableOp_2?3batch_normalization_17/batchnorm/mul/ReadVariableOp??conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMulk
reshape_5/ShapeShapedense_12/MatMul:product:0*
T0*
_output_shapes
:2
reshape_5/Shape?
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_5/strided_slice/stack?
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_1?
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_2?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_5/strided_slicex
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/1x
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/2?
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_5/Reshape/shape?
reshape_5/ReshapeReshapedense_12/MatMul:product:0 reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_5/Reshape~
conv1d_transpose_7/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_7/Shape?
&conv1d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_7/strided_slice/stack?
(conv1d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice/stack_1?
(conv1d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice/stack_2?
 conv1d_transpose_7/strided_sliceStridedSlice!conv1d_transpose_7/Shape:output:0/conv1d_transpose_7/strided_slice/stack:output:01conv1d_transpose_7/strided_slice/stack_1:output:01conv1d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_7/strided_slice?
(conv1d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice_1/stack?
*conv1d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_7/strided_slice_1/stack_1?
*conv1d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_7/strided_slice_1/stack_2?
"conv1d_transpose_7/strided_slice_1StridedSlice!conv1d_transpose_7/Shape:output:01conv1d_transpose_7/strided_slice_1/stack:output:03conv1d_transpose_7/strided_slice_1/stack_1:output:03conv1d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_7/strided_slice_1v
conv1d_transpose_7/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_7/mul/y?
conv1d_transpose_7/mulMul+conv1d_transpose_7/strided_slice_1:output:0!conv1d_transpose_7/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_7/mulz
conv1d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_7/stack/2?
conv1d_transpose_7/stackPack)conv1d_transpose_7/strided_slice:output:0conv1d_transpose_7/mul:z:0#conv1d_transpose_7/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_7/stack?
2conv1d_transpose_7/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_7/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_7/conv1d_transpose/ExpandDims
ExpandDimsreshape_5/Reshape:output:0;conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_7/conv1d_transpose/ExpandDims?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_7/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_7/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_7/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_7/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_7/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_7/stack:output:0@conv1d_transpose_7/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_7/conv1d_transpose/strided_slice?
9conv1d_transpose_7/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_7/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_7/stack:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_7/conv1d_transpose/strided_slice_1?
3conv1d_transpose_7/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_7/conv1d_transpose/concat/values_1?
/conv1d_transpose_7/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_7/conv1d_transpose/concat/axis?
*conv1d_transpose_7/conv1d_transpose/concatConcatV2:conv1d_transpose_7/conv1d_transpose/strided_slice:output:0<conv1d_transpose_7/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_7/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_7/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_7/conv1d_transpose/concat?
#conv1d_transpose_7/conv1d_transposeConv2DBackpropInput3conv1d_transpose_7/conv1d_transpose/concat:output:09conv1d_transpose_7/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_7/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_7/conv1d_transpose?
+conv1d_transpose_7/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_7/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_7/conv1d_transpose/Squeeze?
re_lu_9/ReluRelu4conv1d_transpose_7/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
re_lu_9/Relu?
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp?
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_15/batchnorm/add/y?
$batch_normalization_15/batchnorm/addAddV27batch_normalization_15/batchnorm/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add?
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt?
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp?
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul?
&batch_normalization_15/batchnorm/mul_1Mulre_lu_9/Relu:activations:0(batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/mul_1?
1batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_1?
&batch_normalization_15/batchnorm/mul_2Mul9batch_normalization_15/batchnorm/ReadVariableOp_1:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2?
1batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_2?
$batch_normalization_15/batchnorm/subSub9batch_normalization_15/batchnorm/ReadVariableOp_2:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub?
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/add_1?
conv1d_transpose_8/ShapeShape*batch_normalization_15/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_8/Shape?
&conv1d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_8/strided_slice/stack?
(conv1d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice/stack_1?
(conv1d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice/stack_2?
 conv1d_transpose_8/strided_sliceStridedSlice!conv1d_transpose_8/Shape:output:0/conv1d_transpose_8/strided_slice/stack:output:01conv1d_transpose_8/strided_slice/stack_1:output:01conv1d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_8/strided_slice?
(conv1d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice_1/stack?
*conv1d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_8/strided_slice_1/stack_1?
*conv1d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_8/strided_slice_1/stack_2?
"conv1d_transpose_8/strided_slice_1StridedSlice!conv1d_transpose_8/Shape:output:01conv1d_transpose_8/strided_slice_1/stack:output:03conv1d_transpose_8/strided_slice_1/stack_1:output:03conv1d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_8/strided_slice_1v
conv1d_transpose_8/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_8/mul/y?
conv1d_transpose_8/mulMul+conv1d_transpose_8/strided_slice_1:output:0!conv1d_transpose_8/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_8/mulz
conv1d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_8/stack/2?
conv1d_transpose_8/stackPack)conv1d_transpose_8/strided_slice:output:0conv1d_transpose_8/mul:z:0#conv1d_transpose_8/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_8/stack?
2conv1d_transpose_8/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_8/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_8/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_15/batchnorm/add_1:z:0;conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_8/conv1d_transpose/ExpandDims?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_8/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_8/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_8/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_8/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_8/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_8/stack:output:0@conv1d_transpose_8/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_8/conv1d_transpose/strided_slice?
9conv1d_transpose_8/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_8/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_8/stack:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_8/conv1d_transpose/strided_slice_1?
3conv1d_transpose_8/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_8/conv1d_transpose/concat/values_1?
/conv1d_transpose_8/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_8/conv1d_transpose/concat/axis?
*conv1d_transpose_8/conv1d_transpose/concatConcatV2:conv1d_transpose_8/conv1d_transpose/strided_slice:output:0<conv1d_transpose_8/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_8/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_8/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_8/conv1d_transpose/concat?
#conv1d_transpose_8/conv1d_transposeConv2DBackpropInput3conv1d_transpose_8/conv1d_transpose/concat:output:09conv1d_transpose_8/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_8/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_8/conv1d_transpose?
+conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_8/conv1d_transpose/Squeeze?
re_lu_10/ReluRelu4conv1d_transpose_8/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
re_lu_10/Relu?
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp?
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_16/batchnorm/add/y?
$batch_normalization_16/batchnorm/addAddV27batch_normalization_16/batchnorm/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add?
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt?
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp?
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul?
&batch_normalization_16/batchnorm/mul_1Mulre_lu_10/Relu:activations:0(batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1?
1batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_1?
&batch_normalization_16/batchnorm/mul_2Mul9batch_normalization_16/batchnorm/ReadVariableOp_1:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2?
1batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_2?
$batch_normalization_16/batchnorm/subSub9batch_normalization_16/batchnorm/ReadVariableOp_2:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub?
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1?
conv1d_transpose_9/ShapeShape*batch_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_9/Shape?
&conv1d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_9/strided_slice/stack?
(conv1d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice/stack_1?
(conv1d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice/stack_2?
 conv1d_transpose_9/strided_sliceStridedSlice!conv1d_transpose_9/Shape:output:0/conv1d_transpose_9/strided_slice/stack:output:01conv1d_transpose_9/strided_slice/stack_1:output:01conv1d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_9/strided_slice?
(conv1d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice_1/stack?
*conv1d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_9/strided_slice_1/stack_1?
*conv1d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_9/strided_slice_1/stack_2?
"conv1d_transpose_9/strided_slice_1StridedSlice!conv1d_transpose_9/Shape:output:01conv1d_transpose_9/strided_slice_1/stack:output:03conv1d_transpose_9/strided_slice_1/stack_1:output:03conv1d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_9/strided_slice_1v
conv1d_transpose_9/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_9/mul/y?
conv1d_transpose_9/mulMul+conv1d_transpose_9/strided_slice_1:output:0!conv1d_transpose_9/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_9/mulz
conv1d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_9/stack/2?
conv1d_transpose_9/stackPack)conv1d_transpose_9/strided_slice:output:0conv1d_transpose_9/mul:z:0#conv1d_transpose_9/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_9/stack?
2conv1d_transpose_9/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_9/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_9/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_16/batchnorm/add_1:z:0;conv1d_transpose_9/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_9/conv1d_transpose/ExpandDims?
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_9/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_9/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_9/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_9/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_9/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_9/stack:output:0@conv1d_transpose_9/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_9/conv1d_transpose/strided_slice?
9conv1d_transpose_9/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_9/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_9/stack:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_9/conv1d_transpose/strided_slice_1?
3conv1d_transpose_9/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_9/conv1d_transpose/concat/values_1?
/conv1d_transpose_9/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_9/conv1d_transpose/concat/axis?
*conv1d_transpose_9/conv1d_transpose/concatConcatV2:conv1d_transpose_9/conv1d_transpose/strided_slice:output:0<conv1d_transpose_9/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_9/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_9/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_9/conv1d_transpose/concat?
#conv1d_transpose_9/conv1d_transposeConv2DBackpropInput3conv1d_transpose_9/conv1d_transpose/concat:output:09conv1d_transpose_9/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_9/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_9/conv1d_transpose?
+conv1d_transpose_9/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_9/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_9/conv1d_transpose/Squeezes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshape4conv1d_transpose_9/conv1d_transpose/Squeeze:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshapet
re_lu_11/ReluReluflatten_5/Reshape:output:0*
T0*'
_output_shapes
:?????????2
re_lu_11/Relu?
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp?
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_17/batchnorm/add/y?
$batch_normalization_17/batchnorm/addAddV27batch_normalization_17/batchnorm/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/add?
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/Rsqrt?
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp?
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/mul?
&batch_normalization_17/batchnorm/mul_1Mulre_lu_11/Relu:activations:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/mul_1?
1batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_1?
&batch_normalization_17/batchnorm/mul_2Mul9batch_normalization_17/batchnorm/ReadVariableOp_1:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/mul_2?
1batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_2?
$batch_normalization_17/batchnorm/subSub9batch_normalization_17/batchnorm/ReadVariableOp_2:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/sub?
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/add_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdds
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Tanh?
IdentityIdentitydense_13/Tanh:y:00^batch_normalization_15/batchnorm/ReadVariableOp2^batch_normalization_15/batchnorm/ReadVariableOp_12^batch_normalization_15/batchnorm/ReadVariableOp_24^batch_normalization_15/batchnorm/mul/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp2^batch_normalization_16/batchnorm/ReadVariableOp_12^batch_normalization_16/batchnorm/ReadVariableOp_24^batch_normalization_16/batchnorm/mul/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp2^batch_normalization_17/batchnorm/ReadVariableOp_12^batch_normalization_17/batchnorm/ReadVariableOp_24^batch_normalization_17/batchnorm/mul/ReadVariableOp@^conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2f
1batch_normalization_15/batchnorm/ReadVariableOp_11batch_normalization_15/batchnorm/ReadVariableOp_12f
1batch_normalization_15/batchnorm/ReadVariableOp_21batch_normalization_15/batchnorm/ReadVariableOp_22j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2f
1batch_normalization_16/batchnorm/ReadVariableOp_11batch_normalization_16/batchnorm/ReadVariableOp_12f
1batch_normalization_16/batchnorm/ReadVariableOp_21batch_normalization_16/batchnorm/ReadVariableOp_22j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2f
1batch_normalization_17/batchnorm/ReadVariableOp_11batch_normalization_17/batchnorm/ReadVariableOp_12f
1batch_normalization_17/batchnorm/ReadVariableOp_21batch_normalization_17/batchnorm/ReadVariableOp_22j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53412

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_53140

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_53531

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_53316

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_518242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_12_layer_call_and_return_conditional_losses_52064

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_51824

inputs
assignmovingavg_51799
assignmovingavg_1_51805)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51799*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_51799*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51799*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51799*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_51799AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51799*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51805*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_51805*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51805*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51805*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_51805AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51805*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
D
(__inference_re_lu_11_layer_call_fn_53356

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_11_layer_call_and_return_conditional_losses_522272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_53507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_522702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?A
?
B__inference_Decoder_layer_call_and_return_conditional_losses_52463

inputs
dense_12_52413
conv1d_transpose_7_52417 
batch_normalization_15_52421 
batch_normalization_15_52423 
batch_normalization_15_52425 
batch_normalization_15_52427
conv1d_transpose_8_52430 
batch_normalization_16_52434 
batch_normalization_16_52436 
batch_normalization_16_52438 
batch_normalization_16_52440
conv1d_transpose_9_52443 
batch_normalization_17_52448 
batch_normalization_17_52450 
batch_normalization_17_52452 
batch_normalization_17_52454
dense_13_52457
dense_13_52459
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall?*conv1d_transpose_9/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_52413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_520642"
 dense_12/StatefulPartitionedCall?
reshape_5/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_520892
reshape_5/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_7_52417*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_515352,
*conv1d_transpose_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_521052
re_lu_9/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0batch_normalization_15_52421batch_normalization_15_52423batch_normalization_15_52425batch_normalization_15_52427*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5163920
.batch_normalization_15/StatefulPartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv1d_transpose_8_52430*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_517202,
*conv1d_transpose_8/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_521562
re_lu_10/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0batch_normalization_16_52434batch_normalization_16_52436batch_normalization_16_52438batch_normalization_16_52440*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_5182420
.batch_normalization_16/StatefulPartitionedCall?
*conv1d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv1d_transpose_9_52443*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_519052,
*conv1d_transpose_9/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall3conv1d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_522142
flatten_5/PartitionedCall?
re_lu_11/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_11_layer_call_and_return_conditional_losses_522272
re_lu_11/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall!re_lu_11/PartitionedCall:output:0batch_normalization_17_52448batch_normalization_17_52450batch_normalization_17_52452batch_normalization_17_52454*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_5227020
.batch_normalization_17/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_13_52457dense_13_52459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_523372"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall+^conv1d_transpose_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2X
*conv1d_transpose_9/StatefulPartitionedCall*conv1d_transpose_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53392

inputs
assignmovingavg_53367
assignmovingavg_1_53373)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53367*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53367*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53367*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53367*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53367AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53367*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53373*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53373*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53373*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53373*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53373AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53373*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_Decoder_layer_call_fn_52596
input_6
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_525572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
?
x
2__inference_conv1d_transpose_9_layer_call_fn_51913

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_519052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
}
(__inference_dense_13_layer_call_fn_53540

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_523372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_Decoder_layer_call_fn_52502
input_6
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_524632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
?
?
6__inference_batch_normalization_16_layer_call_fn_53329

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_518572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_53224

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_516392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?1
?	
__inference__traced_save_53617
file_prefix.
*savev2_dense_12_kernel_read_readvariableop8
4savev2_conv1d_transpose_7_kernel_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_8_kernel_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_9_kernel_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop4savev2_conv1d_transpose_7_kernel_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop4savev2_conv1d_transpose_8_kernel_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop4savev2_conv1d_transpose_9_kernel_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
D
(__inference_re_lu_10_layer_call_fn_53247

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_521562
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_53438

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_520422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_51672

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
x
2__inference_conv1d_transpose_8_layer_call_fn_51728

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_517202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_53237

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_516722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_10_layer_call_and_return_conditional_losses_53242

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_53425

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_520092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_51498
input_63
/decoder_dense_12_matmul_readvariableop_resourceT
Pdecoder_conv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resourceD
@decoder_batch_normalization_15_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_15_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_15_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_15_batchnorm_readvariableop_2_resourceT
Pdecoder_conv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resourceD
@decoder_batch_normalization_16_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_16_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_16_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_16_batchnorm_readvariableop_2_resourceT
Pdecoder_conv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resourceD
@decoder_batch_normalization_17_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_17_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_17_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_17_batchnorm_readvariableop_2_resource3
/decoder_dense_13_matmul_readvariableop_resource4
0decoder_dense_13_biasadd_readvariableop_resource
identity??7Decoder/batch_normalization_15/batchnorm/ReadVariableOp?9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_1?9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_2?;Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOp?7Decoder/batch_normalization_16/batchnorm/ReadVariableOp?9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_1?9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_2?;Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOp?7Decoder/batch_normalization_17/batchnorm/ReadVariableOp?9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_1?9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_2?;Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOp?GDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?GDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?GDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?&Decoder/dense_12/MatMul/ReadVariableOp?'Decoder/dense_13/BiasAdd/ReadVariableOp?&Decoder/dense_13/MatMul/ReadVariableOp?
&Decoder/dense_12/MatMul/ReadVariableOpReadVariableOp/decoder_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Decoder/dense_12/MatMul/ReadVariableOp?
Decoder/dense_12/MatMulMatMulinput_6.Decoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Decoder/dense_12/MatMul?
Decoder/reshape_5/ShapeShape!Decoder/dense_12/MatMul:product:0*
T0*
_output_shapes
:2
Decoder/reshape_5/Shape?
%Decoder/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Decoder/reshape_5/strided_slice/stack?
'Decoder/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Decoder/reshape_5/strided_slice/stack_1?
'Decoder/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Decoder/reshape_5/strided_slice/stack_2?
Decoder/reshape_5/strided_sliceStridedSlice Decoder/reshape_5/Shape:output:0.Decoder/reshape_5/strided_slice/stack:output:00Decoder/reshape_5/strided_slice/stack_1:output:00Decoder/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Decoder/reshape_5/strided_slice?
!Decoder/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Decoder/reshape_5/Reshape/shape/1?
!Decoder/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Decoder/reshape_5/Reshape/shape/2?
Decoder/reshape_5/Reshape/shapePack(Decoder/reshape_5/strided_slice:output:0*Decoder/reshape_5/Reshape/shape/1:output:0*Decoder/reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2!
Decoder/reshape_5/Reshape/shape?
Decoder/reshape_5/ReshapeReshape!Decoder/dense_12/MatMul:product:0(Decoder/reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
Decoder/reshape_5/Reshape?
 Decoder/conv1d_transpose_7/ShapeShape"Decoder/reshape_5/Reshape:output:0*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_7/Shape?
.Decoder/conv1d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Decoder/conv1d_transpose_7/strided_slice/stack?
0Decoder/conv1d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_7/strided_slice/stack_1?
0Decoder/conv1d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_7/strided_slice/stack_2?
(Decoder/conv1d_transpose_7/strided_sliceStridedSlice)Decoder/conv1d_transpose_7/Shape:output:07Decoder/conv1d_transpose_7/strided_slice/stack:output:09Decoder/conv1d_transpose_7/strided_slice/stack_1:output:09Decoder/conv1d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Decoder/conv1d_transpose_7/strided_slice?
0Decoder/conv1d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_7/strided_slice_1/stack?
2Decoder/conv1d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_7/strided_slice_1/stack_1?
2Decoder/conv1d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_7/strided_slice_1/stack_2?
*Decoder/conv1d_transpose_7/strided_slice_1StridedSlice)Decoder/conv1d_transpose_7/Shape:output:09Decoder/conv1d_transpose_7/strided_slice_1/stack:output:0;Decoder/conv1d_transpose_7/strided_slice_1/stack_1:output:0;Decoder/conv1d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Decoder/conv1d_transpose_7/strided_slice_1?
 Decoder/conv1d_transpose_7/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Decoder/conv1d_transpose_7/mul/y?
Decoder/conv1d_transpose_7/mulMul3Decoder/conv1d_transpose_7/strided_slice_1:output:0)Decoder/conv1d_transpose_7/mul/y:output:0*
T0*
_output_shapes
: 2 
Decoder/conv1d_transpose_7/mul?
"Decoder/conv1d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Decoder/conv1d_transpose_7/stack/2?
 Decoder/conv1d_transpose_7/stackPack1Decoder/conv1d_transpose_7/strided_slice:output:0"Decoder/conv1d_transpose_7/mul:z:0+Decoder/conv1d_transpose_7/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_7/stack?
:Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dim?
6Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims
ExpandDims"Decoder/reshape_5/Reshape:output:0CDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims?
GDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim?
8Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1
ExpandDimsODecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1?
?Decoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Decoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack?
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1?
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2?
9Decoder/conv1d_transpose_7/conv1d_transpose/strided_sliceStridedSlice)Decoder/conv1d_transpose_7/stack:output:0HDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack:output:0JDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1:output:0JDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Decoder/conv1d_transpose_7/conv1d_transpose/strided_slice?
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack?
CDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1?
CDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2?
;Decoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1StridedSlice)Decoder/conv1d_transpose_7/stack:output:0JDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack:output:0LDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1:output:0LDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Decoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1?
;Decoder/conv1d_transpose_7/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Decoder/conv1d_transpose_7/conv1d_transpose/concat/values_1?
7Decoder/conv1d_transpose_7/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Decoder/conv1d_transpose_7/conv1d_transpose/concat/axis?
2Decoder/conv1d_transpose_7/conv1d_transpose/concatConcatV2BDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice:output:0DDecoder/conv1d_transpose_7/conv1d_transpose/concat/values_1:output:0DDecoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1:output:0@Decoder/conv1d_transpose_7/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Decoder/conv1d_transpose_7/conv1d_transpose/concat?
+Decoder/conv1d_transpose_7/conv1d_transposeConv2DBackpropInput;Decoder/conv1d_transpose_7/conv1d_transpose/concat:output:0ADecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1:output:0?Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Decoder/conv1d_transpose_7/conv1d_transpose?
3Decoder/conv1d_transpose_7/conv1d_transpose/SqueezeSqueeze4Decoder/conv1d_transpose_7/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Decoder/conv1d_transpose_7/conv1d_transpose/Squeeze?
Decoder/re_lu_9/ReluRelu<Decoder/conv1d_transpose_7/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
Decoder/re_lu_9/Relu?
7Decoder/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Decoder/batch_normalization_15/batchnorm/ReadVariableOp?
.Decoder/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Decoder/batch_normalization_15/batchnorm/add/y?
,Decoder/batch_normalization_15/batchnorm/addAddV2?Decoder/batch_normalization_15/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_15/batchnorm/add?
.Decoder/batch_normalization_15/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_15/batchnorm/Rsqrt?
;Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOp?
,Decoder/batch_normalization_15/batchnorm/mulMul2Decoder/batch_normalization_15/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_15/batchnorm/mul?
.Decoder/batch_normalization_15/batchnorm/mul_1Mul"Decoder/re_lu_9/Relu:activations:00Decoder/batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Decoder/batch_normalization_15/batchnorm/mul_1?
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_1?
.Decoder/batch_normalization_15/batchnorm/mul_2MulADecoder/batch_normalization_15/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_15/batchnorm/mul_2?
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_2?
,Decoder/batch_normalization_15/batchnorm/subSubADecoder/batch_normalization_15/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_15/batchnorm/sub?
.Decoder/batch_normalization_15/batchnorm/add_1AddV22Decoder/batch_normalization_15/batchnorm/mul_1:z:00Decoder/batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Decoder/batch_normalization_15/batchnorm/add_1?
 Decoder/conv1d_transpose_8/ShapeShape2Decoder/batch_normalization_15/batchnorm/add_1:z:0*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_8/Shape?
.Decoder/conv1d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Decoder/conv1d_transpose_8/strided_slice/stack?
0Decoder/conv1d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_8/strided_slice/stack_1?
0Decoder/conv1d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_8/strided_slice/stack_2?
(Decoder/conv1d_transpose_8/strided_sliceStridedSlice)Decoder/conv1d_transpose_8/Shape:output:07Decoder/conv1d_transpose_8/strided_slice/stack:output:09Decoder/conv1d_transpose_8/strided_slice/stack_1:output:09Decoder/conv1d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Decoder/conv1d_transpose_8/strided_slice?
0Decoder/conv1d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_8/strided_slice_1/stack?
2Decoder/conv1d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_8/strided_slice_1/stack_1?
2Decoder/conv1d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_8/strided_slice_1/stack_2?
*Decoder/conv1d_transpose_8/strided_slice_1StridedSlice)Decoder/conv1d_transpose_8/Shape:output:09Decoder/conv1d_transpose_8/strided_slice_1/stack:output:0;Decoder/conv1d_transpose_8/strided_slice_1/stack_1:output:0;Decoder/conv1d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Decoder/conv1d_transpose_8/strided_slice_1?
 Decoder/conv1d_transpose_8/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Decoder/conv1d_transpose_8/mul/y?
Decoder/conv1d_transpose_8/mulMul3Decoder/conv1d_transpose_8/strided_slice_1:output:0)Decoder/conv1d_transpose_8/mul/y:output:0*
T0*
_output_shapes
: 2 
Decoder/conv1d_transpose_8/mul?
"Decoder/conv1d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Decoder/conv1d_transpose_8/stack/2?
 Decoder/conv1d_transpose_8/stackPack1Decoder/conv1d_transpose_8/strided_slice:output:0"Decoder/conv1d_transpose_8/mul:z:0+Decoder/conv1d_transpose_8/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_8/stack?
:Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dim?
6Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims
ExpandDims2Decoder/batch_normalization_15/batchnorm/add_1:z:0CDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims?
GDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim?
8Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1
ExpandDimsODecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1?
?Decoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Decoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack?
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1?
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2?
9Decoder/conv1d_transpose_8/conv1d_transpose/strided_sliceStridedSlice)Decoder/conv1d_transpose_8/stack:output:0HDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack:output:0JDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1:output:0JDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Decoder/conv1d_transpose_8/conv1d_transpose/strided_slice?
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack?
CDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1?
CDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2?
;Decoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1StridedSlice)Decoder/conv1d_transpose_8/stack:output:0JDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack:output:0LDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1:output:0LDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Decoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1?
;Decoder/conv1d_transpose_8/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Decoder/conv1d_transpose_8/conv1d_transpose/concat/values_1?
7Decoder/conv1d_transpose_8/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Decoder/conv1d_transpose_8/conv1d_transpose/concat/axis?
2Decoder/conv1d_transpose_8/conv1d_transpose/concatConcatV2BDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice:output:0DDecoder/conv1d_transpose_8/conv1d_transpose/concat/values_1:output:0DDecoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1:output:0@Decoder/conv1d_transpose_8/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Decoder/conv1d_transpose_8/conv1d_transpose/concat?
+Decoder/conv1d_transpose_8/conv1d_transposeConv2DBackpropInput;Decoder/conv1d_transpose_8/conv1d_transpose/concat:output:0ADecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1:output:0?Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Decoder/conv1d_transpose_8/conv1d_transpose?
3Decoder/conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze4Decoder/conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Decoder/conv1d_transpose_8/conv1d_transpose/Squeeze?
Decoder/re_lu_10/ReluRelu<Decoder/conv1d_transpose_8/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
Decoder/re_lu_10/Relu?
7Decoder/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Decoder/batch_normalization_16/batchnorm/ReadVariableOp?
.Decoder/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Decoder/batch_normalization_16/batchnorm/add/y?
,Decoder/batch_normalization_16/batchnorm/addAddV2?Decoder/batch_normalization_16/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_16/batchnorm/add?
.Decoder/batch_normalization_16/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_16/batchnorm/Rsqrt?
;Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOp?
,Decoder/batch_normalization_16/batchnorm/mulMul2Decoder/batch_normalization_16/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_16/batchnorm/mul?
.Decoder/batch_normalization_16/batchnorm/mul_1Mul#Decoder/re_lu_10/Relu:activations:00Decoder/batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Decoder/batch_normalization_16/batchnorm/mul_1?
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_1?
.Decoder/batch_normalization_16/batchnorm/mul_2MulADecoder/batch_normalization_16/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_16/batchnorm/mul_2?
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_2?
,Decoder/batch_normalization_16/batchnorm/subSubADecoder/batch_normalization_16/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_16/batchnorm/sub?
.Decoder/batch_normalization_16/batchnorm/add_1AddV22Decoder/batch_normalization_16/batchnorm/mul_1:z:00Decoder/batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Decoder/batch_normalization_16/batchnorm/add_1?
 Decoder/conv1d_transpose_9/ShapeShape2Decoder/batch_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_9/Shape?
.Decoder/conv1d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Decoder/conv1d_transpose_9/strided_slice/stack?
0Decoder/conv1d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_9/strided_slice/stack_1?
0Decoder/conv1d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_9/strided_slice/stack_2?
(Decoder/conv1d_transpose_9/strided_sliceStridedSlice)Decoder/conv1d_transpose_9/Shape:output:07Decoder/conv1d_transpose_9/strided_slice/stack:output:09Decoder/conv1d_transpose_9/strided_slice/stack_1:output:09Decoder/conv1d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Decoder/conv1d_transpose_9/strided_slice?
0Decoder/conv1d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Decoder/conv1d_transpose_9/strided_slice_1/stack?
2Decoder/conv1d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_9/strided_slice_1/stack_1?
2Decoder/conv1d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Decoder/conv1d_transpose_9/strided_slice_1/stack_2?
*Decoder/conv1d_transpose_9/strided_slice_1StridedSlice)Decoder/conv1d_transpose_9/Shape:output:09Decoder/conv1d_transpose_9/strided_slice_1/stack:output:0;Decoder/conv1d_transpose_9/strided_slice_1/stack_1:output:0;Decoder/conv1d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Decoder/conv1d_transpose_9/strided_slice_1?
 Decoder/conv1d_transpose_9/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Decoder/conv1d_transpose_9/mul/y?
Decoder/conv1d_transpose_9/mulMul3Decoder/conv1d_transpose_9/strided_slice_1:output:0)Decoder/conv1d_transpose_9/mul/y:output:0*
T0*
_output_shapes
: 2 
Decoder/conv1d_transpose_9/mul?
"Decoder/conv1d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Decoder/conv1d_transpose_9/stack/2?
 Decoder/conv1d_transpose_9/stackPack1Decoder/conv1d_transpose_9/strided_slice:output:0"Decoder/conv1d_transpose_9/mul:z:0+Decoder/conv1d_transpose_9/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Decoder/conv1d_transpose_9/stack?
:Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims/dim?
6Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims
ExpandDims2Decoder/batch_normalization_16/batchnorm/add_1:z:0CDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims?
GDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim?
8Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1
ExpandDimsODecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1?
?Decoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Decoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack?
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1?
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2?
9Decoder/conv1d_transpose_9/conv1d_transpose/strided_sliceStridedSlice)Decoder/conv1d_transpose_9/stack:output:0HDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack:output:0JDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1:output:0JDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Decoder/conv1d_transpose_9/conv1d_transpose/strided_slice?
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
ADecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack?
CDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1?
CDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2?
;Decoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1StridedSlice)Decoder/conv1d_transpose_9/stack:output:0JDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack:output:0LDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1:output:0LDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Decoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1?
;Decoder/conv1d_transpose_9/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Decoder/conv1d_transpose_9/conv1d_transpose/concat/values_1?
7Decoder/conv1d_transpose_9/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Decoder/conv1d_transpose_9/conv1d_transpose/concat/axis?
2Decoder/conv1d_transpose_9/conv1d_transpose/concatConcatV2BDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice:output:0DDecoder/conv1d_transpose_9/conv1d_transpose/concat/values_1:output:0DDecoder/conv1d_transpose_9/conv1d_transpose/strided_slice_1:output:0@Decoder/conv1d_transpose_9/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Decoder/conv1d_transpose_9/conv1d_transpose/concat?
+Decoder/conv1d_transpose_9/conv1d_transposeConv2DBackpropInput;Decoder/conv1d_transpose_9/conv1d_transpose/concat:output:0ADecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1:output:0?Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Decoder/conv1d_transpose_9/conv1d_transpose?
3Decoder/conv1d_transpose_9/conv1d_transpose/SqueezeSqueeze4Decoder/conv1d_transpose_9/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Decoder/conv1d_transpose_9/conv1d_transpose/Squeeze?
Decoder/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Decoder/flatten_5/Const?
Decoder/flatten_5/ReshapeReshape<Decoder/conv1d_transpose_9/conv1d_transpose/Squeeze:output:0 Decoder/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
Decoder/flatten_5/Reshape?
Decoder/re_lu_11/ReluRelu"Decoder/flatten_5/Reshape:output:0*
T0*'
_output_shapes
:?????????2
Decoder/re_lu_11/Relu?
7Decoder/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Decoder/batch_normalization_17/batchnorm/ReadVariableOp?
.Decoder/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Decoder/batch_normalization_17/batchnorm/add/y?
,Decoder/batch_normalization_17/batchnorm/addAddV2?Decoder/batch_normalization_17/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_17/batchnorm/add?
.Decoder/batch_normalization_17/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_17/batchnorm/Rsqrt?
;Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOp?
,Decoder/batch_normalization_17/batchnorm/mulMul2Decoder/batch_normalization_17/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_17/batchnorm/mul?
.Decoder/batch_normalization_17/batchnorm/mul_1Mul#Decoder/re_lu_11/Relu:activations:00Decoder/batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.Decoder/batch_normalization_17/batchnorm/mul_1?
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_1?
.Decoder/batch_normalization_17/batchnorm/mul_2MulADecoder/batch_normalization_17/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_17/batchnorm/mul_2?
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_2?
,Decoder/batch_normalization_17/batchnorm/subSubADecoder/batch_normalization_17/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_17/batchnorm/sub?
.Decoder/batch_normalization_17/batchnorm/add_1AddV22Decoder/batch_normalization_17/batchnorm/mul_1:z:00Decoder/batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.Decoder/batch_normalization_17/batchnorm/add_1?
&Decoder/dense_13/MatMul/ReadVariableOpReadVariableOp/decoder_dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Decoder/dense_13/MatMul/ReadVariableOp?
Decoder/dense_13/MatMulMatMul2Decoder/batch_normalization_17/batchnorm/add_1:z:0.Decoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Decoder/dense_13/MatMul?
'Decoder/dense_13/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Decoder/dense_13/BiasAdd/ReadVariableOp?
Decoder/dense_13/BiasAddBiasAdd!Decoder/dense_13/MatMul:product:0/Decoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Decoder/dense_13/BiasAdd?
Decoder/dense_13/TanhTanh!Decoder/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Decoder/dense_13/Tanh?	
IdentityIdentityDecoder/dense_13/Tanh:y:08^Decoder/batch_normalization_15/batchnorm/ReadVariableOp:^Decoder/batch_normalization_15/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_15/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_16/batchnorm/ReadVariableOp:^Decoder/batch_normalization_16/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_16/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_17/batchnorm/ReadVariableOp:^Decoder/batch_normalization_17/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_17/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOpH^Decoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpH^Decoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpH^Decoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp'^Decoder/dense_12/MatMul/ReadVariableOp(^Decoder/dense_13/BiasAdd/ReadVariableOp'^Decoder/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2r
7Decoder/batch_normalization_15/batchnorm/ReadVariableOp7Decoder/batch_normalization_15/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_19Decoder/batch_normalization_15/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_15/batchnorm/ReadVariableOp_29Decoder/batch_normalization_15/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_15/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_16/batchnorm/ReadVariableOp7Decoder/batch_normalization_16/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_19Decoder/batch_normalization_16/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_16/batchnorm/ReadVariableOp_29Decoder/batch_normalization_16/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_16/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_17/batchnorm/ReadVariableOp7Decoder/batch_normalization_17/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_19Decoder/batch_normalization_17/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_17/batchnorm/ReadVariableOp_29Decoder/batch_normalization_17/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_17/batchnorm/mul/ReadVariableOp2?
GDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpGDecoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
GDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpGDecoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
GDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOpGDecoder/conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp2P
&Decoder/dense_12/MatMul/ReadVariableOp&Decoder/dense_12/MatMul/ReadVariableOp2R
'Decoder/dense_13/BiasAdd/ReadVariableOp'Decoder/dense_13/BiasAdd/ReadVariableOp2P
&Decoder/dense_13/MatMul/ReadVariableOp&Decoder/dense_13/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
??
?
B__inference_Decoder_layer_call_and_return_conditional_losses_52859

inputs+
'dense_12_matmul_readvariableop_resourceL
Hconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource0
,batch_normalization_15_assignmovingavg_526952
.batch_normalization_15_assignmovingavg_1_52701@
<batch_normalization_15_batchnorm_mul_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resourceL
Hconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource0
,batch_normalization_16_assignmovingavg_527602
.batch_normalization_16_assignmovingavg_1_52766@
<batch_normalization_16_batchnorm_mul_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resourceL
Hconv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resource0
,batch_normalization_17_assignmovingavg_528272
.batch_normalization_17_assignmovingavg_1_52833@
<batch_normalization_17_batchnorm_mul_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity??:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_15/AssignMovingAvg/ReadVariableOp?<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_15/batchnorm/ReadVariableOp?3batch_normalization_15/batchnorm/mul/ReadVariableOp?:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_16/AssignMovingAvg/ReadVariableOp?<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_16/batchnorm/ReadVariableOp?3batch_normalization_16/batchnorm/mul/ReadVariableOp?:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_17/AssignMovingAvg/ReadVariableOp?<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_17/batchnorm/ReadVariableOp?3batch_normalization_17/batchnorm/mul/ReadVariableOp??conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMulk
reshape_5/ShapeShapedense_12/MatMul:product:0*
T0*
_output_shapes
:2
reshape_5/Shape?
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_5/strided_slice/stack?
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_1?
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_5/strided_slice/stack_2?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_5/strided_slicex
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/1x
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_5/Reshape/shape/2?
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_5/Reshape/shape?
reshape_5/ReshapeReshapedense_12/MatMul:product:0 reshape_5/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_5/Reshape~
conv1d_transpose_7/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_7/Shape?
&conv1d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_7/strided_slice/stack?
(conv1d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice/stack_1?
(conv1d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice/stack_2?
 conv1d_transpose_7/strided_sliceStridedSlice!conv1d_transpose_7/Shape:output:0/conv1d_transpose_7/strided_slice/stack:output:01conv1d_transpose_7/strided_slice/stack_1:output:01conv1d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_7/strided_slice?
(conv1d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_7/strided_slice_1/stack?
*conv1d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_7/strided_slice_1/stack_1?
*conv1d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_7/strided_slice_1/stack_2?
"conv1d_transpose_7/strided_slice_1StridedSlice!conv1d_transpose_7/Shape:output:01conv1d_transpose_7/strided_slice_1/stack:output:03conv1d_transpose_7/strided_slice_1/stack_1:output:03conv1d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_7/strided_slice_1v
conv1d_transpose_7/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_7/mul/y?
conv1d_transpose_7/mulMul+conv1d_transpose_7/strided_slice_1:output:0!conv1d_transpose_7/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_7/mulz
conv1d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_7/stack/2?
conv1d_transpose_7/stackPack)conv1d_transpose_7/strided_slice:output:0conv1d_transpose_7/mul:z:0#conv1d_transpose_7/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_7/stack?
2conv1d_transpose_7/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_7/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_7/conv1d_transpose/ExpandDims
ExpandDimsreshape_5/Reshape:output:0;conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_7/conv1d_transpose/ExpandDims?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_7/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_7/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_7/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_7/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_7/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_7/stack:output:0@conv1d_transpose_7/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_7/conv1d_transpose/strided_slice?
9conv1d_transpose_7/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_7/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_7/stack:output:0Bconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_7/conv1d_transpose/strided_slice_1?
3conv1d_transpose_7/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_7/conv1d_transpose/concat/values_1?
/conv1d_transpose_7/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_7/conv1d_transpose/concat/axis?
*conv1d_transpose_7/conv1d_transpose/concatConcatV2:conv1d_transpose_7/conv1d_transpose/strided_slice:output:0<conv1d_transpose_7/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_7/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_7/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_7/conv1d_transpose/concat?
#conv1d_transpose_7/conv1d_transposeConv2DBackpropInput3conv1d_transpose_7/conv1d_transpose/concat:output:09conv1d_transpose_7/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_7/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_7/conv1d_transpose?
+conv1d_transpose_7/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_7/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_7/conv1d_transpose/Squeeze?
re_lu_9/ReluRelu4conv1d_transpose_7/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
re_lu_9/Relu?
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_15/moments/mean/reduction_indices?
#batch_normalization_15/moments/meanMeanre_lu_9/Relu:activations:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_15/moments/mean?
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_15/moments/StopGradient?
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencere_lu_9/Relu:activations:04batch_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_15/moments/SquaredDifference?
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_15/moments/variance/reduction_indices?
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_15/moments/variance?
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_15/moments/Squeeze?
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1?
,batch_normalization_15/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/52695*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_15/AssignMovingAvg/decay?
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_15_assignmovingavg_52695*
_output_shapes
:*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp?
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/52695*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/sub?
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:05batch_normalization_15/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/52695*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/mul?
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_15_assignmovingavg_52695.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_15/AssignMovingAvg/52695*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_15/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/52701*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_15/AssignMovingAvg_1/decay?
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_15_assignmovingavg_1_52701*
_output_shapes
:*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/52701*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/sub?
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:07batch_normalization_15/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/52701*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/mul?
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_15_assignmovingavg_1_527010batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_15/AssignMovingAvg_1/52701*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_15/batchnorm/add/y?
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add?
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt?
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp?
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul?
&batch_normalization_15/batchnorm/mul_1Mulre_lu_9/Relu:activations:0(batch_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/mul_1?
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2?
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp?
$batch_normalization_15/batchnorm/subSub7batch_normalization_15/batchnorm/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub?
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_15/batchnorm/add_1?
conv1d_transpose_8/ShapeShape*batch_normalization_15/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_8/Shape?
&conv1d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_8/strided_slice/stack?
(conv1d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice/stack_1?
(conv1d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice/stack_2?
 conv1d_transpose_8/strided_sliceStridedSlice!conv1d_transpose_8/Shape:output:0/conv1d_transpose_8/strided_slice/stack:output:01conv1d_transpose_8/strided_slice/stack_1:output:01conv1d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_8/strided_slice?
(conv1d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_8/strided_slice_1/stack?
*conv1d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_8/strided_slice_1/stack_1?
*conv1d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_8/strided_slice_1/stack_2?
"conv1d_transpose_8/strided_slice_1StridedSlice!conv1d_transpose_8/Shape:output:01conv1d_transpose_8/strided_slice_1/stack:output:03conv1d_transpose_8/strided_slice_1/stack_1:output:03conv1d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_8/strided_slice_1v
conv1d_transpose_8/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_8/mul/y?
conv1d_transpose_8/mulMul+conv1d_transpose_8/strided_slice_1:output:0!conv1d_transpose_8/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_8/mulz
conv1d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_8/stack/2?
conv1d_transpose_8/stackPack)conv1d_transpose_8/strided_slice:output:0conv1d_transpose_8/mul:z:0#conv1d_transpose_8/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_8/stack?
2conv1d_transpose_8/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_8/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_8/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_15/batchnorm/add_1:z:0;conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_8/conv1d_transpose/ExpandDims?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_8/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_8/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_8/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_8/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_8/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_8/stack:output:0@conv1d_transpose_8/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_8/conv1d_transpose/strided_slice?
9conv1d_transpose_8/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_8/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_8/stack:output:0Bconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_8/conv1d_transpose/strided_slice_1?
3conv1d_transpose_8/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_8/conv1d_transpose/concat/values_1?
/conv1d_transpose_8/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_8/conv1d_transpose/concat/axis?
*conv1d_transpose_8/conv1d_transpose/concatConcatV2:conv1d_transpose_8/conv1d_transpose/strided_slice:output:0<conv1d_transpose_8/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_8/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_8/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_8/conv1d_transpose/concat?
#conv1d_transpose_8/conv1d_transposeConv2DBackpropInput3conv1d_transpose_8/conv1d_transpose/concat:output:09conv1d_transpose_8/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_8/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_8/conv1d_transpose?
+conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_8/conv1d_transpose/Squeeze?
re_lu_10/ReluRelu4conv1d_transpose_8/conv1d_transpose/Squeeze:output:0*
T0*+
_output_shapes
:?????????2
re_lu_10/Relu?
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_16/moments/mean/reduction_indices?
#batch_normalization_16/moments/meanMeanre_lu_10/Relu:activations:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_16/moments/mean?
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_16/moments/StopGradient?
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencere_lu_10/Relu:activations:04batch_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_16/moments/SquaredDifference?
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_16/moments/variance/reduction_indices?
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_16/moments/variance?
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_16/moments/Squeeze?
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1?
,batch_normalization_16/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/52760*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_16/AssignMovingAvg/decay?
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_16_assignmovingavg_52760*
_output_shapes
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp?
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/52760*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/sub?
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/52760*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/mul?
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_16_assignmovingavg_52760.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_16/AssignMovingAvg/52760*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_16/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/52766*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_16/AssignMovingAvg_1/decay?
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_16_assignmovingavg_1_52766*
_output_shapes
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/52766*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/sub?
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/52766*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/mul?
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_16_assignmovingavg_1_527660batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_16/AssignMovingAvg_1/52766*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_16/batchnorm/add/y?
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add?
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt?
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp?
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul?
&batch_normalization_16/batchnorm/mul_1Mulre_lu_10/Relu:activations:0(batch_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/mul_1?
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2?
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp?
$batch_normalization_16/batchnorm/subSub7batch_normalization_16/batchnorm/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub?
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_16/batchnorm/add_1?
conv1d_transpose_9/ShapeShape*batch_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_9/Shape?
&conv1d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_9/strided_slice/stack?
(conv1d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice/stack_1?
(conv1d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice/stack_2?
 conv1d_transpose_9/strided_sliceStridedSlice!conv1d_transpose_9/Shape:output:0/conv1d_transpose_9/strided_slice/stack:output:01conv1d_transpose_9/strided_slice/stack_1:output:01conv1d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_9/strided_slice?
(conv1d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_9/strided_slice_1/stack?
*conv1d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_9/strided_slice_1/stack_1?
*conv1d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_9/strided_slice_1/stack_2?
"conv1d_transpose_9/strided_slice_1StridedSlice!conv1d_transpose_9/Shape:output:01conv1d_transpose_9/strided_slice_1/stack:output:03conv1d_transpose_9/strided_slice_1/stack_1:output:03conv1d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_9/strided_slice_1v
conv1d_transpose_9/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_9/mul/y?
conv1d_transpose_9/mulMul+conv1d_transpose_9/strided_slice_1:output:0!conv1d_transpose_9/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_9/mulz
conv1d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_9/stack/2?
conv1d_transpose_9/stackPack)conv1d_transpose_9/strided_slice:output:0conv1d_transpose_9/mul:z:0#conv1d_transpose_9/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_9/stack?
2conv1d_transpose_9/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_9/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_9/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_16/batchnorm/add_1:z:0;conv1d_transpose_9/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_9/conv1d_transpose/ExpandDims?
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_9_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_9/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_9/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_9/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_9/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_9/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_9/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_9/stack:output:0@conv1d_transpose_9/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_9/conv1d_transpose/strided_slice?
9conv1d_transpose_9/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_9/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_9/stack:output:0Bconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_9/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_9/conv1d_transpose/strided_slice_1?
3conv1d_transpose_9/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_9/conv1d_transpose/concat/values_1?
/conv1d_transpose_9/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_9/conv1d_transpose/concat/axis?
*conv1d_transpose_9/conv1d_transpose/concatConcatV2:conv1d_transpose_9/conv1d_transpose/strided_slice:output:0<conv1d_transpose_9/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_9/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_9/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_9/conv1d_transpose/concat?
#conv1d_transpose_9/conv1d_transposeConv2DBackpropInput3conv1d_transpose_9/conv1d_transpose/concat:output:09conv1d_transpose_9/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_9/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_9/conv1d_transpose?
+conv1d_transpose_9/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_9/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_9/conv1d_transpose/Squeezes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshape4conv1d_transpose_9/conv1d_transpose/Squeeze:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshapet
re_lu_11/ReluReluflatten_5/Reshape:output:0*
T0*'
_output_shapes
:?????????2
re_lu_11/Relu?
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_17/moments/mean/reduction_indices?
#batch_normalization_17/moments/meanMeanre_lu_11/Relu:activations:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_17/moments/mean?
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_17/moments/StopGradient?
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencere_lu_11/Relu:activations:04batch_normalization_17/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_17/moments/SquaredDifference?
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_17/moments/variance/reduction_indices?
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_17/moments/variance?
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_17/moments/Squeeze?
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_17/moments/Squeeze_1?
,batch_normalization_17/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/52827*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_17/AssignMovingAvg/decay?
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_17_assignmovingavg_52827*
_output_shapes
:*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOp?
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/52827*
_output_shapes
:2,
*batch_normalization_17/AssignMovingAvg/sub?
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/52827*
_output_shapes
:2,
*batch_normalization_17/AssignMovingAvg/mul?
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_17_assignmovingavg_52827.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_17/AssignMovingAvg/52827*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_17/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/52833*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_17/AssignMovingAvg_1/decay?
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_17_assignmovingavg_1_52833*
_output_shapes
:*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/52833*
_output_shapes
:2.
,batch_normalization_17/AssignMovingAvg_1/sub?
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/52833*
_output_shapes
:2.
,batch_normalization_17/AssignMovingAvg_1/mul?
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_17_assignmovingavg_1_528330batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_17/AssignMovingAvg_1/52833*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_17/batchnorm/add/y?
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/add?
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/Rsqrt?
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp?
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/mul?
&batch_normalization_17/batchnorm/mul_1Mulre_lu_11/Relu:activations:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/mul_1?
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_17/batchnorm/mul_2?
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp?
$batch_normalization_17/batchnorm/subSub7batch_normalization_17/batchnorm/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_17/batchnorm/sub?
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_17/batchnorm/add_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdds
dense_13/TanhTanhdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Tanh?
IdentityIdentitydense_13/Tanh:y:0;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_15/AssignMovingAvg/ReadVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_15/batchnorm/ReadVariableOp4^batch_normalization_15/batchnorm/mul/ReadVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_16/AssignMovingAvg/ReadVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp4^batch_normalization_16/batchnorm/mul/ReadVariableOp;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_17/AssignMovingAvg/ReadVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp4^batch_normalization_17/batchnorm/mul/ReadVariableOp@^conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_9/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_52042

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
(__inference_dense_12_layer_call_fn_53127

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_520642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_re_lu_9_layer_call_fn_53155

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_521052
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_53341

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53474

inputs
assignmovingavg_53449
assignmovingavg_1_53455)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53449*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_53449*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53449*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/53449*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_53449AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/53449*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53455*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_53455*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53455*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/53455*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_53455AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/53455*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*#
_output_shapes
:?????????2
batchnorm/addl
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*#
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?-
?
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_51905

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_52290

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?/
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_52009

inputs
assignmovingavg_51984
assignmovingavg_1_51990)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51984*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_51984*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51984*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51984*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_51984AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51984*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51990*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_51990*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51990*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51990*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_51990AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51990*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_51639

inputs
assignmovingavg_51614
assignmovingavg_1_51620)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51614*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_51614*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51614*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/51614*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_51614AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/51614*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51620*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_51620*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51620*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/51620*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_51620AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/51620*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_52089

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?A
?
B__inference_Decoder_layer_call_and_return_conditional_losses_52557

inputs
dense_12_52507
conv1d_transpose_7_52511 
batch_normalization_15_52515 
batch_normalization_15_52517 
batch_normalization_15_52519 
batch_normalization_15_52521
conv1d_transpose_8_52524 
batch_normalization_16_52528 
batch_normalization_16_52530 
batch_normalization_16_52532 
batch_normalization_16_52534
conv1d_transpose_9_52537 
batch_normalization_17_52542 
batch_normalization_17_52544 
batch_normalization_17_52546 
batch_normalization_17_52548
dense_13_52551
dense_13_52553
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall?*conv1d_transpose_9/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_52507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_520642"
 dense_12/StatefulPartitionedCall?
reshape_5/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_520892
reshape_5/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_7_52511*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_515352,
*conv1d_transpose_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_521052
re_lu_9/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0batch_normalization_15_52515batch_normalization_15_52517batch_normalization_15_52519batch_normalization_15_52521*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5167220
.batch_normalization_15/StatefulPartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv1d_transpose_8_52524*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_517202,
*conv1d_transpose_8/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_521562
re_lu_10/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0batch_normalization_16_52528batch_normalization_16_52530batch_normalization_16_52532batch_normalization_16_52534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_5185720
.batch_normalization_16/StatefulPartitionedCall?
*conv1d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv1d_transpose_9_52537*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_519052,
*conv1d_transpose_9/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall3conv1d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_522142
flatten_5/PartitionedCall?
re_lu_11/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_11_layer_call_and_return_conditional_losses_522272
re_lu_11/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall!re_lu_11/PartitionedCall:output:0batch_normalization_17_52542batch_normalization_17_52544batch_normalization_17_52546batch_normalization_17_52548*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_5229020
.batch_normalization_17/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_13_52551dense_13_52553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_523372"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall+^conv1d_transpose_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2X
*conv1d_transpose_9/StatefulPartitionedCall*conv1d_transpose_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_Decoder_layer_call_fn_53113

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
identity??StatefulPartitionedCall?
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_525572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?A
?
B__inference_Decoder_layer_call_and_return_conditional_losses_52407
input_6
dense_12_52357
conv1d_transpose_7_52361 
batch_normalization_15_52365 
batch_normalization_15_52367 
batch_normalization_15_52369 
batch_normalization_15_52371
conv1d_transpose_8_52374 
batch_normalization_16_52378 
batch_normalization_16_52380 
batch_normalization_16_52382 
batch_normalization_16_52384
conv1d_transpose_9_52387 
batch_normalization_17_52392 
batch_normalization_17_52394 
batch_normalization_17_52396 
batch_normalization_17_52398
dense_13_52401
dense_13_52403
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall?*conv1d_transpose_9/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_52357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_520642"
 dense_12/StatefulPartitionedCall?
reshape_5/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_520892
reshape_5/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv1d_transpose_7_52361*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_515352,
*conv1d_transpose_7/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_re_lu_9_layer_call_and_return_conditional_losses_521052
re_lu_9/PartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0batch_normalization_15_52365batch_normalization_15_52367batch_normalization_15_52369batch_normalization_15_52371*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_5167220
.batch_normalization_15/StatefulPartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0conv1d_transpose_8_52374*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_517202,
*conv1d_transpose_8/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_10_layer_call_and_return_conditional_losses_521562
re_lu_10/PartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0batch_normalization_16_52378batch_normalization_16_52380batch_normalization_16_52382batch_normalization_16_52384*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_5185720
.batch_normalization_16/StatefulPartitionedCall?
*conv1d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv1d_transpose_9_52387*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_519052,
*conv1d_transpose_9/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall3conv1d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_522142
flatten_5/PartitionedCall?
re_lu_11/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_11_layer_call_and_return_conditional_losses_522272
re_lu_11/PartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall!re_lu_11/PartitionedCall:output:0batch_normalization_17_52392batch_normalization_17_52394batch_normalization_17_52396batch_normalization_17_52398*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_5229020
.batch_normalization_17/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_13_52401dense_13_52403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_523372"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall+^conv1d_transpose_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2X
*conv1d_transpose_9/StatefulPartitionedCall*conv1d_transpose_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
?
?
C__inference_dense_12_layer_call_and_return_conditional_losses_53120

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_52639
input_6
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_514982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53303

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
'__inference_Decoder_layer_call_fn_53072

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
identity??StatefulPartitionedCall?
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
:?????????*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_524632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53211

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_5_layer_call_fn_53346

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_522142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_52270

inputs
assignmovingavg_52245
assignmovingavg_1_52251)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52245*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_52245*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52245*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/52245*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_52245AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/52245*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52251*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_52251*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52251*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/52251*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_52251AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/52251*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*#
_output_shapes
:?????????2
batchnorm/addl
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*#
_output_shapes
:?????????2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_51857

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_53520

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_522902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
x
2__inference_conv1d_transpose_7_layer_call_fn_51543

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_515352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_51720

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource
identity??,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_11_layer_call_and_return_conditional_losses_52227

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?Q
?
!__inference__traced_restore_53681
file_prefix$
 assignvariableop_dense_12_kernel0
,assignvariableop_1_conv1d_transpose_7_kernel3
/assignvariableop_2_batch_normalization_15_gamma2
.assignvariableop_3_batch_normalization_15_beta9
5assignvariableop_4_batch_normalization_15_moving_mean=
9assignvariableop_5_batch_normalization_15_moving_variance0
,assignvariableop_6_conv1d_transpose_8_kernel3
/assignvariableop_7_batch_normalization_16_gamma2
.assignvariableop_8_batch_normalization_16_beta9
5assignvariableop_9_batch_normalization_16_moving_mean>
:assignvariableop_10_batch_normalization_16_moving_variance1
-assignvariableop_11_conv1d_transpose_9_kernel4
0assignvariableop_12_batch_normalization_17_gamma3
/assignvariableop_13_batch_normalization_17_beta:
6assignvariableop_14_batch_normalization_17_moving_mean>
:assignvariableop_15_batch_normalization_17_moving_variance'
#assignvariableop_16_dense_13_kernel%
!assignvariableop_17_dense_13_bias
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_conv1d_transpose_7_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_15_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_15_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_15_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_15_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv1d_transpose_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_16_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_16_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_16_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_16_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_conv1d_transpose_9_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_17_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_17_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_17_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_17_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_13_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_13_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18?
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
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
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_52337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_re_lu_9_layer_call_and_return_conditional_losses_53150

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_60
serving_default_input_6:0?????????<
dense_130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ǟ
?r
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?n
_tf_keras_network?m{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 2]}}, "name": "reshape_5", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_7", "inbound_nodes": [[["reshape_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["conv1d_transpose_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_8", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_10", "inbound_nodes": [[["conv1d_transpose_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["re_lu_10", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_9", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_9", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["conv1d_transpose_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_11", "inbound_nodes": [[["flatten_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["re_lu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["dense_13", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 2]}}, "name": "reshape_5", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_7", "inbound_nodes": [[["reshape_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["conv1d_transpose_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_8", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_10", "inbound_nodes": [[["conv1d_transpose_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["re_lu_10", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_9", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_9", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["conv1d_transpose_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_11", "inbound_nodes": [[["flatten_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["re_lu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["dense_13", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
?

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 2]}}}
?


kernel
regularization_losses
trainable_variables
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2]}}
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2]}}
?


/kernel
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2]}}
?
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}}
?


Akernel
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_9", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}}
?
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?	
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
 "
trackable_list_wrapper
v
0
1
'2
(3
/4
95
:6
A7
O8
P9
W10
X11"
trackable_list_wrapper
?
0
1
'2
(3
)4
*5
/6
97
:8
;9
<10
A11
O12
P13
Q14
R15
W16
X17"
trackable_list_wrapper
?
]non_trainable_variables
regularization_losses
^layer_metrics
_metrics
`layer_regularization_losses

alayers
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
!:2dense_12/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
trainable_variables
blayer_metrics
cmetrics
dlayer_regularization_losses

elayers
fnon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
trainable_variables
glayer_metrics
hmetrics
ilayer_regularization_losses

jlayers
knon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_7/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
trainable_variables
llayer_metrics
mmetrics
nlayer_regularization_losses

olayers
pnon_trainable_variables
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
#trainable_variables
qlayer_metrics
rmetrics
slayer_regularization_losses

tlayers
unon_trainable_variables
$	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_15/gamma
):'2batch_normalization_15/beta
2:0 (2"batch_normalization_15/moving_mean
6:4 (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
?
+regularization_losses
,trainable_variables
vlayer_metrics
wmetrics
xlayer_regularization_losses

ylayers
znon_trainable_variables
-	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_8/kernel
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
?
0regularization_losses
1trainable_variables
{layer_metrics
|metrics
}layer_regularization_losses

~layers
non_trainable_variables
2	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4regularization_losses
5trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
6	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_16/gamma
):'2batch_normalization_16/beta
2:0 (2"batch_normalization_16/moving_mean
6:4 (2&batch_normalization_16/moving_variance
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
?
=regularization_losses
>trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_9/kernel
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
?
Bregularization_losses
Ctrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fregularization_losses
Gtrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
H	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jregularization_losses
Ktrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
L	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_17/gamma
):'2batch_normalization_17/beta
2:0 (2"batch_normalization_17/moving_mean
6:4 (2&batch_normalization_17/moving_variance
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
?
Sregularization_losses
Ttrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
U	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_13/kernel
:2dense_13/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
Yregularization_losses
Ztrainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
[	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
)0
*1
;2
<3
Q4
R5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
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
.
)0
*1"
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
.
;0
<1"
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
.
Q0
R1"
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
?2?
B__inference_Decoder_layer_call_and_return_conditional_losses_52354
B__inference_Decoder_layer_call_and_return_conditional_losses_53031
B__inference_Decoder_layer_call_and_return_conditional_losses_52859
B__inference_Decoder_layer_call_and_return_conditional_losses_52407?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_51498?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_6?????????
?2?
'__inference_Decoder_layer_call_fn_52596
'__inference_Decoder_layer_call_fn_53072
'__inference_Decoder_layer_call_fn_52502
'__inference_Decoder_layer_call_fn_53113?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_12_layer_call_and_return_conditional_losses_53120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_12_layer_call_fn_53127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_5_layer_call_and_return_conditional_losses_53140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_5_layer_call_fn_53145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_51535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
2__inference_conv1d_transpose_7_layer_call_fn_51543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
B__inference_re_lu_9_layer_call_and_return_conditional_losses_53150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_re_lu_9_layer_call_fn_53155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53191
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53211?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_15_layer_call_fn_53237
6__inference_batch_normalization_15_layer_call_fn_53224?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_51720?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
2__inference_conv1d_transpose_8_layer_call_fn_51728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
C__inference_re_lu_10_layer_call_and_return_conditional_losses_53242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_10_layer_call_fn_53247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53303
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53283?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_16_layer_call_fn_53329
6__inference_batch_normalization_16_layer_call_fn_53316?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_51905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
2__inference_conv1d_transpose_9_layer_call_fn_51913?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
D__inference_flatten_5_layer_call_and_return_conditional_losses_53341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_5_layer_call_fn_53346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_re_lu_11_layer_call_and_return_conditional_losses_53351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_11_layer_call_fn_53356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53494
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53474
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53412
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53392?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_17_layer_call_fn_53507
6__inference_batch_normalization_17_layer_call_fn_53520
6__inference_batch_normalization_17_layer_call_fn_53438
6__inference_batch_normalization_17_layer_call_fn_53425?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_13_layer_call_and_return_conditional_losses_53531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_13_layer_call_fn_53540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_52639input_6"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
B__inference_Decoder_layer_call_and_return_conditional_losses_52354u)*'(/;<9:AQROPWX8?5
.?+
!?
input_6?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_52407u*')(/<9;:AROQPWX8?5
.?+
!?
input_6?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_52859t)*'(/;<9:AQROPWX7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_53031t*')(/<9;:AROQPWX7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
'__inference_Decoder_layer_call_fn_52502h)*'(/;<9:AQROPWX8?5
.?+
!?
input_6?????????
p

 
? "???????????
'__inference_Decoder_layer_call_fn_52596h*')(/<9;:AROQPWX8?5
.?+
!?
input_6?????????
p 

 
? "???????????
'__inference_Decoder_layer_call_fn_53072g)*'(/;<9:AQROPWX7?4
-?*
 ?
inputs?????????
p

 
? "???????????
'__inference_Decoder_layer_call_fn_53113g*')(/<9;:AROQPWX7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
 __inference__wrapped_model_51498{*')(/<9;:AROQPWX0?-
&?#
!?
input_6?????????
? "3?0
.
dense_13"?
dense_13??????????
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53191|)*'(@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_53211|*')(@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
6__inference_batch_normalization_15_layer_call_fn_53224o)*'(@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_15_layer_call_fn_53237o*')(@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53283|;<9:@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_53303|<9;:@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
6__inference_batch_normalization_16_layer_call_fn_53316o;<9:@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
6__inference_batch_normalization_16_layer_call_fn_53329o<9;:@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53392bQROP3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53412bROQP3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53474kQROP<?9
2?/
)?&
inputs??????????????????
p
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_53494kROQP<?9
2?/
)?&
inputs??????????????????
p 
? "%?"
?
0?????????
? ?
6__inference_batch_normalization_17_layer_call_fn_53425UQROP3?0
)?&
 ?
inputs?????????
p
? "???????????
6__inference_batch_normalization_17_layer_call_fn_53438UROQP3?0
)?&
 ?
inputs?????????
p 
? "???????????
6__inference_batch_normalization_17_layer_call_fn_53507^QROP<?9
2?/
)?&
inputs??????????????????
p
? "???????????
6__inference_batch_normalization_17_layer_call_fn_53520^ROQP<?9
2?/
)?&
inputs??????????????????
p 
? "???????????
M__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_51535u<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_7_layer_call_fn_51543h<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
M__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_51720u/<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_8_layer_call_fn_51728h/<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
M__inference_conv1d_transpose_9_layer_call_and_return_conditional_losses_51905uA<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
2__inference_conv1d_transpose_9_layer_call_fn_51913hA<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
C__inference_dense_12_layer_call_and_return_conditional_losses_53120[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
(__inference_dense_12_layer_call_fn_53127N/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_13_layer_call_and_return_conditional_losses_53531\WX/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_13_layer_call_fn_53540OWX/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_flatten_5_layer_call_and_return_conditional_losses_53341n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
)__inference_flatten_5_layer_call_fn_53346a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
C__inference_re_lu_10_layer_call_and_return_conditional_losses_53242r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
(__inference_re_lu_10_layer_call_fn_53247e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
C__inference_re_lu_11_layer_call_and_return_conditional_losses_53351j8?5
.?+
)?&
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
(__inference_re_lu_11_layer_call_fn_53356]8?5
.?+
)?&
inputs??????????????????
? "!????????????????????
B__inference_re_lu_9_layer_call_and_return_conditional_losses_53150r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
'__inference_re_lu_9_layer_call_fn_53155e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
D__inference_reshape_5_layer_call_and_return_conditional_losses_53140\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? |
)__inference_reshape_5_layer_call_fn_53145O/?,
%?"
 ?
inputs?????????
? "???????????
#__inference_signature_wrapper_52639?*')(/<9;:AROQPWX;?8
? 
1?.
,
input_6!?
input_6?????????"3?0
.
dense_13"?
dense_13?????????