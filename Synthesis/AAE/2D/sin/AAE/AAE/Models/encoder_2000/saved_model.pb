??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
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
,
Exp
x"T
y"T"
Ttype:

2
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
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
{
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_47/kernel
t
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel*
_output_shapes
:	?*
dtype0
?
batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_58/gamma
?
0batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_58/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_58/beta
?
/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_58/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_58/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_58/moving_mean
?
6batch_normalization_58/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_58/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_58/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_58/moving_variance
?
:batch_normalization_58/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_58/moving_variance*
_output_shapes	
:?*
dtype0
?
batch_normalization_59/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_59/gamma
?
0batch_normalization_59/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_59/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_59/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_59/beta
?
/batch_normalization_59/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_59/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_59/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_59/moving_mean
?
6batch_normalization_59/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_59/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_59/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_59/moving_variance
?
:batch_normalization_59/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_59/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv1d_transpose_10/kernel
?
.conv1d_transpose_10/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_10/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_60/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_60/gamma
?
0batch_normalization_60/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_60/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_60/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_60/beta
?
/batch_normalization_60/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_60/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_60/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_60/moving_mean
?
6batch_normalization_60/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_60/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_60/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_60/moving_variance
?
:batch_normalization_60/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_60/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv1d_transpose_11/kernel
?
.conv1d_transpose_11/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_11/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_61/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_61/gamma
?
0batch_normalization_61/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_61/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_61/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_61/beta
?
/batch_normalization_61/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_61/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_61/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_61/moving_mean
?
6batch_normalization_61/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_61/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_61/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_61/moving_variance
?
:batch_normalization_61/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_61/moving_variance*
_output_shapes
:*
dtype0
z
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_48/kernel
s
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes

:@*
dtype0
z
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_49/kernel
s
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel*
_output_shapes

:@*
dtype0
?
batch_normalization_62/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_62/gamma
?
0batch_normalization_62/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_62/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_62/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_62/beta
?
/batch_normalization_62/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_62/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_62/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_62/moving_mean
?
6batch_normalization_62/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_62/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_62/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_62/moving_variance
?
:batch_normalization_62/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_62/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
?I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?H
value?HB?H B?H
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	gamma
beta
moving_mean
 moving_variance
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
R
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
^

6kernel
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
R
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
^

Hkernel
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
R
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
^

^kernel
_regularization_losses
`trainable_variables
a	variables
b	keras_api
^

ckernel
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
 
n
0
1
2
.3
/4
65
<6
=7
H8
N9
O10
^11
c12
m13
n14
?
0
1
2
3
 4
.5
/6
07
18
69
<10
=11
>12
?13
H14
N15
O16
P17
Q18
^19
c20
m21
n22
o23
p24
?
regularization_losses
ulayer_metrics
vmetrics
wnon_trainable_variables
trainable_variables
	variables
xlayer_regularization_losses

ylayers
 
[Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
zlayer_metrics
{metrics
|non_trainable_variables
trainable_variables
	variables
}layer_regularization_losses

~layers
 
ge
VARIABLE_VALUEbatch_normalization_58/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_58/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_58/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_58/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
 3
?
!regularization_losses
layer_metrics
?metrics
?non_trainable_variables
"trainable_variables
#	variables
 ?layer_regularization_losses
?layers
 
 
 
?
%regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
&trainable_variables
'	variables
 ?layer_regularization_losses
?layers
 
 
 
?
)regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
*trainable_variables
+	variables
 ?layer_regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_59/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_59/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_59/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_59/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
02
13
?
2regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
3trainable_variables
4	variables
 ?layer_regularization_losses
?layers
fd
VARIABLE_VALUEconv1d_transpose_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

60

60
?
7regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
8trainable_variables
9	variables
 ?layer_regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_60/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_60/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_60/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_60/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
>2
?3
?
@regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Atrainable_variables
B	variables
 ?layer_regularization_losses
?layers
 
 
 
?
Dregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Etrainable_variables
F	variables
 ?layer_regularization_losses
?layers
fd
VARIABLE_VALUEconv1d_transpose_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

H0

H0
?
Iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Jtrainable_variables
K	variables
 ?layer_regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_61/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_61/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_61/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_61/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

N0
O1
P2
Q3
?
Rregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Strainable_variables
T	variables
 ?layer_regularization_losses
?layers
 
 
 
?
Vregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?layers
 
 
 
?
Zregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
[trainable_variables
\	variables
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

^0

^0
?
_regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
`trainable_variables
a	variables
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

c0

c0
?
dregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
etrainable_variables
f	variables
 ?layer_regularization_losses
?layers
 
 
 
?
hregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
itrainable_variables
j	variables
 ?layer_regularization_losses
?layers
 
ge
VARIABLE_VALUEbatch_normalization_62/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_62/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_62/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_62/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
o2
p3
?
qregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
rtrainable_variables
s	variables
 ?layer_regularization_losses
?layers
 
 
F
0
 1
02
13
>4
?5
P6
Q7
o8
p9
 
~
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
14
15
16
 
 
 
 
 
 
 

0
 1
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
00
11
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
>0
?1
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
P0
Q1
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

o0
p1
 
 
{
serving_default_input_19Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19dense_47/kernel&batch_normalization_58/moving_variancebatch_normalization_58/gamma"batch_normalization_58/moving_meanbatch_normalization_58/beta&batch_normalization_59/moving_variancebatch_normalization_59/gamma"batch_normalization_59/moving_meanbatch_normalization_59/betaconv1d_transpose_10/kernel&batch_normalization_60/moving_variancebatch_normalization_60/gamma"batch_normalization_60/moving_meanbatch_normalization_60/betaconv1d_transpose_11/kernel&batch_normalization_61/moving_variancebatch_normalization_61/gamma"batch_normalization_61/moving_meanbatch_normalization_61/betadense_48/kerneldense_49/kernel&batch_normalization_62/moving_variancebatch_normalization_62/gamma"batch_normalization_62/moving_meanbatch_normalization_62/beta*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_41912255
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_47/kernel/Read/ReadVariableOp0batch_normalization_58/gamma/Read/ReadVariableOp/batch_normalization_58/beta/Read/ReadVariableOp6batch_normalization_58/moving_mean/Read/ReadVariableOp:batch_normalization_58/moving_variance/Read/ReadVariableOp0batch_normalization_59/gamma/Read/ReadVariableOp/batch_normalization_59/beta/Read/ReadVariableOp6batch_normalization_59/moving_mean/Read/ReadVariableOp:batch_normalization_59/moving_variance/Read/ReadVariableOp.conv1d_transpose_10/kernel/Read/ReadVariableOp0batch_normalization_60/gamma/Read/ReadVariableOp/batch_normalization_60/beta/Read/ReadVariableOp6batch_normalization_60/moving_mean/Read/ReadVariableOp:batch_normalization_60/moving_variance/Read/ReadVariableOp.conv1d_transpose_11/kernel/Read/ReadVariableOp0batch_normalization_61/gamma/Read/ReadVariableOp/batch_normalization_61/beta/Read/ReadVariableOp6batch_normalization_61/moving_mean/Read/ReadVariableOp:batch_normalization_61/moving_variance/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp0batch_normalization_62/gamma/Read/ReadVariableOp/batch_normalization_62/beta/Read/ReadVariableOp6batch_normalization_62/moving_mean/Read/ReadVariableOp:batch_normalization_62/moving_variance/Read/ReadVariableOpConst*&
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
GPU 2J 8? **
f%R#
!__inference__traced_save_41913556
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_47/kernelbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_variancebatch_normalization_59/gammabatch_normalization_59/beta"batch_normalization_59/moving_mean&batch_normalization_59/moving_varianceconv1d_transpose_10/kernelbatch_normalization_60/gammabatch_normalization_60/beta"batch_normalization_60/moving_mean&batch_normalization_60/moving_varianceconv1d_transpose_11/kernelbatch_normalization_61/gammabatch_normalization_61/beta"batch_normalization_61/moving_mean&batch_normalization_61/moving_variancedense_48/kerneldense_49/kernelbatch_normalization_62/gammabatch_normalization_62/beta"batch_normalization_62/moving_mean&batch_normalization_62/moving_variance*%
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_41913641??
?0
?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913412

inputs
assignmovingavg_41913387
assignmovingavg_1_41913393)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913387*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41913387*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913387*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913387*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41913387AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913387*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913393*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41913393*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913393*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913393*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41913393AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913393*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912519

inputs+
'dense_47_matmul_readvariableop_resource3
/batch_normalization_58_assignmovingavg_419122695
1batch_normalization_58_assignmovingavg_1_41912275@
<batch_normalization_58_batchnorm_mul_readvariableop_resource<
8batch_normalization_58_batchnorm_readvariableop_resource3
/batch_normalization_59_assignmovingavg_419123115
1batch_normalization_59_assignmovingavg_1_41912317@
<batch_normalization_59_batchnorm_mul_readvariableop_resource<
8batch_normalization_59_batchnorm_readvariableop_resourceM
Iconv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resource3
/batch_normalization_60_assignmovingavg_419123755
1batch_normalization_60_assignmovingavg_1_41912381@
<batch_normalization_60_batchnorm_mul_readvariableop_resource<
8batch_normalization_60_batchnorm_readvariableop_resourceM
Iconv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resource3
/batch_normalization_61_assignmovingavg_419124405
1batch_normalization_61_assignmovingavg_1_41912446@
<batch_normalization_61_batchnorm_mul_readvariableop_resource<
8batch_normalization_61_batchnorm_readvariableop_resource+
'dense_48_matmul_readvariableop_resource+
'dense_49_matmul_readvariableop_resource3
/batch_normalization_62_assignmovingavg_419124945
1batch_normalization_62_assignmovingavg_1_41912500@
<batch_normalization_62_batchnorm_mul_readvariableop_resource<
8batch_normalization_62_batchnorm_readvariableop_resource
identity??:batch_normalization_58/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_58/AssignMovingAvg/ReadVariableOp?<batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_58/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_58/batchnorm/ReadVariableOp?3batch_normalization_58/batchnorm/mul/ReadVariableOp?:batch_normalization_59/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_59/AssignMovingAvg/ReadVariableOp?<batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_59/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_59/batchnorm/ReadVariableOp?3batch_normalization_59/batchnorm/mul/ReadVariableOp?:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_60/AssignMovingAvg/ReadVariableOp?<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_60/batchnorm/ReadVariableOp?3batch_normalization_60/batchnorm/mul/ReadVariableOp?:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_61/AssignMovingAvg/ReadVariableOp?<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_61/batchnorm/ReadVariableOp?3batch_normalization_61/batchnorm/mul/ReadVariableOp?:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_62/AssignMovingAvg/ReadVariableOp?<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_62/batchnorm/ReadVariableOp?3batch_normalization_62/batchnorm/mul/ReadVariableOp?@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/MatMul/ReadVariableOp?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMulinputs&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
5batch_normalization_58/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_58/moments/mean/reduction_indices?
#batch_normalization_58/moments/meanMeandense_47/MatMul:product:0>batch_normalization_58/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_58/moments/mean?
+batch_normalization_58/moments/StopGradientStopGradient,batch_normalization_58/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_58/moments/StopGradient?
0batch_normalization_58/moments/SquaredDifferenceSquaredDifferencedense_47/MatMul:product:04batch_normalization_58/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_58/moments/SquaredDifference?
9batch_normalization_58/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_58/moments/variance/reduction_indices?
'batch_normalization_58/moments/varianceMean4batch_normalization_58/moments/SquaredDifference:z:0Bbatch_normalization_58/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_58/moments/variance?
&batch_normalization_58/moments/SqueezeSqueeze,batch_normalization_58/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_58/moments/Squeeze?
(batch_normalization_58/moments/Squeeze_1Squeeze0batch_normalization_58/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_58/moments/Squeeze_1?
,batch_normalization_58/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_58/AssignMovingAvg/41912269*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_58/AssignMovingAvg/decay?
5batch_normalization_58/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_58_assignmovingavg_41912269*
_output_shapes	
:?*
dtype027
5batch_normalization_58/AssignMovingAvg/ReadVariableOp?
*batch_normalization_58/AssignMovingAvg/subSub=batch_normalization_58/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_58/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_58/AssignMovingAvg/41912269*
_output_shapes	
:?2,
*batch_normalization_58/AssignMovingAvg/sub?
*batch_normalization_58/AssignMovingAvg/mulMul.batch_normalization_58/AssignMovingAvg/sub:z:05batch_normalization_58/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_58/AssignMovingAvg/41912269*
_output_shapes	
:?2,
*batch_normalization_58/AssignMovingAvg/mul?
:batch_normalization_58/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_58_assignmovingavg_41912269.batch_normalization_58/AssignMovingAvg/mul:z:06^batch_normalization_58/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_58/AssignMovingAvg/41912269*
_output_shapes
 *
dtype02<
:batch_normalization_58/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_58/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_58/AssignMovingAvg_1/41912275*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_58/AssignMovingAvg_1/decay?
7batch_normalization_58/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_58_assignmovingavg_1_41912275*
_output_shapes	
:?*
dtype029
7batch_normalization_58/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_58/AssignMovingAvg_1/subSub?batch_normalization_58/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_58/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_58/AssignMovingAvg_1/41912275*
_output_shapes	
:?2.
,batch_normalization_58/AssignMovingAvg_1/sub?
,batch_normalization_58/AssignMovingAvg_1/mulMul0batch_normalization_58/AssignMovingAvg_1/sub:z:07batch_normalization_58/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_58/AssignMovingAvg_1/41912275*
_output_shapes	
:?2.
,batch_normalization_58/AssignMovingAvg_1/mul?
<batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_58_assignmovingavg_1_419122750batch_normalization_58/AssignMovingAvg_1/mul:z:08^batch_normalization_58/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_58/AssignMovingAvg_1/41912275*
_output_shapes
 *
dtype02>
<batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_58/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_58/batchnorm/add/y?
$batch_normalization_58/batchnorm/addAddV21batch_normalization_58/moments/Squeeze_1:output:0/batch_normalization_58/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/add?
&batch_normalization_58/batchnorm/RsqrtRsqrt(batch_normalization_58/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_58/batchnorm/Rsqrt?
3batch_normalization_58/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_58_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_58/batchnorm/mul/ReadVariableOp?
$batch_normalization_58/batchnorm/mulMul*batch_normalization_58/batchnorm/Rsqrt:y:0;batch_normalization_58/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/mul?
&batch_normalization_58/batchnorm/mul_1Muldense_47/MatMul:product:0(batch_normalization_58/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_58/batchnorm/mul_1?
&batch_normalization_58/batchnorm/mul_2Mul/batch_normalization_58/moments/Squeeze:output:0(batch_normalization_58/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_58/batchnorm/mul_2?
/batch_normalization_58/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_58_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_58/batchnorm/ReadVariableOp?
$batch_normalization_58/batchnorm/subSub7batch_normalization_58/batchnorm/ReadVariableOp:value:0*batch_normalization_58/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/sub?
&batch_normalization_58/batchnorm/add_1AddV2*batch_normalization_58/batchnorm/mul_1:z:0(batch_normalization_58/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_58/batchnorm/add_1?
re_lu_33/ReluRelu*batch_normalization_58/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
re_lu_33/Reluo
reshape_11/ShapeShapere_lu_33/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape?
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack?
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1?
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2?
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2?
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape?
reshape_11/ReshapeReshapere_lu_33/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_11/Reshape?
5batch_normalization_59/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_59/moments/mean/reduction_indices?
#batch_normalization_59/moments/meanMeanreshape_11/Reshape:output:0>batch_normalization_59/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_59/moments/mean?
+batch_normalization_59/moments/StopGradientStopGradient,batch_normalization_59/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_59/moments/StopGradient?
0batch_normalization_59/moments/SquaredDifferenceSquaredDifferencereshape_11/Reshape:output:04batch_normalization_59/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_59/moments/SquaredDifference?
9batch_normalization_59/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_59/moments/variance/reduction_indices?
'batch_normalization_59/moments/varianceMean4batch_normalization_59/moments/SquaredDifference:z:0Bbatch_normalization_59/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_59/moments/variance?
&batch_normalization_59/moments/SqueezeSqueeze,batch_normalization_59/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_59/moments/Squeeze?
(batch_normalization_59/moments/Squeeze_1Squeeze0batch_normalization_59/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_59/moments/Squeeze_1?
,batch_normalization_59/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_59/AssignMovingAvg/41912311*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_59/AssignMovingAvg/decay?
5batch_normalization_59/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_59_assignmovingavg_41912311*
_output_shapes
:*
dtype027
5batch_normalization_59/AssignMovingAvg/ReadVariableOp?
*batch_normalization_59/AssignMovingAvg/subSub=batch_normalization_59/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_59/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_59/AssignMovingAvg/41912311*
_output_shapes
:2,
*batch_normalization_59/AssignMovingAvg/sub?
*batch_normalization_59/AssignMovingAvg/mulMul.batch_normalization_59/AssignMovingAvg/sub:z:05batch_normalization_59/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_59/AssignMovingAvg/41912311*
_output_shapes
:2,
*batch_normalization_59/AssignMovingAvg/mul?
:batch_normalization_59/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_59_assignmovingavg_41912311.batch_normalization_59/AssignMovingAvg/mul:z:06^batch_normalization_59/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_59/AssignMovingAvg/41912311*
_output_shapes
 *
dtype02<
:batch_normalization_59/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_59/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_59/AssignMovingAvg_1/41912317*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_59/AssignMovingAvg_1/decay?
7batch_normalization_59/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_59_assignmovingavg_1_41912317*
_output_shapes
:*
dtype029
7batch_normalization_59/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_59/AssignMovingAvg_1/subSub?batch_normalization_59/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_59/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_59/AssignMovingAvg_1/41912317*
_output_shapes
:2.
,batch_normalization_59/AssignMovingAvg_1/sub?
,batch_normalization_59/AssignMovingAvg_1/mulMul0batch_normalization_59/AssignMovingAvg_1/sub:z:07batch_normalization_59/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_59/AssignMovingAvg_1/41912317*
_output_shapes
:2.
,batch_normalization_59/AssignMovingAvg_1/mul?
<batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_59_assignmovingavg_1_419123170batch_normalization_59/AssignMovingAvg_1/mul:z:08^batch_normalization_59/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_59/AssignMovingAvg_1/41912317*
_output_shapes
 *
dtype02>
<batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_59/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_59/batchnorm/add/y?
$batch_normalization_59/batchnorm/addAddV21batch_normalization_59/moments/Squeeze_1:output:0/batch_normalization_59/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/add?
&batch_normalization_59/batchnorm/RsqrtRsqrt(batch_normalization_59/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_59/batchnorm/Rsqrt?
3batch_normalization_59/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_59_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_59/batchnorm/mul/ReadVariableOp?
$batch_normalization_59/batchnorm/mulMul*batch_normalization_59/batchnorm/Rsqrt:y:0;batch_normalization_59/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/mul?
&batch_normalization_59/batchnorm/mul_1Mulreshape_11/Reshape:output:0(batch_normalization_59/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_59/batchnorm/mul_1?
&batch_normalization_59/batchnorm/mul_2Mul/batch_normalization_59/moments/Squeeze:output:0(batch_normalization_59/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_59/batchnorm/mul_2?
/batch_normalization_59/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_59_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_59/batchnorm/ReadVariableOp?
$batch_normalization_59/batchnorm/subSub7batch_normalization_59/batchnorm/ReadVariableOp:value:0*batch_normalization_59/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/sub?
&batch_normalization_59/batchnorm/add_1AddV2*batch_normalization_59/batchnorm/mul_1:z:0(batch_normalization_59/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_59/batchnorm/add_1?
conv1d_transpose_10/ShapeShape*batch_normalization_59/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_10/Shape?
'conv1d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv1d_transpose_10/strided_slice/stack?
)conv1d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice/stack_1?
)conv1d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice/stack_2?
!conv1d_transpose_10/strided_sliceStridedSlice"conv1d_transpose_10/Shape:output:00conv1d_transpose_10/strided_slice/stack:output:02conv1d_transpose_10/strided_slice/stack_1:output:02conv1d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv1d_transpose_10/strided_slice?
)conv1d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice_1/stack?
+conv1d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_10/strided_slice_1/stack_1?
+conv1d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_10/strided_slice_1/stack_2?
#conv1d_transpose_10/strided_slice_1StridedSlice"conv1d_transpose_10/Shape:output:02conv1d_transpose_10/strided_slice_1/stack:output:04conv1d_transpose_10/strided_slice_1/stack_1:output:04conv1d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv1d_transpose_10/strided_slice_1x
conv1d_transpose_10/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_10/mul/y?
conv1d_transpose_10/mulMul,conv1d_transpose_10/strided_slice_1:output:0"conv1d_transpose_10/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_10/mul|
conv1d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_10/stack/2?
conv1d_transpose_10/stackPack*conv1d_transpose_10/strided_slice:output:0conv1d_transpose_10/mul:z:0$conv1d_transpose_10/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_10/stack?
3conv1d_transpose_10/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3conv1d_transpose_10/conv1d_transpose/ExpandDims/dim?
/conv1d_transpose_10/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_59/batchnorm/add_1:z:0<conv1d_transpose_10/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????21
/conv1d_transpose_10/conv1d_transpose/ExpandDims?
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpIconv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02B
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?
5conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim?
1conv1d_transpose_10/conv1d_transpose/ExpandDims_1
ExpandDimsHconv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0>conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:23
1conv1d_transpose_10/conv1d_transpose/ExpandDims_1?
8conv1d_transpose_10/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8conv1d_transpose_10/conv1d_transpose/strided_slice/stack?
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1?
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2?
2conv1d_transpose_10/conv1d_transpose/strided_sliceStridedSlice"conv1d_transpose_10/stack:output:0Aconv1d_transpose_10/conv1d_transpose/strided_slice/stack:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice/stack_1:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2conv1d_transpose_10/conv1d_transpose/strided_slice?
:conv1d_transpose_10/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack?
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1?
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2?
4conv1d_transpose_10/conv1d_transpose/strided_slice_1StridedSlice"conv1d_transpose_10/stack:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice_1/stack:output:0Econv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1:output:0Econv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask26
4conv1d_transpose_10/conv1d_transpose/strided_slice_1?
4conv1d_transpose_10/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:26
4conv1d_transpose_10/conv1d_transpose/concat/values_1?
0conv1d_transpose_10/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0conv1d_transpose_10/conv1d_transpose/concat/axis?
+conv1d_transpose_10/conv1d_transpose/concatConcatV2;conv1d_transpose_10/conv1d_transpose/strided_slice:output:0=conv1d_transpose_10/conv1d_transpose/concat/values_1:output:0=conv1d_transpose_10/conv1d_transpose/strided_slice_1:output:09conv1d_transpose_10/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_10/conv1d_transpose/concat?
$conv1d_transpose_10/conv1d_transposeConv2DBackpropInput4conv1d_transpose_10/conv1d_transpose/concat:output:0:conv1d_transpose_10/conv1d_transpose/ExpandDims_1:output:08conv1d_transpose_10/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2&
$conv1d_transpose_10/conv1d_transpose?
,conv1d_transpose_10/conv1d_transpose/SqueezeSqueeze-conv1d_transpose_10/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2.
,conv1d_transpose_10/conv1d_transpose/Squeeze?
5batch_normalization_60/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_60/moments/mean/reduction_indices?
#batch_normalization_60/moments/meanMean5conv1d_transpose_10/conv1d_transpose/Squeeze:output:0>batch_normalization_60/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_60/moments/mean?
+batch_normalization_60/moments/StopGradientStopGradient,batch_normalization_60/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_60/moments/StopGradient?
0batch_normalization_60/moments/SquaredDifferenceSquaredDifference5conv1d_transpose_10/conv1d_transpose/Squeeze:output:04batch_normalization_60/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_60/moments/SquaredDifference?
9batch_normalization_60/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_60/moments/variance/reduction_indices?
'batch_normalization_60/moments/varianceMean4batch_normalization_60/moments/SquaredDifference:z:0Bbatch_normalization_60/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_60/moments/variance?
&batch_normalization_60/moments/SqueezeSqueeze,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_60/moments/Squeeze?
(batch_normalization_60/moments/Squeeze_1Squeeze0batch_normalization_60/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_60/moments/Squeeze_1?
,batch_normalization_60/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_60/AssignMovingAvg/41912375*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_60/AssignMovingAvg/decay?
5batch_normalization_60/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_60_assignmovingavg_41912375*
_output_shapes
:*
dtype027
5batch_normalization_60/AssignMovingAvg/ReadVariableOp?
*batch_normalization_60/AssignMovingAvg/subSub=batch_normalization_60/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_60/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_60/AssignMovingAvg/41912375*
_output_shapes
:2,
*batch_normalization_60/AssignMovingAvg/sub?
*batch_normalization_60/AssignMovingAvg/mulMul.batch_normalization_60/AssignMovingAvg/sub:z:05batch_normalization_60/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_60/AssignMovingAvg/41912375*
_output_shapes
:2,
*batch_normalization_60/AssignMovingAvg/mul?
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_60_assignmovingavg_41912375.batch_normalization_60/AssignMovingAvg/mul:z:06^batch_normalization_60/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_60/AssignMovingAvg/41912375*
_output_shapes
 *
dtype02<
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_60/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_60/AssignMovingAvg_1/41912381*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_60/AssignMovingAvg_1/decay?
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_60_assignmovingavg_1_41912381*
_output_shapes
:*
dtype029
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_60/AssignMovingAvg_1/subSub?batch_normalization_60/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_60/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_60/AssignMovingAvg_1/41912381*
_output_shapes
:2.
,batch_normalization_60/AssignMovingAvg_1/sub?
,batch_normalization_60/AssignMovingAvg_1/mulMul0batch_normalization_60/AssignMovingAvg_1/sub:z:07batch_normalization_60/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_60/AssignMovingAvg_1/41912381*
_output_shapes
:2.
,batch_normalization_60/AssignMovingAvg_1/mul?
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_60_assignmovingavg_1_419123810batch_normalization_60/AssignMovingAvg_1/mul:z:08^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_60/AssignMovingAvg_1/41912381*
_output_shapes
 *
dtype02>
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_60/batchnorm/add/y?
$batch_normalization_60/batchnorm/addAddV21batch_normalization_60/moments/Squeeze_1:output:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/add?
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_60/batchnorm/Rsqrt?
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOp?
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/mul?
&batch_normalization_60/batchnorm/mul_1Mul5conv1d_transpose_10/conv1d_transpose/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_60/batchnorm/mul_1?
&batch_normalization_60/batchnorm/mul_2Mul/batch_normalization_60/moments/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_60/batchnorm/mul_2?
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOp?
$batch_normalization_60/batchnorm/subSub7batch_normalization_60/batchnorm/ReadVariableOp:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/sub?
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_60/batchnorm/add_1?
re_lu_34/ReluRelu*batch_normalization_60/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_34/Relu?
conv1d_transpose_11/ShapeShapere_lu_34/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_11/Shape?
'conv1d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv1d_transpose_11/strided_slice/stack?
)conv1d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice/stack_1?
)conv1d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice/stack_2?
!conv1d_transpose_11/strided_sliceStridedSlice"conv1d_transpose_11/Shape:output:00conv1d_transpose_11/strided_slice/stack:output:02conv1d_transpose_11/strided_slice/stack_1:output:02conv1d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv1d_transpose_11/strided_slice?
)conv1d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice_1/stack?
+conv1d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_11/strided_slice_1/stack_1?
+conv1d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_11/strided_slice_1/stack_2?
#conv1d_transpose_11/strided_slice_1StridedSlice"conv1d_transpose_11/Shape:output:02conv1d_transpose_11/strided_slice_1/stack:output:04conv1d_transpose_11/strided_slice_1/stack_1:output:04conv1d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv1d_transpose_11/strided_slice_1x
conv1d_transpose_11/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_11/mul/y?
conv1d_transpose_11/mulMul,conv1d_transpose_11/strided_slice_1:output:0"conv1d_transpose_11/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_11/mul|
conv1d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_11/stack/2?
conv1d_transpose_11/stackPack*conv1d_transpose_11/strided_slice:output:0conv1d_transpose_11/mul:z:0$conv1d_transpose_11/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_11/stack?
3conv1d_transpose_11/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3conv1d_transpose_11/conv1d_transpose/ExpandDims/dim?
/conv1d_transpose_11/conv1d_transpose/ExpandDims
ExpandDimsre_lu_34/Relu:activations:0<conv1d_transpose_11/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????21
/conv1d_transpose_11/conv1d_transpose/ExpandDims?
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpIconv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02B
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?
5conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim?
1conv1d_transpose_11/conv1d_transpose/ExpandDims_1
ExpandDimsHconv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0>conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:23
1conv1d_transpose_11/conv1d_transpose/ExpandDims_1?
8conv1d_transpose_11/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8conv1d_transpose_11/conv1d_transpose/strided_slice/stack?
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1?
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2?
2conv1d_transpose_11/conv1d_transpose/strided_sliceStridedSlice"conv1d_transpose_11/stack:output:0Aconv1d_transpose_11/conv1d_transpose/strided_slice/stack:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice/stack_1:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2conv1d_transpose_11/conv1d_transpose/strided_slice?
:conv1d_transpose_11/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack?
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1?
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2?
4conv1d_transpose_11/conv1d_transpose/strided_slice_1StridedSlice"conv1d_transpose_11/stack:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice_1/stack:output:0Econv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1:output:0Econv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask26
4conv1d_transpose_11/conv1d_transpose/strided_slice_1?
4conv1d_transpose_11/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:26
4conv1d_transpose_11/conv1d_transpose/concat/values_1?
0conv1d_transpose_11/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0conv1d_transpose_11/conv1d_transpose/concat/axis?
+conv1d_transpose_11/conv1d_transpose/concatConcatV2;conv1d_transpose_11/conv1d_transpose/strided_slice:output:0=conv1d_transpose_11/conv1d_transpose/concat/values_1:output:0=conv1d_transpose_11/conv1d_transpose/strided_slice_1:output:09conv1d_transpose_11/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_11/conv1d_transpose/concat?
$conv1d_transpose_11/conv1d_transposeConv2DBackpropInput4conv1d_transpose_11/conv1d_transpose/concat:output:0:conv1d_transpose_11/conv1d_transpose/ExpandDims_1:output:08conv1d_transpose_11/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2&
$conv1d_transpose_11/conv1d_transpose?
,conv1d_transpose_11/conv1d_transpose/SqueezeSqueeze-conv1d_transpose_11/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2.
,conv1d_transpose_11/conv1d_transpose/Squeeze?
5batch_normalization_61/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_61/moments/mean/reduction_indices?
#batch_normalization_61/moments/meanMean5conv1d_transpose_11/conv1d_transpose/Squeeze:output:0>batch_normalization_61/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_61/moments/mean?
+batch_normalization_61/moments/StopGradientStopGradient,batch_normalization_61/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_61/moments/StopGradient?
0batch_normalization_61/moments/SquaredDifferenceSquaredDifference5conv1d_transpose_11/conv1d_transpose/Squeeze:output:04batch_normalization_61/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_61/moments/SquaredDifference?
9batch_normalization_61/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_61/moments/variance/reduction_indices?
'batch_normalization_61/moments/varianceMean4batch_normalization_61/moments/SquaredDifference:z:0Bbatch_normalization_61/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_61/moments/variance?
&batch_normalization_61/moments/SqueezeSqueeze,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_61/moments/Squeeze?
(batch_normalization_61/moments/Squeeze_1Squeeze0batch_normalization_61/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_61/moments/Squeeze_1?
,batch_normalization_61/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_61/AssignMovingAvg/41912440*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_61/AssignMovingAvg/decay?
5batch_normalization_61/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_61_assignmovingavg_41912440*
_output_shapes
:*
dtype027
5batch_normalization_61/AssignMovingAvg/ReadVariableOp?
*batch_normalization_61/AssignMovingAvg/subSub=batch_normalization_61/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_61/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_61/AssignMovingAvg/41912440*
_output_shapes
:2,
*batch_normalization_61/AssignMovingAvg/sub?
*batch_normalization_61/AssignMovingAvg/mulMul.batch_normalization_61/AssignMovingAvg/sub:z:05batch_normalization_61/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_61/AssignMovingAvg/41912440*
_output_shapes
:2,
*batch_normalization_61/AssignMovingAvg/mul?
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_61_assignmovingavg_41912440.batch_normalization_61/AssignMovingAvg/mul:z:06^batch_normalization_61/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_61/AssignMovingAvg/41912440*
_output_shapes
 *
dtype02<
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_61/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_61/AssignMovingAvg_1/41912446*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_61/AssignMovingAvg_1/decay?
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_61_assignmovingavg_1_41912446*
_output_shapes
:*
dtype029
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_61/AssignMovingAvg_1/subSub?batch_normalization_61/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_61/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_61/AssignMovingAvg_1/41912446*
_output_shapes
:2.
,batch_normalization_61/AssignMovingAvg_1/sub?
,batch_normalization_61/AssignMovingAvg_1/mulMul0batch_normalization_61/AssignMovingAvg_1/sub:z:07batch_normalization_61/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_61/AssignMovingAvg_1/41912446*
_output_shapes
:2.
,batch_normalization_61/AssignMovingAvg_1/mul?
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_61_assignmovingavg_1_419124460batch_normalization_61/AssignMovingAvg_1/mul:z:08^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_61/AssignMovingAvg_1/41912446*
_output_shapes
 *
dtype02>
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_61/batchnorm/add/y?
$batch_normalization_61/batchnorm/addAddV21batch_normalization_61/moments/Squeeze_1:output:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/add?
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_61/batchnorm/Rsqrt?
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOp?
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/mul?
&batch_normalization_61/batchnorm/mul_1Mul5conv1d_transpose_11/conv1d_transpose/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_61/batchnorm/mul_1?
&batch_normalization_61/batchnorm/mul_2Mul/batch_normalization_61/moments/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_61/batchnorm/mul_2?
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp?
$batch_normalization_61/batchnorm/subSub7batch_normalization_61/batchnorm/ReadVariableOp:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/sub?
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_61/batchnorm/add_1?
re_lu_35/ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_35/Reluu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_11/Const?
flatten_11/ReshapeReshapere_lu_35/Relu:activations:0flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_11/Reshape?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMulflatten_11/Reshape:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_48/MatMuls
dense_48/TanhTanhdense_48/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_48/Tanh?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMulflatten_11/Reshape:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_49/MatMuls
dense_49/TanhTanhdense_49/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_49/Tanhm
lambda_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_5/truediv/y?
lambda_5/truedivRealDivdense_49/Tanh:y:0lambda_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/truedivk
lambda_5/ExpExplambda_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_5/Expz
lambda_5/mulMuldense_48/Tanh:y:0lambda_5/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_5/mul`
lambda_5/ShapeShapelambda_5/mul:z:0*
T0*
_output_shapes
:2
lambda_5/Shape
lambda_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_5/random_normal/mean?
lambda_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda_5/random_normal/stddev?
+lambda_5/random_normal/RandomStandardNormalRandomStandardNormallambda_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??b2-
+lambda_5/random_normal/RandomStandardNormal?
lambda_5/random_normal/mulMul4lambda_5/random_normal/RandomStandardNormal:output:0&lambda_5/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/random_normal/mul?
lambda_5/random_normalAddlambda_5/random_normal/mul:z:0$lambda_5/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/random_normal?
lambda_5/addAddV2dense_48/Tanh:y:0lambda_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
lambda_5/add?
5batch_normalization_62/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_62/moments/mean/reduction_indices?
#batch_normalization_62/moments/meanMeanlambda_5/add:z:0>batch_normalization_62/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_62/moments/mean?
+batch_normalization_62/moments/StopGradientStopGradient,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_62/moments/StopGradient?
0batch_normalization_62/moments/SquaredDifferenceSquaredDifferencelambda_5/add:z:04batch_normalization_62/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_62/moments/SquaredDifference?
9batch_normalization_62/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_62/moments/variance/reduction_indices?
'batch_normalization_62/moments/varianceMean4batch_normalization_62/moments/SquaredDifference:z:0Bbatch_normalization_62/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_62/moments/variance?
&batch_normalization_62/moments/SqueezeSqueeze,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_62/moments/Squeeze?
(batch_normalization_62/moments/Squeeze_1Squeeze0batch_normalization_62/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_62/moments/Squeeze_1?
,batch_normalization_62/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_62/AssignMovingAvg/41912494*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_62/AssignMovingAvg/decay?
5batch_normalization_62/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_62_assignmovingavg_41912494*
_output_shapes
:*
dtype027
5batch_normalization_62/AssignMovingAvg/ReadVariableOp?
*batch_normalization_62/AssignMovingAvg/subSub=batch_normalization_62/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_62/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_62/AssignMovingAvg/41912494*
_output_shapes
:2,
*batch_normalization_62/AssignMovingAvg/sub?
*batch_normalization_62/AssignMovingAvg/mulMul.batch_normalization_62/AssignMovingAvg/sub:z:05batch_normalization_62/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_62/AssignMovingAvg/41912494*
_output_shapes
:2,
*batch_normalization_62/AssignMovingAvg/mul?
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_62_assignmovingavg_41912494.batch_normalization_62/AssignMovingAvg/mul:z:06^batch_normalization_62/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_62/AssignMovingAvg/41912494*
_output_shapes
 *
dtype02<
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_62/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_62/AssignMovingAvg_1/41912500*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_62/AssignMovingAvg_1/decay?
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_62_assignmovingavg_1_41912500*
_output_shapes
:*
dtype029
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_62/AssignMovingAvg_1/subSub?batch_normalization_62/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_62/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_62/AssignMovingAvg_1/41912500*
_output_shapes
:2.
,batch_normalization_62/AssignMovingAvg_1/sub?
,batch_normalization_62/AssignMovingAvg_1/mulMul0batch_normalization_62/AssignMovingAvg_1/sub:z:07batch_normalization_62/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_62/AssignMovingAvg_1/41912500*
_output_shapes
:2.
,batch_normalization_62/AssignMovingAvg_1/mul?
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_62_assignmovingavg_1_419125000batch_normalization_62/AssignMovingAvg_1/mul:z:08^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_62/AssignMovingAvg_1/41912500*
_output_shapes
 *
dtype02>
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_62/batchnorm/add/y?
$batch_normalization_62/batchnorm/addAddV21batch_normalization_62/moments/Squeeze_1:output:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/add?
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_62/batchnorm/Rsqrt?
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOp?
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/mul?
&batch_normalization_62/batchnorm/mul_1Mullambda_5/add:z:0(batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_62/batchnorm/mul_1?
&batch_normalization_62/batchnorm/mul_2Mul/batch_normalization_62/moments/Squeeze:output:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_62/batchnorm/mul_2?
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOp?
$batch_normalization_62/batchnorm/subSub7batch_normalization_62/batchnorm/ReadVariableOp:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/sub?
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_62/batchnorm/add_1?
IdentityIdentity*batch_normalization_62/batchnorm/add_1:z:0;^batch_normalization_58/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_58/AssignMovingAvg/ReadVariableOp=^batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_58/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_58/batchnorm/ReadVariableOp4^batch_normalization_58/batchnorm/mul/ReadVariableOp;^batch_normalization_59/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_59/AssignMovingAvg/ReadVariableOp=^batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_59/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_59/batchnorm/ReadVariableOp4^batch_normalization_59/batchnorm/mul/ReadVariableOp;^batch_normalization_60/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_60/AssignMovingAvg/ReadVariableOp=^batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_60/batchnorm/ReadVariableOp4^batch_normalization_60/batchnorm/mul/ReadVariableOp;^batch_normalization_61/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_61/AssignMovingAvg/ReadVariableOp=^batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp4^batch_normalization_61/batchnorm/mul/ReadVariableOp;^batch_normalization_62/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_62/AssignMovingAvg/ReadVariableOp=^batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp4^batch_normalization_62/batchnorm/mul/ReadVariableOpA^conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpA^conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_47/MatMul/ReadVariableOp^dense_48/MatMul/ReadVariableOp^dense_49/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2x
:batch_normalization_58/AssignMovingAvg/AssignSubVariableOp:batch_normalization_58/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_58/AssignMovingAvg/ReadVariableOp5batch_normalization_58/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_58/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_58/AssignMovingAvg_1/ReadVariableOp7batch_normalization_58/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_58/batchnorm/ReadVariableOp/batch_normalization_58/batchnorm/ReadVariableOp2j
3batch_normalization_58/batchnorm/mul/ReadVariableOp3batch_normalization_58/batchnorm/mul/ReadVariableOp2x
:batch_normalization_59/AssignMovingAvg/AssignSubVariableOp:batch_normalization_59/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_59/AssignMovingAvg/ReadVariableOp5batch_normalization_59/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_59/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_59/AssignMovingAvg_1/ReadVariableOp7batch_normalization_59/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_59/batchnorm/ReadVariableOp/batch_normalization_59/batchnorm/ReadVariableOp2j
3batch_normalization_59/batchnorm/mul/ReadVariableOp3batch_normalization_59/batchnorm/mul/ReadVariableOp2x
:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp:batch_normalization_60/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_60/AssignMovingAvg/ReadVariableOp5batch_normalization_60/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_60/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2x
:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp:batch_normalization_61/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_61/AssignMovingAvg/ReadVariableOp5batch_normalization_61/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_61/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2x
:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp:batch_normalization_62/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_62/AssignMovingAvg/ReadVariableOp5batch_normalization_62/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_62/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2?
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913432

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_62_layer_call_fn_41913445

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_419114052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41911113

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
q
+__inference_dense_49_layer_call_fn_41913332

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_419117822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
|
6__inference_conv1d_transpose_10_layer_call_fn_41910984

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_419109762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
G
+__inference_re_lu_33_layer_call_fn_41912919

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_33_layer_call_and_return_conditional_losses_419115122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_60_layer_call_fn_41913170

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
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_419110802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
d
H__inference_reshape_11_layer_call_and_return_conditional_losses_41911533

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
q
+__inference_dense_48_layer_call_fn_41913317

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_419117622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_11_layer_call_and_return_conditional_losses_41911746

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
?
s
F__inference_lambda_5_layer_call_and_return_conditional_losses_41911810

inputs
inputs_1
identity?[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpT
mulMulinputsExp:y:0*
T0*'
_output_shapes
:?????????2
mulE
ShapeShapemul:z:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??.2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal`
addAddV2inputsrandom_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913157

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
*__inference_Encoder_layer_call_fn_41912758

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
identity??StatefulPartitionedCall?
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
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Encoder_layer_call_and_return_conditional_losses_419120202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?n
?
$__inference__traced_restore_41913641
file_prefix$
 assignvariableop_dense_47_kernel3
/assignvariableop_1_batch_normalization_58_gamma2
.assignvariableop_2_batch_normalization_58_beta9
5assignvariableop_3_batch_normalization_58_moving_mean=
9assignvariableop_4_batch_normalization_58_moving_variance3
/assignvariableop_5_batch_normalization_59_gamma2
.assignvariableop_6_batch_normalization_59_beta9
5assignvariableop_7_batch_normalization_59_moving_mean=
9assignvariableop_8_batch_normalization_59_moving_variance1
-assignvariableop_9_conv1d_transpose_10_kernel4
0assignvariableop_10_batch_normalization_60_gamma3
/assignvariableop_11_batch_normalization_60_beta:
6assignvariableop_12_batch_normalization_60_moving_mean>
:assignvariableop_13_batch_normalization_60_moving_variance2
.assignvariableop_14_conv1d_transpose_11_kernel4
0assignvariableop_15_batch_normalization_61_gamma3
/assignvariableop_16_batch_normalization_61_beta:
6assignvariableop_17_batch_normalization_61_moving_mean>
:assignvariableop_18_batch_normalization_61_moving_variance'
#assignvariableop_19_dense_48_kernel'
#assignvariableop_20_dense_49_kernel4
0assignvariableop_21_batch_normalization_62_gamma3
/assignvariableop_22_batch_normalization_62_beta:
6assignvariableop_23_batch_normalization_62_moving_mean>
:assignvariableop_24_batch_normalization_62_moving_variance
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
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

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_47_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_58_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_58_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_58_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_58_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp/assignvariableop_5_batch_normalization_59_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_59_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp5assignvariableop_7_batch_normalization_59_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_batch_normalization_59_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_conv1d_transpose_10_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_60_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_60_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_60_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_60_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_conv1d_transpose_11_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_61_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_61_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_batch_normalization_61_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_batch_normalization_61_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_48_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_49_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_62_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_62_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_batch_normalization_62_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_batch_normalization_62_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
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
?
?
*__inference_Encoder_layer_call_fn_41912813

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
identity??StatefulPartitionedCall?
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
:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Encoder_layer_call_and_return_conditional_losses_419121452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_48_layer_call_and_return_conditional_losses_41911762

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_62_layer_call_fn_41913458

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_419114382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41910788

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
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
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_11_layer_call_fn_41913302

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
GPU 2J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_419117462
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
?1
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41910895

inputs
assignmovingavg_41910870
assignmovingavg_1_41910876)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41910870*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41910870*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41910870*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41910870*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41910870AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41910870*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41910876*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41910876*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41910876*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41910876*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41910876AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41910876*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?1
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912973

inputs
assignmovingavg_41912948
assignmovingavg_1_41912954)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41912948*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41912948*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41912948*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41912948*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41912948AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41912948*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41912954*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41912954*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41912954*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41912954*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41912954AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41912954*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_41910659
input_193
/encoder_dense_47_matmul_readvariableop_resourceD
@encoder_batch_normalization_58_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_58_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_58_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_58_batchnorm_readvariableop_2_resourceD
@encoder_batch_normalization_59_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_59_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_59_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_59_batchnorm_readvariableop_2_resourceU
Qencoder_conv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resourceD
@encoder_batch_normalization_60_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_60_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_60_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_60_batchnorm_readvariableop_2_resourceU
Qencoder_conv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resourceD
@encoder_batch_normalization_61_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_61_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_61_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_61_batchnorm_readvariableop_2_resource3
/encoder_dense_48_matmul_readvariableop_resource3
/encoder_dense_49_matmul_readvariableop_resourceD
@encoder_batch_normalization_62_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_62_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_62_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_62_batchnorm_readvariableop_2_resource
identity??7Encoder/batch_normalization_58/batchnorm/ReadVariableOp?9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_59/batchnorm/ReadVariableOp?9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_60/batchnorm/ReadVariableOp?9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_61/batchnorm/ReadVariableOp?9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_62/batchnorm/ReadVariableOp?9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOp?HEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?HEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?&Encoder/dense_47/MatMul/ReadVariableOp?&Encoder/dense_48/MatMul/ReadVariableOp?&Encoder/dense_49/MatMul/ReadVariableOp?
&Encoder/dense_47/MatMul/ReadVariableOpReadVariableOp/encoder_dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Encoder/dense_47/MatMul/ReadVariableOp?
Encoder/dense_47/MatMulMatMulinput_19.Encoder/dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Encoder/dense_47/MatMul?
7Encoder/batch_normalization_58/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_58_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7Encoder/batch_normalization_58/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_58/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_58/batchnorm/add/y?
,Encoder/batch_normalization_58/batchnorm/addAddV2?Encoder/batch_normalization_58/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_58/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,Encoder/batch_normalization_58/batchnorm/add?
.Encoder/batch_normalization_58/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_58/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.Encoder/batch_normalization_58/batchnorm/Rsqrt?
;Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_58_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_58/batchnorm/mulMul2Encoder/batch_normalization_58/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_58/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,Encoder/batch_normalization_58/batchnorm/mul?
.Encoder/batch_normalization_58/batchnorm/mul_1Mul!Encoder/dense_47/MatMul:product:00Encoder/batch_normalization_58/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.Encoder/batch_normalization_58/batchnorm/mul_1?
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_58_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_58/batchnorm/mul_2MulAEncoder/batch_normalization_58/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_58/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.Encoder/batch_normalization_58/batchnorm/mul_2?
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_58_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_58/batchnorm/subSubAEncoder/batch_normalization_58/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_58/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,Encoder/batch_normalization_58/batchnorm/sub?
.Encoder/batch_normalization_58/batchnorm/add_1AddV22Encoder/batch_normalization_58/batchnorm/mul_1:z:00Encoder/batch_normalization_58/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.Encoder/batch_normalization_58/batchnorm/add_1?
Encoder/re_lu_33/ReluRelu2Encoder/batch_normalization_58/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
Encoder/re_lu_33/Relu?
Encoder/reshape_11/ShapeShape#Encoder/re_lu_33/Relu:activations:0*
T0*
_output_shapes
:2
Encoder/reshape_11/Shape?
&Encoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Encoder/reshape_11/strided_slice/stack?
(Encoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/reshape_11/strided_slice/stack_1?
(Encoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/reshape_11/strided_slice/stack_2?
 Encoder/reshape_11/strided_sliceStridedSlice!Encoder/reshape_11/Shape:output:0/Encoder/reshape_11/strided_slice/stack:output:01Encoder/reshape_11/strided_slice/stack_1:output:01Encoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/reshape_11/strided_slice?
"Encoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"Encoder/reshape_11/Reshape/shape/1?
"Encoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Encoder/reshape_11/Reshape/shape/2?
 Encoder/reshape_11/Reshape/shapePack)Encoder/reshape_11/strided_slice:output:0+Encoder/reshape_11/Reshape/shape/1:output:0+Encoder/reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 Encoder/reshape_11/Reshape/shape?
Encoder/reshape_11/ReshapeReshape#Encoder/re_lu_33/Relu:activations:0)Encoder/reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
Encoder/reshape_11/Reshape?
7Encoder/batch_normalization_59/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_59_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_59/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_59/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_59/batchnorm/add/y?
,Encoder/batch_normalization_59/batchnorm/addAddV2?Encoder/batch_normalization_59/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_59/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_59/batchnorm/add?
.Encoder/batch_normalization_59/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_59/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_59/batchnorm/Rsqrt?
;Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_59_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_59/batchnorm/mulMul2Encoder/batch_normalization_59/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_59/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_59/batchnorm/mul?
.Encoder/batch_normalization_59/batchnorm/mul_1Mul#Encoder/reshape_11/Reshape:output:00Encoder/batch_normalization_59/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_59/batchnorm/mul_1?
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_59_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_59/batchnorm/mul_2MulAEncoder/batch_normalization_59/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_59/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_59/batchnorm/mul_2?
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_59_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_59/batchnorm/subSubAEncoder/batch_normalization_59/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_59/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_59/batchnorm/sub?
.Encoder/batch_normalization_59/batchnorm/add_1AddV22Encoder/batch_normalization_59/batchnorm/mul_1:z:00Encoder/batch_normalization_59/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_59/batchnorm/add_1?
!Encoder/conv1d_transpose_10/ShapeShape2Encoder/batch_normalization_59/batchnorm/add_1:z:0*
T0*
_output_shapes
:2#
!Encoder/conv1d_transpose_10/Shape?
/Encoder/conv1d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/Encoder/conv1d_transpose_10/strided_slice/stack?
1Encoder/conv1d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_10/strided_slice/stack_1?
1Encoder/conv1d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_10/strided_slice/stack_2?
)Encoder/conv1d_transpose_10/strided_sliceStridedSlice*Encoder/conv1d_transpose_10/Shape:output:08Encoder/conv1d_transpose_10/strided_slice/stack:output:0:Encoder/conv1d_transpose_10/strided_slice/stack_1:output:0:Encoder/conv1d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)Encoder/conv1d_transpose_10/strided_slice?
1Encoder/conv1d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_10/strided_slice_1/stack?
3Encoder/conv1d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/conv1d_transpose_10/strided_slice_1/stack_1?
3Encoder/conv1d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/conv1d_transpose_10/strided_slice_1/stack_2?
+Encoder/conv1d_transpose_10/strided_slice_1StridedSlice*Encoder/conv1d_transpose_10/Shape:output:0:Encoder/conv1d_transpose_10/strided_slice_1/stack:output:0<Encoder/conv1d_transpose_10/strided_slice_1/stack_1:output:0<Encoder/conv1d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/conv1d_transpose_10/strided_slice_1?
!Encoder/conv1d_transpose_10/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!Encoder/conv1d_transpose_10/mul/y?
Encoder/conv1d_transpose_10/mulMul4Encoder/conv1d_transpose_10/strided_slice_1:output:0*Encoder/conv1d_transpose_10/mul/y:output:0*
T0*
_output_shapes
: 2!
Encoder/conv1d_transpose_10/mul?
#Encoder/conv1d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/conv1d_transpose_10/stack/2?
!Encoder/conv1d_transpose_10/stackPack2Encoder/conv1d_transpose_10/strided_slice:output:0#Encoder/conv1d_transpose_10/mul:z:0,Encoder/conv1d_transpose_10/stack/2:output:0*
N*
T0*
_output_shapes
:2#
!Encoder/conv1d_transpose_10/stack?
;Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims/dim?
7Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims
ExpandDims2Encoder/batch_normalization_59/batchnorm/add_1:z:0DEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????29
7Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims?
HEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpQencoder_conv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02J
HEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?
=Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim?
9Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1
ExpandDimsPEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0FEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2;
9Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1?
@Encoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@Encoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack?
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1?
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2?
:Encoder/conv1d_transpose_10/conv1d_transpose/strided_sliceStridedSlice*Encoder/conv1d_transpose_10/stack:output:0IEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack:output:0KEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1:output:0KEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2<
:Encoder/conv1d_transpose_10/conv1d_transpose/strided_slice?
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack?
DEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
DEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1?
DEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2?
<Encoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1StridedSlice*Encoder/conv1d_transpose_10/stack:output:0KEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack:output:0MEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1:output:0MEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2>
<Encoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1?
<Encoder/conv1d_transpose_10/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<Encoder/conv1d_transpose_10/conv1d_transpose/concat/values_1?
8Encoder/conv1d_transpose_10/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8Encoder/conv1d_transpose_10/conv1d_transpose/concat/axis?
3Encoder/conv1d_transpose_10/conv1d_transpose/concatConcatV2CEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice:output:0EEncoder/conv1d_transpose_10/conv1d_transpose/concat/values_1:output:0EEncoder/conv1d_transpose_10/conv1d_transpose/strided_slice_1:output:0AEncoder/conv1d_transpose_10/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:25
3Encoder/conv1d_transpose_10/conv1d_transpose/concat?
,Encoder/conv1d_transpose_10/conv1d_transposeConv2DBackpropInput<Encoder/conv1d_transpose_10/conv1d_transpose/concat:output:0BEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1:output:0@Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2.
,Encoder/conv1d_transpose_10/conv1d_transpose?
4Encoder/conv1d_transpose_10/conv1d_transpose/SqueezeSqueeze5Encoder/conv1d_transpose_10/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
26
4Encoder/conv1d_transpose_10/conv1d_transpose/Squeeze?
7Encoder/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_60/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_60/batchnorm/add/y?
,Encoder/batch_normalization_60/batchnorm/addAddV2?Encoder/batch_normalization_60/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_60/batchnorm/add?
.Encoder/batch_normalization_60/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_60/batchnorm/Rsqrt?
;Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_60/batchnorm/mulMul2Encoder/batch_normalization_60/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_60/batchnorm/mul?
.Encoder/batch_normalization_60/batchnorm/mul_1Mul=Encoder/conv1d_transpose_10/conv1d_transpose/Squeeze:output:00Encoder/batch_normalization_60/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_60/batchnorm/mul_1?
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_60/batchnorm/mul_2MulAEncoder/batch_normalization_60/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_60/batchnorm/mul_2?
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_60/batchnorm/subSubAEncoder/batch_normalization_60/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_60/batchnorm/sub?
.Encoder/batch_normalization_60/batchnorm/add_1AddV22Encoder/batch_normalization_60/batchnorm/mul_1:z:00Encoder/batch_normalization_60/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_60/batchnorm/add_1?
Encoder/re_lu_34/ReluRelu2Encoder/batch_normalization_60/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
Encoder/re_lu_34/Relu?
!Encoder/conv1d_transpose_11/ShapeShape#Encoder/re_lu_34/Relu:activations:0*
T0*
_output_shapes
:2#
!Encoder/conv1d_transpose_11/Shape?
/Encoder/conv1d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/Encoder/conv1d_transpose_11/strided_slice/stack?
1Encoder/conv1d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_11/strided_slice/stack_1?
1Encoder/conv1d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_11/strided_slice/stack_2?
)Encoder/conv1d_transpose_11/strided_sliceStridedSlice*Encoder/conv1d_transpose_11/Shape:output:08Encoder/conv1d_transpose_11/strided_slice/stack:output:0:Encoder/conv1d_transpose_11/strided_slice/stack_1:output:0:Encoder/conv1d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)Encoder/conv1d_transpose_11/strided_slice?
1Encoder/conv1d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1Encoder/conv1d_transpose_11/strided_slice_1/stack?
3Encoder/conv1d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/conv1d_transpose_11/strided_slice_1/stack_1?
3Encoder/conv1d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/conv1d_transpose_11/strided_slice_1/stack_2?
+Encoder/conv1d_transpose_11/strided_slice_1StridedSlice*Encoder/conv1d_transpose_11/Shape:output:0:Encoder/conv1d_transpose_11/strided_slice_1/stack:output:0<Encoder/conv1d_transpose_11/strided_slice_1/stack_1:output:0<Encoder/conv1d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/conv1d_transpose_11/strided_slice_1?
!Encoder/conv1d_transpose_11/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!Encoder/conv1d_transpose_11/mul/y?
Encoder/conv1d_transpose_11/mulMul4Encoder/conv1d_transpose_11/strided_slice_1:output:0*Encoder/conv1d_transpose_11/mul/y:output:0*
T0*
_output_shapes
: 2!
Encoder/conv1d_transpose_11/mul?
#Encoder/conv1d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/conv1d_transpose_11/stack/2?
!Encoder/conv1d_transpose_11/stackPack2Encoder/conv1d_transpose_11/strided_slice:output:0#Encoder/conv1d_transpose_11/mul:z:0,Encoder/conv1d_transpose_11/stack/2:output:0*
N*
T0*
_output_shapes
:2#
!Encoder/conv1d_transpose_11/stack?
;Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims/dim?
7Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims
ExpandDims#Encoder/re_lu_34/Relu:activations:0DEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????29
7Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims?
HEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpQencoder_conv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02J
HEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?
=Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim?
9Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1
ExpandDimsPEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0FEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2;
9Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1?
@Encoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@Encoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack?
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1?
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2?
:Encoder/conv1d_transpose_11/conv1d_transpose/strided_sliceStridedSlice*Encoder/conv1d_transpose_11/stack:output:0IEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack:output:0KEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1:output:0KEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2<
:Encoder/conv1d_transpose_11/conv1d_transpose/strided_slice?
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
BEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack?
DEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
DEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1?
DEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2?
<Encoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1StridedSlice*Encoder/conv1d_transpose_11/stack:output:0KEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack:output:0MEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1:output:0MEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2>
<Encoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1?
<Encoder/conv1d_transpose_11/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<Encoder/conv1d_transpose_11/conv1d_transpose/concat/values_1?
8Encoder/conv1d_transpose_11/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8Encoder/conv1d_transpose_11/conv1d_transpose/concat/axis?
3Encoder/conv1d_transpose_11/conv1d_transpose/concatConcatV2CEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice:output:0EEncoder/conv1d_transpose_11/conv1d_transpose/concat/values_1:output:0EEncoder/conv1d_transpose_11/conv1d_transpose/strided_slice_1:output:0AEncoder/conv1d_transpose_11/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:25
3Encoder/conv1d_transpose_11/conv1d_transpose/concat?
,Encoder/conv1d_transpose_11/conv1d_transposeConv2DBackpropInput<Encoder/conv1d_transpose_11/conv1d_transpose/concat:output:0BEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1:output:0@Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2.
,Encoder/conv1d_transpose_11/conv1d_transpose?
4Encoder/conv1d_transpose_11/conv1d_transpose/SqueezeSqueeze5Encoder/conv1d_transpose_11/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
26
4Encoder/conv1d_transpose_11/conv1d_transpose/Squeeze?
7Encoder/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_61/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_61/batchnorm/add/y?
,Encoder/batch_normalization_61/batchnorm/addAddV2?Encoder/batch_normalization_61/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_61/batchnorm/add?
.Encoder/batch_normalization_61/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_61/batchnorm/Rsqrt?
;Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_61/batchnorm/mulMul2Encoder/batch_normalization_61/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_61/batchnorm/mul?
.Encoder/batch_normalization_61/batchnorm/mul_1Mul=Encoder/conv1d_transpose_11/conv1d_transpose/Squeeze:output:00Encoder/batch_normalization_61/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_61/batchnorm/mul_1?
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_61/batchnorm/mul_2MulAEncoder/batch_normalization_61/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_61/batchnorm/mul_2?
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_61/batchnorm/subSubAEncoder/batch_normalization_61/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_61/batchnorm/sub?
.Encoder/batch_normalization_61/batchnorm/add_1AddV22Encoder/batch_normalization_61/batchnorm/mul_1:z:00Encoder/batch_normalization_61/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_61/batchnorm/add_1?
Encoder/re_lu_35/ReluRelu2Encoder/batch_normalization_61/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
Encoder/re_lu_35/Relu?
Encoder/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Encoder/flatten_11/Const?
Encoder/flatten_11/ReshapeReshape#Encoder/re_lu_35/Relu:activations:0!Encoder/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????@2
Encoder/flatten_11/Reshape?
&Encoder/dense_48/MatMul/ReadVariableOpReadVariableOp/encoder_dense_48_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&Encoder/dense_48/MatMul/ReadVariableOp?
Encoder/dense_48/MatMulMatMul#Encoder/flatten_11/Reshape:output:0.Encoder/dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_48/MatMul?
Encoder/dense_48/TanhTanh!Encoder/dense_48/MatMul:product:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_48/Tanh?
&Encoder/dense_49/MatMul/ReadVariableOpReadVariableOp/encoder_dense_49_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&Encoder/dense_49/MatMul/ReadVariableOp?
Encoder/dense_49/MatMulMatMul#Encoder/flatten_11/Reshape:output:0.Encoder/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_49/MatMul?
Encoder/dense_49/TanhTanh!Encoder/dense_49/MatMul:product:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_49/Tanh}
Encoder/lambda_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Encoder/lambda_5/truediv/y?
Encoder/lambda_5/truedivRealDivEncoder/dense_49/Tanh:y:0#Encoder/lambda_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_5/truediv?
Encoder/lambda_5/ExpExpEncoder/lambda_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_5/Exp?
Encoder/lambda_5/mulMulEncoder/dense_48/Tanh:y:0Encoder/lambda_5/Exp:y:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_5/mulx
Encoder/lambda_5/ShapeShapeEncoder/lambda_5/mul:z:0*
T0*
_output_shapes
:2
Encoder/lambda_5/Shape?
#Encoder/lambda_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#Encoder/lambda_5/random_normal/mean?
%Encoder/lambda_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%Encoder/lambda_5/random_normal/stddev?
3Encoder/lambda_5/random_normal/RandomStandardNormalRandomStandardNormalEncoder/lambda_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???25
3Encoder/lambda_5/random_normal/RandomStandardNormal?
"Encoder/lambda_5/random_normal/mulMul<Encoder/lambda_5/random_normal/RandomStandardNormal:output:0.Encoder/lambda_5/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2$
"Encoder/lambda_5/random_normal/mul?
Encoder/lambda_5/random_normalAdd&Encoder/lambda_5/random_normal/mul:z:0,Encoder/lambda_5/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2 
Encoder/lambda_5/random_normal?
Encoder/lambda_5/addAddV2Encoder/dense_48/Tanh:y:0"Encoder/lambda_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_5/add?
7Encoder/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_62/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_62/batchnorm/add/y?
,Encoder/batch_normalization_62/batchnorm/addAddV2?Encoder/batch_normalization_62/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_62/batchnorm/add?
.Encoder/batch_normalization_62/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_62/batchnorm/Rsqrt?
;Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_62/batchnorm/mulMul2Encoder/batch_normalization_62/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_62/batchnorm/mul?
.Encoder/batch_normalization_62/batchnorm/mul_1MulEncoder/lambda_5/add:z:00Encoder/batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????20
.Encoder/batch_normalization_62/batchnorm/mul_1?
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_62/batchnorm/mul_2MulAEncoder/batch_normalization_62/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_62/batchnorm/mul_2?
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_62/batchnorm/subSubAEncoder/batch_normalization_62/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_62/batchnorm/sub?
.Encoder/batch_normalization_62/batchnorm/add_1AddV22Encoder/batch_normalization_62/batchnorm/mul_1:z:00Encoder/batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????20
.Encoder/batch_normalization_62/batchnorm/add_1?
IdentityIdentity2Encoder/batch_normalization_62/batchnorm/add_1:z:08^Encoder/batch_normalization_58/batchnorm/ReadVariableOp:^Encoder/batch_normalization_58/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_58/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_59/batchnorm/ReadVariableOp:^Encoder/batch_normalization_59/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_59/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_60/batchnorm/ReadVariableOp:^Encoder/batch_normalization_60/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_60/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_61/batchnorm/ReadVariableOp:^Encoder/batch_normalization_61/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_61/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_62/batchnorm/ReadVariableOp:^Encoder/batch_normalization_62/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_62/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOpI^Encoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpI^Encoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp'^Encoder/dense_47/MatMul/ReadVariableOp'^Encoder/dense_48/MatMul/ReadVariableOp'^Encoder/dense_49/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2r
7Encoder/batch_normalization_58/batchnorm/ReadVariableOp7Encoder/batch_normalization_58/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_19Encoder/batch_normalization_58/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_58/batchnorm/ReadVariableOp_29Encoder/batch_normalization_58/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_58/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_59/batchnorm/ReadVariableOp7Encoder/batch_normalization_59/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_19Encoder/batch_normalization_59/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_59/batchnorm/ReadVariableOp_29Encoder/batch_normalization_59/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_59/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_60/batchnorm/ReadVariableOp7Encoder/batch_normalization_60/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_19Encoder/batch_normalization_60/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_60/batchnorm/ReadVariableOp_29Encoder/batch_normalization_60/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_60/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_61/batchnorm/ReadVariableOp7Encoder/batch_normalization_61/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_19Encoder/batch_normalization_61/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_61/batchnorm/ReadVariableOp_29Encoder/batch_normalization_61/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_61/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_62/batchnorm/ReadVariableOp7Encoder/batch_normalization_62/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_19Encoder/batch_normalization_62/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_62/batchnorm/ReadVariableOp_29Encoder/batch_normalization_62/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_62/batchnorm/mul/ReadVariableOp2?
HEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpHEncoder/conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
HEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOpHEncoder/conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp2P
&Encoder/dense_47/MatMul/ReadVariableOp&Encoder/dense_47/MatMul/ReadVariableOp2P
&Encoder/dense_48/MatMul/ReadVariableOp&Encoder/dense_48/MatMul/ReadVariableOp2P
&Encoder/dense_49/MatMul/ReadVariableOp&Encoder/dense_49/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912993

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41910928

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
q
+__inference_dense_47_layer_call_fn_41912827

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_419114602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912020

inputs
dense_47_41911953#
batch_normalization_58_41911956#
batch_normalization_58_41911958#
batch_normalization_58_41911960#
batch_normalization_58_41911962#
batch_normalization_59_41911967#
batch_normalization_59_41911969#
batch_normalization_59_41911971#
batch_normalization_59_41911973 
conv1d_transpose_10_41911976#
batch_normalization_60_41911979#
batch_normalization_60_41911981#
batch_normalization_60_41911983#
batch_normalization_60_41911985 
conv1d_transpose_11_41911989#
batch_normalization_61_41911992#
batch_normalization_61_41911994#
batch_normalization_61_41911996#
batch_normalization_61_41911998
dense_48_41912003
dense_49_41912006#
batch_normalization_62_41912010#
batch_normalization_62_41912012#
batch_normalization_62_41912014#
batch_normalization_62_41912016
identity??.batch_normalization_58/StatefulPartitionedCall?.batch_normalization_59/StatefulPartitionedCall?.batch_normalization_60/StatefulPartitionedCall?.batch_normalization_61/StatefulPartitionedCall?.batch_normalization_62/StatefulPartitionedCall?+conv1d_transpose_10/StatefulPartitionedCall?+conv1d_transpose_11/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? lambda_5/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCallinputsdense_47_41911953*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_419114602"
 dense_47/StatefulPartitionedCall?
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0batch_normalization_58_41911956batch_normalization_58_41911958batch_normalization_58_41911960batch_normalization_58_41911962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_4191075520
.batch_normalization_58/StatefulPartitionedCall?
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_33_layer_call_and_return_conditional_losses_419115122
re_lu_33/PartitionedCall?
reshape_11/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_419115332
reshape_11/PartitionedCall?
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall#reshape_11/PartitionedCall:output:0batch_normalization_59_41911967batch_normalization_59_41911969batch_normalization_59_41911971batch_normalization_59_41911973*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_4191157620
.batch_normalization_59/StatefulPartitionedCall?
+conv1d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0conv1d_transpose_10_41911976*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_419109762-
+conv1d_transpose_10/StatefulPartitionedCall?
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_10/StatefulPartitionedCall:output:0batch_normalization_60_41911979batch_normalization_60_41911981batch_normalization_60_41911983batch_normalization_60_41911985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_4191108020
.batch_normalization_60/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_34_layer_call_and_return_conditional_losses_419116752
re_lu_34/PartitionedCall?
+conv1d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0conv1d_transpose_11_41911989*
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
GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_419111612-
+conv1d_transpose_11/StatefulPartitionedCall?
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_11/StatefulPartitionedCall:output:0batch_normalization_61_41911992batch_normalization_61_41911994batch_normalization_61_41911996batch_normalization_61_41911998*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_4191126520
.batch_normalization_61/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_35_layer_call_and_return_conditional_losses_419117262
re_lu_35/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_419117462
flatten_11/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_48_41912003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_419117622"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_49_41912006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_419117822"
 dense_49/StatefulPartitionedCall?
 lambda_5/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118102"
 lambda_5/StatefulPartitionedCall?
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)lambda_5/StatefulPartitionedCall:output:0batch_normalization_62_41912010batch_normalization_62_41912012batch_normalization_62_41912014batch_normalization_62_41912016*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_4191140520
.batch_normalization_62/StatefulPartitionedCall?
IdentityIdentity7batch_normalization_62/StatefulPartitionedCall:output:0/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall,^conv1d_transpose_10/StatefulPartitionedCall,^conv1d_transpose_11/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^lambda_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2Z
+conv1d_transpose_10/StatefulPartitionedCall+conv1d_transpose_10/StatefulPartitionedCall2Z
+conv1d_transpose_11/StatefulPartitionedCall+conv1d_transpose_11/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 lambda_5/StatefulPartitionedCall lambda_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41911265

inputs
assignmovingavg_41911240
assignmovingavg_1_41911246)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911240*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41911240*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911240*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911240*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41911240AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911240*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911246*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41911246*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911246*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911246*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41911246AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911246*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41911576

inputs
assignmovingavg_41911551
assignmovingavg_1_41911557)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911551*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41911551*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911551*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911551*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41911551AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911551*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911557*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41911557*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911557*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911557*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41911557AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911557*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41911438

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_reshape_11_layer_call_fn_41912937

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_419115332
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_Encoder_layer_call_fn_41912198
input_19
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Encoder_layer_call_and_return_conditional_losses_419121452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?
d
H__inference_reshape_11_layer_call_and_return_conditional_losses_41912932

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_34_layer_call_and_return_conditional_losses_41911675

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_35_layer_call_and_return_conditional_losses_41911726

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

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
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41910755

inputs
assignmovingavg_41910730
assignmovingavg_1_41910736)
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
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
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
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41910730*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41910730*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41910730*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41910730*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41910730AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41910730*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41910736*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41910736*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41910736*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41910736*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41910736AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41910736*
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
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_59_layer_call_fn_41913088

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
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_419115762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913364
inputs_0
inputs_1
identity?[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpV
mulMulinputs_0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulE
ShapeShapemul:z:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normalb
addAddV2inputs_0random_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
9__inference_batch_normalization_58_layer_call_fn_41912909

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
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_419107882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912145

inputs
dense_47_41912078#
batch_normalization_58_41912081#
batch_normalization_58_41912083#
batch_normalization_58_41912085#
batch_normalization_58_41912087#
batch_normalization_59_41912092#
batch_normalization_59_41912094#
batch_normalization_59_41912096#
batch_normalization_59_41912098 
conv1d_transpose_10_41912101#
batch_normalization_60_41912104#
batch_normalization_60_41912106#
batch_normalization_60_41912108#
batch_normalization_60_41912110 
conv1d_transpose_11_41912114#
batch_normalization_61_41912117#
batch_normalization_61_41912119#
batch_normalization_61_41912121#
batch_normalization_61_41912123
dense_48_41912128
dense_49_41912131#
batch_normalization_62_41912135#
batch_normalization_62_41912137#
batch_normalization_62_41912139#
batch_normalization_62_41912141
identity??.batch_normalization_58/StatefulPartitionedCall?.batch_normalization_59/StatefulPartitionedCall?.batch_normalization_60/StatefulPartitionedCall?.batch_normalization_61/StatefulPartitionedCall?.batch_normalization_62/StatefulPartitionedCall?+conv1d_transpose_10/StatefulPartitionedCall?+conv1d_transpose_11/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? lambda_5/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCallinputsdense_47_41912078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_419114602"
 dense_47/StatefulPartitionedCall?
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0batch_normalization_58_41912081batch_normalization_58_41912083batch_normalization_58_41912085batch_normalization_58_41912087*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_4191078820
.batch_normalization_58/StatefulPartitionedCall?
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_33_layer_call_and_return_conditional_losses_419115122
re_lu_33/PartitionedCall?
reshape_11/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_419115332
reshape_11/PartitionedCall?
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall#reshape_11/PartitionedCall:output:0batch_normalization_59_41912092batch_normalization_59_41912094batch_normalization_59_41912096batch_normalization_59_41912098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_4191159620
.batch_normalization_59/StatefulPartitionedCall?
+conv1d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0conv1d_transpose_10_41912101*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_419109762-
+conv1d_transpose_10/StatefulPartitionedCall?
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_10/StatefulPartitionedCall:output:0batch_normalization_60_41912104batch_normalization_60_41912106batch_normalization_60_41912108batch_normalization_60_41912110*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_4191111320
.batch_normalization_60/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_34_layer_call_and_return_conditional_losses_419116752
re_lu_34/PartitionedCall?
+conv1d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0conv1d_transpose_11_41912114*
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
GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_419111612-
+conv1d_transpose_11/StatefulPartitionedCall?
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_11/StatefulPartitionedCall:output:0batch_normalization_61_41912117batch_normalization_61_41912119batch_normalization_61_41912121batch_normalization_61_41912123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_4191129820
.batch_normalization_61/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_35_layer_call_and_return_conditional_losses_419117262
re_lu_35/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_419117462
flatten_11/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_48_41912128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_419117622"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_49_41912131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_419117822"
 dense_49/StatefulPartitionedCall?
 lambda_5/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118262"
 lambda_5/StatefulPartitionedCall?
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)lambda_5/StatefulPartitionedCall:output:0batch_normalization_62_41912135batch_normalization_62_41912137batch_normalization_62_41912139batch_normalization_62_41912141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_4191143820
.batch_normalization_62/StatefulPartitionedCall?
IdentityIdentity7batch_normalization_62/StatefulPartitionedCall:output:0/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall,^conv1d_transpose_10/StatefulPartitionedCall,^conv1d_transpose_11/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^lambda_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2Z
+conv1d_transpose_10/StatefulPartitionedCall+conv1d_transpose_10/StatefulPartitionedCall2Z
+conv1d_transpose_11/StatefulPartitionedCall+conv1d_transpose_11/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 lambda_5/StatefulPartitionedCall lambda_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_34_layer_call_and_return_conditional_losses_41913188

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
F__inference_dense_48_layer_call_and_return_conditional_losses_41913310

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
G
+__inference_re_lu_34_layer_call_fn_41913193

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
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_34_layer_call_and_return_conditional_losses_419116752
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_59_layer_call_fn_41913101

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
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_419115962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_41910976

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
value	B :2	
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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
IdentityIdentity!conv1d_transpose/Squeeze:output:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_33_layer_call_and_return_conditional_losses_41912914

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_Encoder_layer_call_fn_41912073
input_19
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_Encoder_layer_call_and_return_conditional_losses_419120202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?1
?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913137

inputs
assignmovingavg_41913112
assignmovingavg_1_41913118)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913112*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41913112*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913112*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913112*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41913112AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913112*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913118*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41913118*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913118*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913118*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41913118AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913118*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912703

inputs+
'dense_47_matmul_readvariableop_resource<
8batch_normalization_58_batchnorm_readvariableop_resource@
<batch_normalization_58_batchnorm_mul_readvariableop_resource>
:batch_normalization_58_batchnorm_readvariableop_1_resource>
:batch_normalization_58_batchnorm_readvariableop_2_resource<
8batch_normalization_59_batchnorm_readvariableop_resource@
<batch_normalization_59_batchnorm_mul_readvariableop_resource>
:batch_normalization_59_batchnorm_readvariableop_1_resource>
:batch_normalization_59_batchnorm_readvariableop_2_resourceM
Iconv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_60_batchnorm_readvariableop_resource@
<batch_normalization_60_batchnorm_mul_readvariableop_resource>
:batch_normalization_60_batchnorm_readvariableop_1_resource>
:batch_normalization_60_batchnorm_readvariableop_2_resourceM
Iconv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_61_batchnorm_readvariableop_resource@
<batch_normalization_61_batchnorm_mul_readvariableop_resource>
:batch_normalization_61_batchnorm_readvariableop_1_resource>
:batch_normalization_61_batchnorm_readvariableop_2_resource+
'dense_48_matmul_readvariableop_resource+
'dense_49_matmul_readvariableop_resource<
8batch_normalization_62_batchnorm_readvariableop_resource@
<batch_normalization_62_batchnorm_mul_readvariableop_resource>
:batch_normalization_62_batchnorm_readvariableop_1_resource>
:batch_normalization_62_batchnorm_readvariableop_2_resource
identity??/batch_normalization_58/batchnorm/ReadVariableOp?1batch_normalization_58/batchnorm/ReadVariableOp_1?1batch_normalization_58/batchnorm/ReadVariableOp_2?3batch_normalization_58/batchnorm/mul/ReadVariableOp?/batch_normalization_59/batchnorm/ReadVariableOp?1batch_normalization_59/batchnorm/ReadVariableOp_1?1batch_normalization_59/batchnorm/ReadVariableOp_2?3batch_normalization_59/batchnorm/mul/ReadVariableOp?/batch_normalization_60/batchnorm/ReadVariableOp?1batch_normalization_60/batchnorm/ReadVariableOp_1?1batch_normalization_60/batchnorm/ReadVariableOp_2?3batch_normalization_60/batchnorm/mul/ReadVariableOp?/batch_normalization_61/batchnorm/ReadVariableOp?1batch_normalization_61/batchnorm/ReadVariableOp_1?1batch_normalization_61/batchnorm/ReadVariableOp_2?3batch_normalization_61/batchnorm/mul/ReadVariableOp?/batch_normalization_62/batchnorm/ReadVariableOp?1batch_normalization_62/batchnorm/ReadVariableOp_1?1batch_normalization_62/batchnorm/ReadVariableOp_2?3batch_normalization_62/batchnorm/mul/ReadVariableOp?@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/MatMul/ReadVariableOp?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMulinputs&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
/batch_normalization_58/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_58_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_58/batchnorm/ReadVariableOp?
&batch_normalization_58/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_58/batchnorm/add/y?
$batch_normalization_58/batchnorm/addAddV27batch_normalization_58/batchnorm/ReadVariableOp:value:0/batch_normalization_58/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/add?
&batch_normalization_58/batchnorm/RsqrtRsqrt(batch_normalization_58/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_58/batchnorm/Rsqrt?
3batch_normalization_58/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_58_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_58/batchnorm/mul/ReadVariableOp?
$batch_normalization_58/batchnorm/mulMul*batch_normalization_58/batchnorm/Rsqrt:y:0;batch_normalization_58/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/mul?
&batch_normalization_58/batchnorm/mul_1Muldense_47/MatMul:product:0(batch_normalization_58/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_58/batchnorm/mul_1?
1batch_normalization_58/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_58_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_58/batchnorm/ReadVariableOp_1?
&batch_normalization_58/batchnorm/mul_2Mul9batch_normalization_58/batchnorm/ReadVariableOp_1:value:0(batch_normalization_58/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_58/batchnorm/mul_2?
1batch_normalization_58/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_58_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_58/batchnorm/ReadVariableOp_2?
$batch_normalization_58/batchnorm/subSub9batch_normalization_58/batchnorm/ReadVariableOp_2:value:0*batch_normalization_58/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_58/batchnorm/sub?
&batch_normalization_58/batchnorm/add_1AddV2*batch_normalization_58/batchnorm/mul_1:z:0(batch_normalization_58/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_58/batchnorm/add_1?
re_lu_33/ReluRelu*batch_normalization_58/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
re_lu_33/Reluo
reshape_11/ShapeShapere_lu_33/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape?
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack?
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1?
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2?
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2?
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape?
reshape_11/ReshapeReshapere_lu_33/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_11/Reshape?
/batch_normalization_59/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_59_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_59/batchnorm/ReadVariableOp?
&batch_normalization_59/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_59/batchnorm/add/y?
$batch_normalization_59/batchnorm/addAddV27batch_normalization_59/batchnorm/ReadVariableOp:value:0/batch_normalization_59/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/add?
&batch_normalization_59/batchnorm/RsqrtRsqrt(batch_normalization_59/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_59/batchnorm/Rsqrt?
3batch_normalization_59/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_59_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_59/batchnorm/mul/ReadVariableOp?
$batch_normalization_59/batchnorm/mulMul*batch_normalization_59/batchnorm/Rsqrt:y:0;batch_normalization_59/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/mul?
&batch_normalization_59/batchnorm/mul_1Mulreshape_11/Reshape:output:0(batch_normalization_59/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_59/batchnorm/mul_1?
1batch_normalization_59/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_59_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_59/batchnorm/ReadVariableOp_1?
&batch_normalization_59/batchnorm/mul_2Mul9batch_normalization_59/batchnorm/ReadVariableOp_1:value:0(batch_normalization_59/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_59/batchnorm/mul_2?
1batch_normalization_59/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_59_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_59/batchnorm/ReadVariableOp_2?
$batch_normalization_59/batchnorm/subSub9batch_normalization_59/batchnorm/ReadVariableOp_2:value:0*batch_normalization_59/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_59/batchnorm/sub?
&batch_normalization_59/batchnorm/add_1AddV2*batch_normalization_59/batchnorm/mul_1:z:0(batch_normalization_59/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_59/batchnorm/add_1?
conv1d_transpose_10/ShapeShape*batch_normalization_59/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
conv1d_transpose_10/Shape?
'conv1d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv1d_transpose_10/strided_slice/stack?
)conv1d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice/stack_1?
)conv1d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice/stack_2?
!conv1d_transpose_10/strided_sliceStridedSlice"conv1d_transpose_10/Shape:output:00conv1d_transpose_10/strided_slice/stack:output:02conv1d_transpose_10/strided_slice/stack_1:output:02conv1d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv1d_transpose_10/strided_slice?
)conv1d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_10/strided_slice_1/stack?
+conv1d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_10/strided_slice_1/stack_1?
+conv1d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_10/strided_slice_1/stack_2?
#conv1d_transpose_10/strided_slice_1StridedSlice"conv1d_transpose_10/Shape:output:02conv1d_transpose_10/strided_slice_1/stack:output:04conv1d_transpose_10/strided_slice_1/stack_1:output:04conv1d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv1d_transpose_10/strided_slice_1x
conv1d_transpose_10/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_10/mul/y?
conv1d_transpose_10/mulMul,conv1d_transpose_10/strided_slice_1:output:0"conv1d_transpose_10/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_10/mul|
conv1d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_10/stack/2?
conv1d_transpose_10/stackPack*conv1d_transpose_10/strided_slice:output:0conv1d_transpose_10/mul:z:0$conv1d_transpose_10/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_10/stack?
3conv1d_transpose_10/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3conv1d_transpose_10/conv1d_transpose/ExpandDims/dim?
/conv1d_transpose_10/conv1d_transpose/ExpandDims
ExpandDims*batch_normalization_59/batchnorm/add_1:z:0<conv1d_transpose_10/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????21
/conv1d_transpose_10/conv1d_transpose/ExpandDims?
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpIconv1d_transpose_10_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02B
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp?
5conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim?
1conv1d_transpose_10/conv1d_transpose/ExpandDims_1
ExpandDimsHconv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0>conv1d_transpose_10/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:23
1conv1d_transpose_10/conv1d_transpose/ExpandDims_1?
8conv1d_transpose_10/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8conv1d_transpose_10/conv1d_transpose/strided_slice/stack?
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_1?
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice/stack_2?
2conv1d_transpose_10/conv1d_transpose/strided_sliceStridedSlice"conv1d_transpose_10/stack:output:0Aconv1d_transpose_10/conv1d_transpose/strided_slice/stack:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice/stack_1:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2conv1d_transpose_10/conv1d_transpose/strided_slice?
:conv1d_transpose_10/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack?
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1?
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<conv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2?
4conv1d_transpose_10/conv1d_transpose/strided_slice_1StridedSlice"conv1d_transpose_10/stack:output:0Cconv1d_transpose_10/conv1d_transpose/strided_slice_1/stack:output:0Econv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_1:output:0Econv1d_transpose_10/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask26
4conv1d_transpose_10/conv1d_transpose/strided_slice_1?
4conv1d_transpose_10/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:26
4conv1d_transpose_10/conv1d_transpose/concat/values_1?
0conv1d_transpose_10/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0conv1d_transpose_10/conv1d_transpose/concat/axis?
+conv1d_transpose_10/conv1d_transpose/concatConcatV2;conv1d_transpose_10/conv1d_transpose/strided_slice:output:0=conv1d_transpose_10/conv1d_transpose/concat/values_1:output:0=conv1d_transpose_10/conv1d_transpose/strided_slice_1:output:09conv1d_transpose_10/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_10/conv1d_transpose/concat?
$conv1d_transpose_10/conv1d_transposeConv2DBackpropInput4conv1d_transpose_10/conv1d_transpose/concat:output:0:conv1d_transpose_10/conv1d_transpose/ExpandDims_1:output:08conv1d_transpose_10/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2&
$conv1d_transpose_10/conv1d_transpose?
,conv1d_transpose_10/conv1d_transpose/SqueezeSqueeze-conv1d_transpose_10/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2.
,conv1d_transpose_10/conv1d_transpose/Squeeze?
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOp?
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_60/batchnorm/add/y?
$batch_normalization_60/batchnorm/addAddV27batch_normalization_60/batchnorm/ReadVariableOp:value:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/add?
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_60/batchnorm/Rsqrt?
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOp?
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/mul?
&batch_normalization_60/batchnorm/mul_1Mul5conv1d_transpose_10/conv1d_transpose/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_60/batchnorm/mul_1?
1batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_1?
&batch_normalization_60/batchnorm/mul_2Mul9batch_normalization_60/batchnorm/ReadVariableOp_1:value:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_60/batchnorm/mul_2?
1batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_2?
$batch_normalization_60/batchnorm/subSub9batch_normalization_60/batchnorm/ReadVariableOp_2:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_60/batchnorm/sub?
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_60/batchnorm/add_1?
re_lu_34/ReluRelu*batch_normalization_60/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_34/Relu?
conv1d_transpose_11/ShapeShapere_lu_34/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_11/Shape?
'conv1d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv1d_transpose_11/strided_slice/stack?
)conv1d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice/stack_1?
)conv1d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice/stack_2?
!conv1d_transpose_11/strided_sliceStridedSlice"conv1d_transpose_11/Shape:output:00conv1d_transpose_11/strided_slice/stack:output:02conv1d_transpose_11/strided_slice/stack_1:output:02conv1d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv1d_transpose_11/strided_slice?
)conv1d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)conv1d_transpose_11/strided_slice_1/stack?
+conv1d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_11/strided_slice_1/stack_1?
+conv1d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv1d_transpose_11/strided_slice_1/stack_2?
#conv1d_transpose_11/strided_slice_1StridedSlice"conv1d_transpose_11/Shape:output:02conv1d_transpose_11/strided_slice_1/stack:output:04conv1d_transpose_11/strided_slice_1/stack_1:output:04conv1d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv1d_transpose_11/strided_slice_1x
conv1d_transpose_11/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_11/mul/y?
conv1d_transpose_11/mulMul,conv1d_transpose_11/strided_slice_1:output:0"conv1d_transpose_11/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_11/mul|
conv1d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_11/stack/2?
conv1d_transpose_11/stackPack*conv1d_transpose_11/strided_slice:output:0conv1d_transpose_11/mul:z:0$conv1d_transpose_11/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_11/stack?
3conv1d_transpose_11/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3conv1d_transpose_11/conv1d_transpose/ExpandDims/dim?
/conv1d_transpose_11/conv1d_transpose/ExpandDims
ExpandDimsre_lu_34/Relu:activations:0<conv1d_transpose_11/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????21
/conv1d_transpose_11/conv1d_transpose/ExpandDims?
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpIconv1d_transpose_11_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02B
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp?
5conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim?
1conv1d_transpose_11/conv1d_transpose/ExpandDims_1
ExpandDimsHconv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0>conv1d_transpose_11/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:23
1conv1d_transpose_11/conv1d_transpose/ExpandDims_1?
8conv1d_transpose_11/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8conv1d_transpose_11/conv1d_transpose/strided_slice/stack?
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_1?
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice/stack_2?
2conv1d_transpose_11/conv1d_transpose/strided_sliceStridedSlice"conv1d_transpose_11/stack:output:0Aconv1d_transpose_11/conv1d_transpose/strided_slice/stack:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice/stack_1:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2conv1d_transpose_11/conv1d_transpose/strided_slice?
:conv1d_transpose_11/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack?
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1?
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<conv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2?
4conv1d_transpose_11/conv1d_transpose/strided_slice_1StridedSlice"conv1d_transpose_11/stack:output:0Cconv1d_transpose_11/conv1d_transpose/strided_slice_1/stack:output:0Econv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_1:output:0Econv1d_transpose_11/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask26
4conv1d_transpose_11/conv1d_transpose/strided_slice_1?
4conv1d_transpose_11/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:26
4conv1d_transpose_11/conv1d_transpose/concat/values_1?
0conv1d_transpose_11/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0conv1d_transpose_11/conv1d_transpose/concat/axis?
+conv1d_transpose_11/conv1d_transpose/concatConcatV2;conv1d_transpose_11/conv1d_transpose/strided_slice:output:0=conv1d_transpose_11/conv1d_transpose/concat/values_1:output:0=conv1d_transpose_11/conv1d_transpose/strided_slice_1:output:09conv1d_transpose_11/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+conv1d_transpose_11/conv1d_transpose/concat?
$conv1d_transpose_11/conv1d_transposeConv2DBackpropInput4conv1d_transpose_11/conv1d_transpose/concat:output:0:conv1d_transpose_11/conv1d_transpose/ExpandDims_1:output:08conv1d_transpose_11/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2&
$conv1d_transpose_11/conv1d_transpose?
,conv1d_transpose_11/conv1d_transpose/SqueezeSqueeze-conv1d_transpose_11/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2.
,conv1d_transpose_11/conv1d_transpose/Squeeze?
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp?
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_61/batchnorm/add/y?
$batch_normalization_61/batchnorm/addAddV27batch_normalization_61/batchnorm/ReadVariableOp:value:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/add?
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_61/batchnorm/Rsqrt?
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOp?
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/mul?
&batch_normalization_61/batchnorm/mul_1Mul5conv1d_transpose_11/conv1d_transpose/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_61/batchnorm/mul_1?
1batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_1?
&batch_normalization_61/batchnorm/mul_2Mul9batch_normalization_61/batchnorm/ReadVariableOp_1:value:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_61/batchnorm/mul_2?
1batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_2?
$batch_normalization_61/batchnorm/subSub9batch_normalization_61/batchnorm/ReadVariableOp_2:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_61/batchnorm/sub?
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_61/batchnorm/add_1?
re_lu_35/ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_35/Reluu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@   2
flatten_11/Const?
flatten_11/ReshapeReshapere_lu_35/Relu:activations:0flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????@2
flatten_11/Reshape?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMulflatten_11/Reshape:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_48/MatMuls
dense_48/TanhTanhdense_48/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_48/Tanh?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMulflatten_11/Reshape:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_49/MatMuls
dense_49/TanhTanhdense_49/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_49/Tanhm
lambda_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_5/truediv/y?
lambda_5/truedivRealDivdense_49/Tanh:y:0lambda_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/truedivk
lambda_5/ExpExplambda_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_5/Expz
lambda_5/mulMuldense_48/Tanh:y:0lambda_5/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_5/mul`
lambda_5/ShapeShapelambda_5/mul:z:0*
T0*
_output_shapes
:2
lambda_5/Shape
lambda_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_5/random_normal/mean?
lambda_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda_5/random_normal/stddev?
+lambda_5/random_normal/RandomStandardNormalRandomStandardNormallambda_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??c2-
+lambda_5/random_normal/RandomStandardNormal?
lambda_5/random_normal/mulMul4lambda_5/random_normal/RandomStandardNormal:output:0&lambda_5/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/random_normal/mul?
lambda_5/random_normalAddlambda_5/random_normal/mul:z:0$lambda_5/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_5/random_normal?
lambda_5/addAddV2dense_48/Tanh:y:0lambda_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
lambda_5/add?
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOp?
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_62/batchnorm/add/y?
$batch_normalization_62/batchnorm/addAddV27batch_normalization_62/batchnorm/ReadVariableOp:value:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/add?
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_62/batchnorm/Rsqrt?
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOp?
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/mul?
&batch_normalization_62/batchnorm/mul_1Mullambda_5/add:z:0(batch_normalization_62/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_62/batchnorm/mul_1?
1batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_1?
&batch_normalization_62/batchnorm/mul_2Mul9batch_normalization_62/batchnorm/ReadVariableOp_1:value:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_62/batchnorm/mul_2?
1batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_2?
$batch_normalization_62/batchnorm/subSub9batch_normalization_62/batchnorm/ReadVariableOp_2:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_62/batchnorm/sub?
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_62/batchnorm/add_1?

IdentityIdentity*batch_normalization_62/batchnorm/add_1:z:00^batch_normalization_58/batchnorm/ReadVariableOp2^batch_normalization_58/batchnorm/ReadVariableOp_12^batch_normalization_58/batchnorm/ReadVariableOp_24^batch_normalization_58/batchnorm/mul/ReadVariableOp0^batch_normalization_59/batchnorm/ReadVariableOp2^batch_normalization_59/batchnorm/ReadVariableOp_12^batch_normalization_59/batchnorm/ReadVariableOp_24^batch_normalization_59/batchnorm/mul/ReadVariableOp0^batch_normalization_60/batchnorm/ReadVariableOp2^batch_normalization_60/batchnorm/ReadVariableOp_12^batch_normalization_60/batchnorm/ReadVariableOp_24^batch_normalization_60/batchnorm/mul/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp2^batch_normalization_61/batchnorm/ReadVariableOp_12^batch_normalization_61/batchnorm/ReadVariableOp_24^batch_normalization_61/batchnorm/mul/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp2^batch_normalization_62/batchnorm/ReadVariableOp_12^batch_normalization_62/batchnorm/ReadVariableOp_24^batch_normalization_62/batchnorm/mul/ReadVariableOpA^conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOpA^conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_47/MatMul/ReadVariableOp^dense_48/MatMul/ReadVariableOp^dense_49/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2b
/batch_normalization_58/batchnorm/ReadVariableOp/batch_normalization_58/batchnorm/ReadVariableOp2f
1batch_normalization_58/batchnorm/ReadVariableOp_11batch_normalization_58/batchnorm/ReadVariableOp_12f
1batch_normalization_58/batchnorm/ReadVariableOp_21batch_normalization_58/batchnorm/ReadVariableOp_22j
3batch_normalization_58/batchnorm/mul/ReadVariableOp3batch_normalization_58/batchnorm/mul/ReadVariableOp2b
/batch_normalization_59/batchnorm/ReadVariableOp/batch_normalization_59/batchnorm/ReadVariableOp2f
1batch_normalization_59/batchnorm/ReadVariableOp_11batch_normalization_59/batchnorm/ReadVariableOp_12f
1batch_normalization_59/batchnorm/ReadVariableOp_21batch_normalization_59/batchnorm/ReadVariableOp_22j
3batch_normalization_59/batchnorm/mul/ReadVariableOp3batch_normalization_59/batchnorm/mul/ReadVariableOp2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2f
1batch_normalization_60/batchnorm/ReadVariableOp_11batch_normalization_60/batchnorm/ReadVariableOp_12f
1batch_normalization_60/batchnorm/ReadVariableOp_21batch_normalization_60/batchnorm/ReadVariableOp_22j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2f
1batch_normalization_61/batchnorm/ReadVariableOp_11batch_normalization_61/batchnorm/ReadVariableOp_12f
1batch_normalization_61/batchnorm/ReadVariableOp_21batch_normalization_61/batchnorm/ReadVariableOp_22j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2f
1batch_normalization_62/batchnorm/ReadVariableOp_11batch_normalization_62/batchnorm/ReadVariableOp_12f
1batch_normalization_62/batchnorm/ReadVariableOp_21batch_normalization_62/batchnorm/ReadVariableOp_22j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2?
@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp@conv1d_transpose_10/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp@conv1d_transpose_11/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_11_layer_call_and_return_conditional_losses_41913297

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
?
t
+__inference_lambda_5_layer_call_fn_41913376
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41911596

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41911298

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_41911161

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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
F__inference_dense_49_layer_call_and_return_conditional_losses_41911782

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913249

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?1
?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913229

inputs
assignmovingavg_41913204
assignmovingavg_1_41913210)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913204*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41913204*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913204*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913204*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41913204AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913204*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913210*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41913210*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913210*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913210*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41913210AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913210*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?>
?
!__inference__traced_save_41913556
file_prefix.
*savev2_dense_47_kernel_read_readvariableop;
7savev2_batch_normalization_58_gamma_read_readvariableop:
6savev2_batch_normalization_58_beta_read_readvariableopA
=savev2_batch_normalization_58_moving_mean_read_readvariableopE
Asavev2_batch_normalization_58_moving_variance_read_readvariableop;
7savev2_batch_normalization_59_gamma_read_readvariableop:
6savev2_batch_normalization_59_beta_read_readvariableopA
=savev2_batch_normalization_59_moving_mean_read_readvariableopE
Asavev2_batch_normalization_59_moving_variance_read_readvariableop9
5savev2_conv1d_transpose_10_kernel_read_readvariableop;
7savev2_batch_normalization_60_gamma_read_readvariableop:
6savev2_batch_normalization_60_beta_read_readvariableopA
=savev2_batch_normalization_60_moving_mean_read_readvariableopE
Asavev2_batch_normalization_60_moving_variance_read_readvariableop9
5savev2_conv1d_transpose_11_kernel_read_readvariableop;
7savev2_batch_normalization_61_gamma_read_readvariableop:
6savev2_batch_normalization_61_beta_read_readvariableopA
=savev2_batch_normalization_61_moving_mean_read_readvariableopE
Asavev2_batch_normalization_61_moving_variance_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop;
7savev2_batch_normalization_62_gamma_read_readvariableop:
6savev2_batch_normalization_62_beta_read_readvariableopA
=savev2_batch_normalization_62_moving_mean_read_readvariableopE
Asavev2_batch_normalization_62_moving_variance_read_readvariableop
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
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_47_kernel_read_readvariableop7savev2_batch_normalization_58_gamma_read_readvariableop6savev2_batch_normalization_58_beta_read_readvariableop=savev2_batch_normalization_58_moving_mean_read_readvariableopAsavev2_batch_normalization_58_moving_variance_read_readvariableop7savev2_batch_normalization_59_gamma_read_readvariableop6savev2_batch_normalization_59_beta_read_readvariableop=savev2_batch_normalization_59_moving_mean_read_readvariableopAsavev2_batch_normalization_59_moving_variance_read_readvariableop5savev2_conv1d_transpose_10_kernel_read_readvariableop7savev2_batch_normalization_60_gamma_read_readvariableop6savev2_batch_normalization_60_beta_read_readvariableop=savev2_batch_normalization_60_moving_mean_read_readvariableopAsavev2_batch_normalization_60_moving_variance_read_readvariableop5savev2_conv1d_transpose_11_kernel_read_readvariableop7savev2_batch_normalization_61_gamma_read_readvariableop6savev2_batch_normalization_61_beta_read_readvariableop=savev2_batch_normalization_61_moving_mean_read_readvariableopAsavev2_batch_normalization_61_moving_variance_read_readvariableop*savev2_dense_48_kernel_read_readvariableop*savev2_dense_49_kernel_read_readvariableop7savev2_batch_normalization_62_gamma_read_readvariableop6savev2_batch_normalization_62_beta_read_readvariableop=savev2_batch_normalization_62_moving_mean_read_readvariableopAsavev2_batch_normalization_62_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
22
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
?: :	?:?:?:?:?:::::::::::::::@:@::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
::(
$
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?1
?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41911080

inputs
assignmovingavg_41911055
assignmovingavg_1_41911061)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911055*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41911055*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911055*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911055*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41911055AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911055*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911061*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41911061*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911061*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911061*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41911061AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911061*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
t
+__inference_lambda_5_layer_call_fn_41913370
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
u
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913348
inputs_0
inputs_1
identity?[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpV
mulMulinputs_0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulE
ShapeShapemul:z:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??32$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normalb
addAddV2inputs_0random_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?0
?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41911405

inputs
assignmovingavg_41911380
assignmovingavg_1_41911386)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911380*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41911380*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911380*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41911380*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41911380AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41911380*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911386*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41911386*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911386*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41911386*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41911386AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41911386*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_59_layer_call_fn_41913019

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
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_419109282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_59_layer_call_fn_41913006

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
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_419108952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_41912255
input_19
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:?????????*;
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_419106592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?U
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41911877
input_19
dense_47_41911469#
batch_normalization_58_41911498#
batch_normalization_58_41911500#
batch_normalization_58_41911502#
batch_normalization_58_41911504#
batch_normalization_59_41911623#
batch_normalization_59_41911625#
batch_normalization_59_41911627#
batch_normalization_59_41911629 
conv1d_transpose_10_41911632#
batch_normalization_60_41911661#
batch_normalization_60_41911663#
batch_normalization_60_41911665#
batch_normalization_60_41911667 
conv1d_transpose_11_41911683#
batch_normalization_61_41911712#
batch_normalization_61_41911714#
batch_normalization_61_41911716#
batch_normalization_61_41911718
dense_48_41911771
dense_49_41911791#
batch_normalization_62_41911867#
batch_normalization_62_41911869#
batch_normalization_62_41911871#
batch_normalization_62_41911873
identity??.batch_normalization_58/StatefulPartitionedCall?.batch_normalization_59/StatefulPartitionedCall?.batch_normalization_60/StatefulPartitionedCall?.batch_normalization_61/StatefulPartitionedCall?.batch_normalization_62/StatefulPartitionedCall?+conv1d_transpose_10/StatefulPartitionedCall?+conv1d_transpose_11/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? lambda_5/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_47_41911469*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_419114602"
 dense_47/StatefulPartitionedCall?
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0batch_normalization_58_41911498batch_normalization_58_41911500batch_normalization_58_41911502batch_normalization_58_41911504*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_4191075520
.batch_normalization_58/StatefulPartitionedCall?
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_33_layer_call_and_return_conditional_losses_419115122
re_lu_33/PartitionedCall?
reshape_11/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_419115332
reshape_11/PartitionedCall?
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall#reshape_11/PartitionedCall:output:0batch_normalization_59_41911623batch_normalization_59_41911625batch_normalization_59_41911627batch_normalization_59_41911629*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_4191157620
.batch_normalization_59/StatefulPartitionedCall?
+conv1d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0conv1d_transpose_10_41911632*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_419109762-
+conv1d_transpose_10/StatefulPartitionedCall?
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_10/StatefulPartitionedCall:output:0batch_normalization_60_41911661batch_normalization_60_41911663batch_normalization_60_41911665batch_normalization_60_41911667*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_4191108020
.batch_normalization_60/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_34_layer_call_and_return_conditional_losses_419116752
re_lu_34/PartitionedCall?
+conv1d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0conv1d_transpose_11_41911683*
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
GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_419111612-
+conv1d_transpose_11/StatefulPartitionedCall?
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_11/StatefulPartitionedCall:output:0batch_normalization_61_41911712batch_normalization_61_41911714batch_normalization_61_41911716batch_normalization_61_41911718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_4191126520
.batch_normalization_61/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_35_layer_call_and_return_conditional_losses_419117262
re_lu_35/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_419117462
flatten_11/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_48_41911771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_419117622"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_49_41911791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_419117822"
 dense_49/StatefulPartitionedCall?
 lambda_5/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118102"
 lambda_5/StatefulPartitionedCall?
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)lambda_5/StatefulPartitionedCall:output:0batch_normalization_62_41911867batch_normalization_62_41911869batch_normalization_62_41911871batch_normalization_62_41911873*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_4191140520
.batch_normalization_62/StatefulPartitionedCall?
IdentityIdentity7batch_normalization_62/StatefulPartitionedCall:output:0/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall,^conv1d_transpose_10/StatefulPartitionedCall,^conv1d_transpose_11/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^lambda_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2Z
+conv1d_transpose_10/StatefulPartitionedCall+conv1d_transpose_10/StatefulPartitionedCall2Z
+conv1d_transpose_11/StatefulPartitionedCall+conv1d_transpose_11/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 lambda_5/StatefulPartitionedCall lambda_5/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?
|
6__inference_conv1d_transpose_11_layer_call_fn_41911169

inputs
unknown
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_419111612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_35_layer_call_and_return_conditional_losses_41913280

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
s
F__inference_lambda_5_layer_call_and_return_conditional_losses_41911826

inputs
inputs_1
identity?[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ym
truedivRealDivinputs_1truediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpT
mulMulinputsExp:y:0*
T0*'
_output_shapes
:?????????2
mulE
ShapeShapemul:z:0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal`
addAddV2inputsrandom_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913075

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_47_layer_call_and_return_conditional_losses_41912820

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912863

inputs
assignmovingavg_41912838
assignmovingavg_1_41912844)
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
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
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
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41912838*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41912838*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41912838*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41912838*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41912838AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41912838*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41912844*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41912844*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41912844*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41912844*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41912844AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41912844*
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
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_61_layer_call_fn_41913262

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
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_419112652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
b
F__inference_re_lu_33_layer_call_and_return_conditional_losses_41911512

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_58_layer_call_fn_41912896

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
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_419107552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_dense_47_layer_call_and_return_conditional_losses_41911460

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_60_layer_call_fn_41913183

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
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_419111132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
G
+__inference_re_lu_35_layer_call_fn_41913285

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
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_35_layer_call_and_return_conditional_losses_419117262
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_61_layer_call_fn_41913275

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
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_419112982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912883

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
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
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
E__inference_Encoder_layer_call_and_return_conditional_losses_41911947
input_19
dense_47_41911880#
batch_normalization_58_41911883#
batch_normalization_58_41911885#
batch_normalization_58_41911887#
batch_normalization_58_41911889#
batch_normalization_59_41911894#
batch_normalization_59_41911896#
batch_normalization_59_41911898#
batch_normalization_59_41911900 
conv1d_transpose_10_41911903#
batch_normalization_60_41911906#
batch_normalization_60_41911908#
batch_normalization_60_41911910#
batch_normalization_60_41911912 
conv1d_transpose_11_41911916#
batch_normalization_61_41911919#
batch_normalization_61_41911921#
batch_normalization_61_41911923#
batch_normalization_61_41911925
dense_48_41911930
dense_49_41911933#
batch_normalization_62_41911937#
batch_normalization_62_41911939#
batch_normalization_62_41911941#
batch_normalization_62_41911943
identity??.batch_normalization_58/StatefulPartitionedCall?.batch_normalization_59/StatefulPartitionedCall?.batch_normalization_60/StatefulPartitionedCall?.batch_normalization_61/StatefulPartitionedCall?.batch_normalization_62/StatefulPartitionedCall?+conv1d_transpose_10/StatefulPartitionedCall?+conv1d_transpose_11/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? lambda_5/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_47_41911880*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_419114602"
 dense_47/StatefulPartitionedCall?
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0batch_normalization_58_41911883batch_normalization_58_41911885batch_normalization_58_41911887batch_normalization_58_41911889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_4191078820
.batch_normalization_58/StatefulPartitionedCall?
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_33_layer_call_and_return_conditional_losses_419115122
re_lu_33/PartitionedCall?
reshape_11/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_419115332
reshape_11/PartitionedCall?
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall#reshape_11/PartitionedCall:output:0batch_normalization_59_41911894batch_normalization_59_41911896batch_normalization_59_41911898batch_normalization_59_41911900*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_4191159620
.batch_normalization_59/StatefulPartitionedCall?
+conv1d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0conv1d_transpose_10_41911903*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_419109762-
+conv1d_transpose_10/StatefulPartitionedCall?
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_10/StatefulPartitionedCall:output:0batch_normalization_60_41911906batch_normalization_60_41911908batch_normalization_60_41911910batch_normalization_60_41911912*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_4191111320
.batch_normalization_60/StatefulPartitionedCall?
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_34_layer_call_and_return_conditional_losses_419116752
re_lu_34/PartitionedCall?
+conv1d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0conv1d_transpose_11_41911916*
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
GPU 2J 8? *Z
fURS
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_419111612-
+conv1d_transpose_11/StatefulPartitionedCall?
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall4conv1d_transpose_11/StatefulPartitionedCall:output:0batch_normalization_61_41911919batch_normalization_61_41911921batch_normalization_61_41911923batch_normalization_61_41911925*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_4191129820
.batch_normalization_61/StatefulPartitionedCall?
re_lu_35/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_re_lu_35_layer_call_and_return_conditional_losses_419117262
re_lu_35/PartitionedCall?
flatten_11/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_419117462
flatten_11/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_48_41911930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_419117622"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_49_41911933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_419117822"
 dense_49/StatefulPartitionedCall?
 lambda_5/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_419118262"
 lambda_5/StatefulPartitionedCall?
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)lambda_5/StatefulPartitionedCall:output:0batch_normalization_62_41911937batch_normalization_62_41911939batch_normalization_62_41911941batch_normalization_62_41911943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_4191143820
.batch_normalization_62/StatefulPartitionedCall?
IdentityIdentity7batch_normalization_62/StatefulPartitionedCall:output:0/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall,^conv1d_transpose_10/StatefulPartitionedCall,^conv1d_transpose_11/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^lambda_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:?????????:::::::::::::::::::::::::2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2Z
+conv1d_transpose_10/StatefulPartitionedCall+conv1d_transpose_10/StatefulPartitionedCall2Z
+conv1d_transpose_11/StatefulPartitionedCall+conv1d_transpose_11/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 lambda_5/StatefulPartitionedCall lambda_5/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_19
?0
?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913055

inputs
assignmovingavg_41913030
assignmovingavg_1_41913036)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:?????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913030*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41913030*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913030*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913030*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41913030AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913030*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913036*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41913036*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913036*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913036*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41913036AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913036*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2
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
T0*+
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_49_layer_call_and_return_conditional_losses_41913325

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_191
serving_default_input_19:0?????????J
batch_normalization_620
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_47", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_58", "inbound_nodes": [[["dense_47", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_33", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_33", "inbound_nodes": [[["batch_normalization_58", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_11", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16]}}, "name": "reshape_11", "inbound_nodes": [[["re_lu_33", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_59", "inbound_nodes": [[["reshape_11", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_10", "inbound_nodes": [[["batch_normalization_59", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_60", "inbound_nodes": [[["conv1d_transpose_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_34", "inbound_nodes": [[["batch_normalization_60", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_11", "inbound_nodes": [[["re_lu_34", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_61", "inbound_nodes": [[["conv1d_transpose_11", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_35", "inbound_nodes": [[["batch_normalization_61", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["re_lu_35", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_48", "inbound_nodes": [[["flatten_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_49", "inbound_nodes": [[["flatten_11", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+mQvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9BQUUw\nNjI4L0FBRS9uZXR3b3JrLnB52gg8bGFtYmRhPkoAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_5", "inbound_nodes": [[["dense_48", 0, 0, {}], ["dense_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_62", "inbound_nodes": [[["lambda_5", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["batch_normalization_62", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_47", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_58", "inbound_nodes": [[["dense_47", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_33", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_33", "inbound_nodes": [[["batch_normalization_58", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_11", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16]}}, "name": "reshape_11", "inbound_nodes": [[["re_lu_33", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_59", "inbound_nodes": [[["reshape_11", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_10", "inbound_nodes": [[["batch_normalization_59", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_60", "inbound_nodes": [[["conv1d_transpose_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_34", "inbound_nodes": [[["batch_normalization_60", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_11", "inbound_nodes": [[["re_lu_34", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_61", "inbound_nodes": [[["conv1d_transpose_11", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_35", "inbound_nodes": [[["batch_normalization_61", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["re_lu_35", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_48", "inbound_nodes": [[["flatten_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_49", "inbound_nodes": [[["flatten_11", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+mQvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9BQUUw\nNjI4L0FBRS9uZXR3b3JrLnB52gg8bGFtYmRhPkoAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_5", "inbound_nodes": [[["dense_48", 0, 0, {}], ["dense_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_62", "inbound_nodes": [[["lambda_5", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["batch_normalization_62", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_19", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}
?

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?	
axis
	gamma
beta
moving_mean
 moving_variance
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_58", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_58", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_33", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_11", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16]}}}
?	
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_59", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16]}}
?


6kernel
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16]}}
?	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16]}}
?
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_34", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


Hkernel
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16]}}
?	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 4]}}
?
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

^kernel
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

ckernel
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+mQvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9BQUUw\nNjI4L0FBRS9uZXR3b3JrLnB52gg8bGFtYmRhPkoAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?	
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
 "
trackable_list_wrapper
?
0
1
2
.3
/4
65
<6
=7
H8
N9
O10
^11
c12
m13
n14"
trackable_list_wrapper
?
0
1
2
3
 4
.5
/6
07
18
69
<10
=11
>12
?13
H14
N15
O16
P17
Q18
^19
c20
m21
n22
o23
p24"
trackable_list_wrapper
?
regularization_losses
ulayer_metrics
vmetrics
wnon_trainable_variables
trainable_variables
	variables
xlayer_regularization_losses

ylayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
": 	?2dense_47/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
zlayer_metrics
{metrics
|non_trainable_variables
trainable_variables
	variables
}layer_regularization_losses

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_58/gamma
*:(?2batch_normalization_58/beta
3:1? (2"batch_normalization_58/moving_mean
7:5? (2&batch_normalization_58/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
?
!regularization_losses
layer_metrics
?metrics
?non_trainable_variables
"trainable_variables
#	variables
 ?layer_regularization_losses
?layers
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
%regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
&trainable_variables
'	variables
 ?layer_regularization_losses
?layers
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
)regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
*trainable_variables
+	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_59/gamma
):'2batch_normalization_59/beta
2:0 (2"batch_normalization_59/moving_mean
6:4 (2&batch_normalization_59/moving_variance
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
?
2regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
3trainable_variables
4	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv1d_transpose_10/kernel
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
?
7regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
8trainable_variables
9	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_60/gamma
):'2batch_normalization_60/beta
2:0 (2"batch_normalization_60/moving_mean
6:4 (2&batch_normalization_60/moving_variance
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
?
@regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Atrainable_variables
B	variables
 ?layer_regularization_losses
?layers
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
Dregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Etrainable_variables
F	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.2conv1d_transpose_11/kernel
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
?
Iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Jtrainable_variables
K	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_61/gamma
):'2batch_normalization_61/beta
2:0 (2"batch_normalization_61/moving_mean
6:4 (2&batch_normalization_61/moving_variance
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
?
Rregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Strainable_variables
T	variables
 ?layer_regularization_losses
?layers
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
Vregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
Wtrainable_variables
X	variables
 ?layer_regularization_losses
?layers
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
Zregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
[trainable_variables
\	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_48/kernel
 "
trackable_list_wrapper
'
^0"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
?
_regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
`trainable_variables
a	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_49/kernel
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
?
dregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
etrainable_variables
f	variables
 ?layer_regularization_losses
?layers
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
hregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
itrainable_variables
j	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_62/gamma
):'2batch_normalization_62/beta
2:0 (2"batch_normalization_62/moving_mean
6:4 (2&batch_normalization_62/moving_variance
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
?
qregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
rtrainable_variables
s	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
 1
02
13
>4
?5
P6
Q7
o8
p9"
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
13
14
15
16"
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
0
 1"
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
.
00
11"
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
>0
?1"
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
.
P0
Q1"
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
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912703
E__inference_Encoder_layer_call_and_return_conditional_losses_41911947
E__inference_Encoder_layer_call_and_return_conditional_losses_41911877
E__inference_Encoder_layer_call_and_return_conditional_losses_41912519?
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
?2?
*__inference_Encoder_layer_call_fn_41912758
*__inference_Encoder_layer_call_fn_41912073
*__inference_Encoder_layer_call_fn_41912813
*__inference_Encoder_layer_call_fn_41912198?
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
#__inference__wrapped_model_41910659?
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
annotations? *'?$
"?
input_19?????????
?2?
F__inference_dense_47_layer_call_and_return_conditional_losses_41912820?
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
+__inference_dense_47_layer_call_fn_41912827?
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
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912883
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912863?
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
9__inference_batch_normalization_58_layer_call_fn_41912896
9__inference_batch_normalization_58_layer_call_fn_41912909?
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
F__inference_re_lu_33_layer_call_and_return_conditional_losses_41912914?
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
+__inference_re_lu_33_layer_call_fn_41912919?
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
H__inference_reshape_11_layer_call_and_return_conditional_losses_41912932?
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
-__inference_reshape_11_layer_call_fn_41912937?
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
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912993
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912973
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913075
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913055?
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
9__inference_batch_normalization_59_layer_call_fn_41913006
9__inference_batch_normalization_59_layer_call_fn_41913101
9__inference_batch_normalization_59_layer_call_fn_41913088
9__inference_batch_normalization_59_layer_call_fn_41913019?
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
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_41910976?
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
%?"??????????????????
?2?
6__inference_conv1d_transpose_10_layer_call_fn_41910984?
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
%?"??????????????????
?2?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913137
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913157?
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
9__inference_batch_normalization_60_layer_call_fn_41913183
9__inference_batch_normalization_60_layer_call_fn_41913170?
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
F__inference_re_lu_34_layer_call_and_return_conditional_losses_41913188?
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
+__inference_re_lu_34_layer_call_fn_41913193?
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
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_41911161?
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
%?"??????????????????
?2?
6__inference_conv1d_transpose_11_layer_call_fn_41911169?
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
%?"??????????????????
?2?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913249
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913229?
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
9__inference_batch_normalization_61_layer_call_fn_41913262
9__inference_batch_normalization_61_layer_call_fn_41913275?
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
F__inference_re_lu_35_layer_call_and_return_conditional_losses_41913280?
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
+__inference_re_lu_35_layer_call_fn_41913285?
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
H__inference_flatten_11_layer_call_and_return_conditional_losses_41913297?
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
-__inference_flatten_11_layer_call_fn_41913302?
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
F__inference_dense_48_layer_call_and_return_conditional_losses_41913310?
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
+__inference_dense_48_layer_call_fn_41913317?
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
F__inference_dense_49_layer_call_and_return_conditional_losses_41913325?
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
+__inference_dense_49_layer_call_fn_41913332?
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
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913364
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913348?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lambda_5_layer_call_fn_41913370
+__inference_lambda_5_layer_call_fn_41913376?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913412
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913432?
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
9__inference_batch_normalization_62_layer_call_fn_41913445
9__inference_batch_normalization_62_layer_call_fn_41913458?
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
?B?
&__inference_signature_wrapper_41912255input_19"?
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
E__inference_Encoder_layer_call_and_return_conditional_losses_41911877} 01./6>?<=HPQNO^copmn9?6
/?,
"?
input_19?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_Encoder_layer_call_and_return_conditional_losses_41911947} 1.0/6?<>=HQNPO^cpmon9?6
/?,
"?
input_19?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912519{ 01./6>?<=HPQNO^copmn7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_Encoder_layer_call_and_return_conditional_losses_41912703{ 1.0/6?<>=HQNPO^cpmon7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
*__inference_Encoder_layer_call_fn_41912073p 01./6>?<=HPQNO^copmn9?6
/?,
"?
input_19?????????
p

 
? "???????????
*__inference_Encoder_layer_call_fn_41912198p 1.0/6?<>=HQNPO^cpmon9?6
/?,
"?
input_19?????????
p 

 
? "???????????
*__inference_Encoder_layer_call_fn_41912758n 01./6>?<=HPQNO^copmn7?4
-?*
 ?
inputs?????????
p

 
? "???????????
*__inference_Encoder_layer_call_fn_41912813n 1.0/6?<>=HQNPO^cpmon7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference__wrapped_model_41910659? 1.0/6?<>=HQNPO^cpmon1?.
'?$
"?
input_19?????????
? "O?L
J
batch_normalization_620?-
batch_normalization_62??????????
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912863d 4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
T__inference_batch_normalization_58_layer_call_and_return_conditional_losses_41912883d 4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
9__inference_batch_normalization_58_layer_call_fn_41912896W 4?1
*?'
!?
inputs??????????
p
? "????????????
9__inference_batch_normalization_58_layer_call_fn_41912909W 4?1
*?'
!?
inputs??????????
p 
? "????????????
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912973|01./@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41912993|1.0/@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913055j01./7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
T__inference_batch_normalization_59_layer_call_and_return_conditional_losses_41913075j1.0/7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
9__inference_batch_normalization_59_layer_call_fn_41913006o01./@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
9__inference_batch_normalization_59_layer_call_fn_41913019o1.0/@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
9__inference_batch_normalization_59_layer_call_fn_41913088]01./7?4
-?*
$?!
inputs?????????
p
? "???????????
9__inference_batch_normalization_59_layer_call_fn_41913101]1.0/7?4
-?*
$?!
inputs?????????
p 
? "???????????
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913137|>?<=@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
T__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41913157|?<>=@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
9__inference_batch_normalization_60_layer_call_fn_41913170o>?<=@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
9__inference_batch_normalization_60_layer_call_fn_41913183o?<>=@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913229|PQNO@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
T__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41913249|QNPO@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
9__inference_batch_normalization_61_layer_call_fn_41913262oPQNO@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
9__inference_batch_normalization_61_layer_call_fn_41913275oQNPO@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913412bopmn3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
T__inference_batch_normalization_62_layer_call_and_return_conditional_losses_41913432bpmon3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
9__inference_batch_normalization_62_layer_call_fn_41913445Uopmn3?0
)?&
 ?
inputs?????????
p
? "???????????
9__inference_batch_normalization_62_layer_call_fn_41913458Upmon3?0
)?&
 ?
inputs?????????
p 
? "???????????
Q__inference_conv1d_transpose_10_layer_call_and_return_conditional_losses_41910976u6<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
6__inference_conv1d_transpose_10_layer_call_fn_41910984h6<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
Q__inference_conv1d_transpose_11_layer_call_and_return_conditional_losses_41911161uH<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
6__inference_conv1d_transpose_11_layer_call_fn_41911169hH<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
F__inference_dense_47_layer_call_and_return_conditional_losses_41912820\/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ~
+__inference_dense_47_layer_call_fn_41912827O/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_48_layer_call_and_return_conditional_losses_41913310d^8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
+__inference_dense_48_layer_call_fn_41913317W^8?5
.?+
)?&
inputs??????????????????
? "???????????
F__inference_dense_49_layer_call_and_return_conditional_losses_41913325dc8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
+__inference_dense_49_layer_call_fn_41913332Wc8?5
.?+
)?&
inputs??????????????????
? "???????????
H__inference_flatten_11_layer_call_and_return_conditional_losses_41913297n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
-__inference_flatten_11_layer_call_fn_41913302a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913348?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "%?"
?
0?????????
? ?
F__inference_lambda_5_layer_call_and_return_conditional_losses_41913364?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "%?"
?
0?????????
? ?
+__inference_lambda_5_layer_call_fn_41913370~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "???????????
+__inference_lambda_5_layer_call_fn_41913376~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "???????????
F__inference_re_lu_33_layer_call_and_return_conditional_losses_41912914Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
+__inference_re_lu_33_layer_call_fn_41912919M0?-
&?#
!?
inputs??????????
? "????????????
F__inference_re_lu_34_layer_call_and_return_conditional_losses_41913188r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
+__inference_re_lu_34_layer_call_fn_41913193e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
F__inference_re_lu_35_layer_call_and_return_conditional_losses_41913280r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
+__inference_re_lu_35_layer_call_fn_41913285e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
H__inference_reshape_11_layer_call_and_return_conditional_losses_41912932]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????
? ?
-__inference_reshape_11_layer_call_fn_41912937P0?-
&?#
!?
inputs??????????
? "???????????
&__inference_signature_wrapper_41912255? 1.0/6?<>=HQNPO^cpmon=?:
? 
3?0
.
input_19"?
input_19?????????"O?L
J
batch_normalization_620?-
batch_normalization_62?????????