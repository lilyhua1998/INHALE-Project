??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_6/kernel
?
-conv1d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_6/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_7/kernel
?
-conv1d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_7/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_8/kernel
?
-conv1d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_8/kernel*"
_output_shapes
:*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
?G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?F
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
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
^

kernel
trainable_variables
	variables
regularization_losses
	keras_api
?
axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
^

.kernel
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9	variables
:regularization_losses
;	keras_api
R
<trainable_variables
=	variables
>regularization_losses
?	keras_api
^

@kernel
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
R
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
^

Rkernel
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
?
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\trainable_variables
]	variables
^regularization_losses
_	keras_api
R
`trainable_variables
a	variables
bregularization_losses
c	keras_api
R
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
^

hkernel
itrainable_variables
j	variables
kregularization_losses
l	keras_api
^

mkernel
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
R
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
f
0
1
2
.3
44
55
@6
F7
G8
R9
X10
Y11
h12
m13
?
0
1
2
 3
!4
.5
46
57
68
79
@10
F11
G12
H13
I14
R15
X16
Y17
Z18
[19
h20
m21
 
?

vlayers
trainable_variables
wnon_trainable_variables
xlayer_regularization_losses
ymetrics
	variables
zlayer_metrics
regularization_losses
 
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
{layer_metrics
|layer_regularization_losses

}layers
trainable_variables
~metrics
	variables
non_trainable_variables
regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 2
!3
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
"trainable_variables
?metrics
#	variables
?non_trainable_variables
$regularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
&trainable_variables
?metrics
'	variables
?non_trainable_variables
(regularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
*trainable_variables
?metrics
+	variables
?non_trainable_variables
,regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

.0

.0
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
/trainable_variables
?metrics
0	variables
?non_trainable_variables
1regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
62
73
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
8trainable_variables
?metrics
9	variables
?non_trainable_variables
:regularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
<trainable_variables
?metrics
=	variables
?non_trainable_variables
>regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

@0

@0
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
Atrainable_variables
?metrics
B	variables
?non_trainable_variables
Cregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
H2
I3
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
Jtrainable_variables
?metrics
K	variables
?non_trainable_variables
Lregularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
Ntrainable_variables
?metrics
O	variables
?non_trainable_variables
Pregularization_losses
ec
VARIABLE_VALUEconv1d_transpose_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

R0

R0
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
Strainable_variables
?metrics
T	variables
?non_trainable_variables
Uregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

X0
Y1
Z2
[3
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
\trainable_variables
?metrics
]	variables
?non_trainable_variables
^regularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
`trainable_variables
?metrics
a	variables
?non_trainable_variables
bregularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
dtrainable_variables
?metrics
e	variables
?non_trainable_variables
fregularization_losses
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

h0

h0
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
itrainable_variables
?metrics
j	variables
?non_trainable_variables
kregularization_losses
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE

m0

m0
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
ntrainable_variables
?metrics
o	variables
?non_trainable_variables
pregularization_losses
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
?layers
rtrainable_variables
?metrics
s	variables
?non_trainable_variables
tregularization_losses
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
16
17
8
 0
!1
62
73
H4
I5
Z6
[7
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
 0
!1
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
60
71
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
H0
I1
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
Z0
[1
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
z
serving_default_input_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5dense_9/kernel%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/betaconv1d_transpose_6/kernel&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betaconv1d_transpose_7/kernel&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betaconv1d_transpose_8/kernel&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betadense_10/kerneldense_11/kernel*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_8543351
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp-conv1d_transpose_6/kernel/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp-conv1d_transpose_7/kernel/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp-conv1d_transpose_8/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOpConst*#
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_8544495
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv1d_transpose_6/kernelbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv1d_transpose_7/kernelbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv1d_transpose_8/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancedense_10/kerneldense_11/kernel*"
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_8544571??
?
r
E__inference_lambda_1_layer_call_and_return_conditional_losses_8542990

inputs
inputs_1
identity?D
ShapeShapeinputs*
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
seed2??2$
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
random_normal[
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
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulV
addAddV2inputsmul:z:0*
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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544095

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
??
?
D__inference_Encoder_layer_call_and_return_conditional_losses_8543616

inputs*
&dense_9_matmul_readvariableop_resource1
-batch_normalization_9_assignmovingavg_85433653
/batch_normalization_9_assignmovingavg_1_8543371?
;batch_normalization_9_batchnorm_mul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resourceL
Hconv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resource2
.batch_normalization_10_assignmovingavg_85434394
0batch_normalization_10_assignmovingavg_1_8543445@
<batch_normalization_10_batchnorm_mul_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resourceL
Hconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource2
.batch_normalization_11_assignmovingavg_85435044
0batch_normalization_11_assignmovingavg_1_8543510@
<batch_normalization_11_batchnorm_mul_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resourceL
Hconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource2
.batch_normalization_12_assignmovingavg_85435694
0batch_normalization_12_assignmovingavg_1_8543575@
<batch_normalization_12_batchnorm_mul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource+
'dense_11_matmul_readvariableop_resource
identity??:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_10/AssignMovingAvg/ReadVariableOp?<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_10/batchnorm/ReadVariableOp?3batch_normalization_10/batchnorm/mul/ReadVariableOp?:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_11/AssignMovingAvg/ReadVariableOp?<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_11/batchnorm/ReadVariableOp?3batch_normalization_11/batchnorm/mul/ReadVariableOp?:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_12/AssignMovingAvg/ReadVariableOp?<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_12/batchnorm/ReadVariableOp?3batch_normalization_12/batchnorm/mul/ReadVariableOp?9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_9/AssignMovingAvg/ReadVariableOp?;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_9/batchnorm/ReadVariableOp?2batch_normalization_9/batchnorm/mul/ReadVariableOp??conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices?
"batch_normalization_9/moments/meanMeandense_9/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean?
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradient?
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_9/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????21
/batch_normalization_9/moments/SquaredDifference?
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices?
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/variance?
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze?
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1?
+batch_normalization_9/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_9/AssignMovingAvg/8543365*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_9/AssignMovingAvg/decay?
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_9_assignmovingavg_8543365*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp?
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_9/AssignMovingAvg/8543365*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub?
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_9/AssignMovingAvg/8543365*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul?
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_9_assignmovingavg_8543365-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_9/AssignMovingAvg/8543365*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_9/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_9/AssignMovingAvg_1/8543371*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_9/AssignMovingAvg_1/decay?
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_9_assignmovingavg_1_8543371*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_9/AssignMovingAvg_1/8543371*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub?
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_9/AssignMovingAvg_1/8543371*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul?
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_9_assignmovingavg_1_8543371/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_9/AssignMovingAvg_1/8543371*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_9/batchnorm/add/y?
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add?
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt?
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOp?
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul?
%batch_normalization_9/batchnorm/mul_1Muldense_9/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_9/batchnorm/mul_1?
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2?
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_9/batchnorm/ReadVariableOp?
#batch_normalization_9/batchnorm/subSub6batch_normalization_9/batchnorm/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub?
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_9/batchnorm/add_1?
re_lu_7/ReluRelu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_7/Relul
reshape_2/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapere_lu_7/Relu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_2/Reshape~
conv1d_transpose_6/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_6/Shape?
&conv1d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_6/strided_slice/stack?
(conv1d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice/stack_1?
(conv1d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice/stack_2?
 conv1d_transpose_6/strided_sliceStridedSlice!conv1d_transpose_6/Shape:output:0/conv1d_transpose_6/strided_slice/stack:output:01conv1d_transpose_6/strided_slice/stack_1:output:01conv1d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_6/strided_slice?
(conv1d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice_1/stack?
*conv1d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_6/strided_slice_1/stack_1?
*conv1d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_6/strided_slice_1/stack_2?
"conv1d_transpose_6/strided_slice_1StridedSlice!conv1d_transpose_6/Shape:output:01conv1d_transpose_6/strided_slice_1/stack:output:03conv1d_transpose_6/strided_slice_1/stack_1:output:03conv1d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_6/strided_slice_1v
conv1d_transpose_6/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_6/mul/y?
conv1d_transpose_6/mulMul+conv1d_transpose_6/strided_slice_1:output:0!conv1d_transpose_6/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_6/mulz
conv1d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_6/stack/2?
conv1d_transpose_6/stackPack)conv1d_transpose_6/strided_slice:output:0conv1d_transpose_6/mul:z:0#conv1d_transpose_6/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_6/stack?
2conv1d_transpose_6/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_6/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_6/conv1d_transpose/ExpandDims
ExpandDimsreshape_2/Reshape:output:0;conv1d_transpose_6/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_6/conv1d_transpose/ExpandDims?
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_6/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_6/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_6/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_6/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_6/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_6/stack:output:0@conv1d_transpose_6/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_6/conv1d_transpose/strided_slice?
9conv1d_transpose_6/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_6/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_6/stack:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_6/conv1d_transpose/strided_slice_1?
3conv1d_transpose_6/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_6/conv1d_transpose/concat/values_1?
/conv1d_transpose_6/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_6/conv1d_transpose/concat/axis?
*conv1d_transpose_6/conv1d_transpose/concatConcatV2:conv1d_transpose_6/conv1d_transpose/strided_slice:output:0<conv1d_transpose_6/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_6/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_6/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_6/conv1d_transpose/concat?
#conv1d_transpose_6/conv1d_transposeConv2DBackpropInput3conv1d_transpose_6/conv1d_transpose/concat:output:09conv1d_transpose_6/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_6/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_6/conv1d_transpose?
+conv1d_transpose_6/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_6/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_6/conv1d_transpose/Squeeze?
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_10/moments/mean/reduction_indices?
#batch_normalization_10/moments/meanMean4conv1d_transpose_6/conv1d_transpose/Squeeze:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_10/moments/mean?
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_10/moments/StopGradient?
0batch_normalization_10/moments/SquaredDifferenceSquaredDifference4conv1d_transpose_6/conv1d_transpose/Squeeze:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_10/moments/SquaredDifference?
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_10/moments/variance/reduction_indices?
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_10/moments/variance?
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze?
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1?
,batch_normalization_10/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_10/AssignMovingAvg/8543439*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_10/AssignMovingAvg/decay?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_10_assignmovingavg_8543439*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp?
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_10/AssignMovingAvg/8543439*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/sub?
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_10/AssignMovingAvg/8543439*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul?
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_10_assignmovingavg_8543439.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_10/AssignMovingAvg/8543439*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_10/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_10/AssignMovingAvg_1/8543445*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_10/AssignMovingAvg_1/decay?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_10_assignmovingavg_1_8543445*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_10/AssignMovingAvg_1/8543445*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/sub?
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_10/AssignMovingAvg_1/8543445*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul?
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_10_assignmovingavg_1_85434450batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_10/AssignMovingAvg_1/8543445*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_10/batchnorm/add/y?
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt?
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Mul4conv1d_transpose_6/conv1d_transpose/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/mul_1?
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2?
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp?
$batch_normalization_10/batchnorm/subSub7batch_normalization_10/batchnorm/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/add_1?
re_lu_8/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_8/Relu~
conv1d_transpose_7/ShapeShapere_lu_8/Relu:activations:0*
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
ExpandDimsre_lu_8/Relu:activations:0;conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_7/conv1d_transpose/ExpandDims?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:22
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
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_11/moments/mean/reduction_indices?
#batch_normalization_11/moments/meanMean4conv1d_transpose_7/conv1d_transpose/Squeeze:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_11/moments/mean?
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_11/moments/StopGradient?
0batch_normalization_11/moments/SquaredDifferenceSquaredDifference4conv1d_transpose_7/conv1d_transpose/Squeeze:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_11/moments/SquaredDifference?
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_11/moments/variance/reduction_indices?
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_11/moments/variance?
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze?
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1?
,batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/8543504*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_11/AssignMovingAvg/decay?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_11_assignmovingavg_8543504*
_output_shapes
:*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp?
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/8543504*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/sub?
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/8543504*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/mul?
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_11_assignmovingavg_8543504.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/8543504*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/8543510*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_11/AssignMovingAvg_1/decay?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_11_assignmovingavg_1_8543510*
_output_shapes
:*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/8543510*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/sub?
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/8543510*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/mul?
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_11_assignmovingavg_1_85435100batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/8543510*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_11/batchnorm/add/y?
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/Rsqrt?
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOp?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Mul4conv1d_transpose_7/conv1d_transpose/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/mul_1?
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/mul_2?
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_11/batchnorm/ReadVariableOp?
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/add_1?
re_lu_9/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_9/Relu~
conv1d_transpose_8/ShapeShapere_lu_9/Relu:activations:0*
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
value	B :2
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
ExpandDimsre_lu_9/Relu:activations:0;conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_8/conv1d_transpose/ExpandDims?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:22
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
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_8/conv1d_transpose?
+conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_8/conv1d_transpose/Squeeze?
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_12/moments/mean/reduction_indices?
#batch_normalization_12/moments/meanMean4conv1d_transpose_8/conv1d_transpose/Squeeze:output:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_12/moments/mean?
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_12/moments/StopGradient?
0batch_normalization_12/moments/SquaredDifferenceSquaredDifference4conv1d_transpose_8/conv1d_transpose/Squeeze:output:04batch_normalization_12/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
0batch_normalization_12/moments/SquaredDifference?
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_12/moments/variance/reduction_indices?
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_12/moments/variance?
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_12/moments/Squeeze?
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1?
,batch_normalization_12/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/8543569*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_12/AssignMovingAvg/decay?
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_12_assignmovingavg_8543569*
_output_shapes
:*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOp?
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/8543569*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/sub?
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/8543569*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/mul?
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_12_assignmovingavg_8543569.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/8543569*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_12/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/8543575*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_12/AssignMovingAvg_1/decay?
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_12_assignmovingavg_1_8543575*
_output_shapes
:*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/8543575*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/sub?
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/8543575*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/mul?
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_12_assignmovingavg_1_85435750batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/8543575*
_output_shapes
 *
dtype02>
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_12/batchnorm/add/y?
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Mul4conv1d_transpose_8/conv1d_transpose/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/mul_1?
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp?
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/add_1?
re_lu_10/ReluRelu*batch_normalization_12/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_10/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_10/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulflatten_4/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMuls
dense_10/TanhTanhdense_10/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_10/Tanh?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_4/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanha
lambda_1/ShapeShapedense_10/Tanh:y:0*
T0*
_output_shapes
:2
lambda_1/Shape
lambda_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_1/random_normal/mean?
lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda_1/random_normal/stddev?
+lambda_1/random_normal/RandomStandardNormalRandomStandardNormallambda_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2-
+lambda_1/random_normal/RandomStandardNormal?
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal/mul?
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normalm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y?
lambda_1/truedivRealDivdense_11/Tanh:y:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Exp?
lambda_1/mulMullambda_1/random_normal:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_1/mul|
lambda_1/addAddV2dense_10/Tanh:y:0lambda_1/mul:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/add?
IdentityIdentitylambda_1/add:z:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_10/batchnorm/ReadVariableOp4^batch_normalization_10/batchnorm/mul/ReadVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_12/AssignMovingAvg/ReadVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp3^batch_normalization_9/batchnorm/mul/ReadVariableOp@^conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2x
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2?
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_8542135

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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_9_layer_call_fn_8544011

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420872
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
?
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_8544327

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
`
D__inference_re_lu_9_layer_call_and_return_conditional_losses_8542839

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
?
?
8__inference_batch_normalization_10_layer_call_fn_8544108

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_85422392
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
8__inference_batch_normalization_12_layer_call_fn_8544292

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_85426092
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
?
p
*__inference_dense_10_layer_call_fn_8544347

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
GPU 2J 8? *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_85429262
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544187

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
?
?
E__inference_dense_11_layer_call_and_return_conditional_losses_8542946

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
?
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_8542788

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
??
?
D__inference_Encoder_layer_call_and_return_conditional_losses_8543817

inputs*
&dense_9_matmul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resource?
;batch_normalization_9_batchnorm_mul_readvariableop_resource=
9batch_normalization_9_batchnorm_readvariableop_1_resource=
9batch_normalization_9_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource@
<batch_normalization_10_batchnorm_mul_readvariableop_resource>
:batch_normalization_10_batchnorm_readvariableop_1_resource>
:batch_normalization_10_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource@
<batch_normalization_11_batchnorm_mul_readvariableop_resource>
:batch_normalization_11_batchnorm_readvariableop_1_resource>
:batch_normalization_11_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource@
<batch_normalization_12_batchnorm_mul_readvariableop_resource>
:batch_normalization_12_batchnorm_readvariableop_1_resource>
:batch_normalization_12_batchnorm_readvariableop_2_resource+
'dense_10_matmul_readvariableop_resource+
'dense_11_matmul_readvariableop_resource
identity??/batch_normalization_10/batchnorm/ReadVariableOp?1batch_normalization_10/batchnorm/ReadVariableOp_1?1batch_normalization_10/batchnorm/ReadVariableOp_2?3batch_normalization_10/batchnorm/mul/ReadVariableOp?/batch_normalization_11/batchnorm/ReadVariableOp?1batch_normalization_11/batchnorm/ReadVariableOp_1?1batch_normalization_11/batchnorm/ReadVariableOp_2?3batch_normalization_11/batchnorm/mul/ReadVariableOp?/batch_normalization_12/batchnorm/ReadVariableOp?1batch_normalization_12/batchnorm/ReadVariableOp_1?1batch_normalization_12/batchnorm/ReadVariableOp_2?3batch_normalization_12/batchnorm/mul/ReadVariableOp?.batch_normalization_9/batchnorm/ReadVariableOp?0batch_normalization_9/batchnorm/ReadVariableOp_1?0batch_normalization_9/batchnorm/ReadVariableOp_2?2batch_normalization_9/batchnorm/mul/ReadVariableOp??conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp??conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_9/batchnorm/ReadVariableOp?
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_9/batchnorm/add/y?
#batch_normalization_9/batchnorm/addAddV26batch_normalization_9/batchnorm/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add?
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt?
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOp?
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul?
%batch_normalization_9/batchnorm/mul_1Muldense_9/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_9/batchnorm/mul_1?
0batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_1?
%batch_normalization_9/batchnorm/mul_2Mul8batch_normalization_9/batchnorm/ReadVariableOp_1:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2?
0batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_2?
#batch_normalization_9/batchnorm/subSub8batch_normalization_9/batchnorm/ReadVariableOp_2:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub?
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_9/batchnorm/add_1?
re_lu_7/ReluRelu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_7/Relul
reshape_2/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapere_lu_7/Relu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_2/Reshape~
conv1d_transpose_6/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_6/Shape?
&conv1d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_6/strided_slice/stack?
(conv1d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice/stack_1?
(conv1d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice/stack_2?
 conv1d_transpose_6/strided_sliceStridedSlice!conv1d_transpose_6/Shape:output:0/conv1d_transpose_6/strided_slice/stack:output:01conv1d_transpose_6/strided_slice/stack_1:output:01conv1d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_6/strided_slice?
(conv1d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_6/strided_slice_1/stack?
*conv1d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_6/strided_slice_1/stack_1?
*conv1d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_6/strided_slice_1/stack_2?
"conv1d_transpose_6/strided_slice_1StridedSlice!conv1d_transpose_6/Shape:output:01conv1d_transpose_6/strided_slice_1/stack:output:03conv1d_transpose_6/strided_slice_1/stack_1:output:03conv1d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_6/strided_slice_1v
conv1d_transpose_6/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_6/mul/y?
conv1d_transpose_6/mulMul+conv1d_transpose_6/strided_slice_1:output:0!conv1d_transpose_6/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_6/mulz
conv1d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_6/stack/2?
conv1d_transpose_6/stackPack)conv1d_transpose_6/strided_slice:output:0conv1d_transpose_6/mul:z:0#conv1d_transpose_6/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_6/stack?
2conv1d_transpose_6/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_6/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_6/conv1d_transpose/ExpandDims
ExpandDimsreshape_2/Reshape:output:0;conv1d_transpose_6/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_6/conv1d_transpose/ExpandDims?
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_6/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_6/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_6/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_6/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_6/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_6/stack:output:0@conv1d_transpose_6/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_6/conv1d_transpose/strided_slice?
9conv1d_transpose_6/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_6/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_6/stack:output:0Bconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_6/conv1d_transpose/strided_slice_1?
3conv1d_transpose_6/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_6/conv1d_transpose/concat/values_1?
/conv1d_transpose_6/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_6/conv1d_transpose/concat/axis?
*conv1d_transpose_6/conv1d_transpose/concatConcatV2:conv1d_transpose_6/conv1d_transpose/strided_slice:output:0<conv1d_transpose_6/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_6/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_6/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_6/conv1d_transpose/concat?
#conv1d_transpose_6/conv1d_transposeConv2DBackpropInput3conv1d_transpose_6/conv1d_transpose/concat:output:09conv1d_transpose_6/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_6/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_6/conv1d_transpose?
+conv1d_transpose_6/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_6/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_6/conv1d_transpose/Squeeze?
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp?
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_10/batchnorm/add/y?
$batch_normalization_10/batchnorm/addAddV27batch_normalization_10/batchnorm/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt?
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Mul4conv1d_transpose_6/conv1d_transpose/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/mul_1?
1batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_1?
&batch_normalization_10/batchnorm/mul_2Mul9batch_normalization_10/batchnorm/ReadVariableOp_1:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2?
1batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_2?
$batch_normalization_10/batchnorm/subSub9batch_normalization_10/batchnorm/ReadVariableOp_2:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/add_1?
re_lu_8/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_8/Relu~
conv1d_transpose_7/ShapeShapere_lu_8/Relu:activations:0*
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
ExpandDimsre_lu_8/Relu:activations:0;conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_7/conv1d_transpose/ExpandDims?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:22
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
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_11/batchnorm/ReadVariableOp?
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_11/batchnorm/add/y?
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/Rsqrt?
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOp?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Mul4conv1d_transpose_7/conv1d_transpose/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/mul_1?
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_1?
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/mul_2?
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_2?
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/add_1?
re_lu_9/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_9/Relu~
conv1d_transpose_8/ShapeShapere_lu_9/Relu:activations:0*
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
value	B :2
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
ExpandDimsre_lu_9/Relu:activations:0;conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_8/conv1d_transpose/ExpandDims?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:22
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
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_8/conv1d_transpose?
+conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_8/conv1d_transpose/Squeeze?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp?
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_12/batchnorm/add/y?
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Mul4conv1d_transpose_8/conv1d_transpose/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/mul_1?
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1?
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2?
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2?
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_12/batchnorm/add_1?
re_lu_10/ReluRelu*batch_normalization_12/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_10/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_10/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulflatten_4/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_10/MatMuls
dense_10/TanhTanhdense_10/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_10/Tanh?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulflatten_4/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanha
lambda_1/ShapeShapedense_10/Tanh:y:0*
T0*
_output_shapes
:2
lambda_1/Shape
lambda_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_1/random_normal/mean?
lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lambda_1/random_normal/stddev?
+lambda_1/random_normal/RandomStandardNormalRandomStandardNormallambda_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???2-
+lambda_1/random_normal/RandomStandardNormal?
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal/mul?
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normalm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y?
lambda_1/truedivRealDivdense_11/Tanh:y:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Exp?
lambda_1/mulMullambda_1/random_normal:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_1/mul|
lambda_1/addAddV2dense_10/Tanh:y:0lambda_1/mul:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/add?	
IdentityIdentitylambda_1/add:z:00^batch_normalization_10/batchnorm/ReadVariableOp2^batch_normalization_10/batchnorm/ReadVariableOp_12^batch_normalization_10/batchnorm/ReadVariableOp_24^batch_normalization_10/batchnorm/mul/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp1^batch_normalization_9/batchnorm/ReadVariableOp_11^batch_normalization_9/batchnorm/ReadVariableOp_23^batch_normalization_9/batchnorm/mul/ReadVariableOp@^conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2f
1batch_normalization_10/batchnorm/ReadVariableOp_11batch_normalization_10/batchnorm/ReadVariableOp_12f
1batch_normalization_10/batchnorm/ReadVariableOp_21batch_normalization_10/batchnorm/ReadVariableOp_22j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2d
0batch_normalization_9/batchnorm/ReadVariableOp_10batch_normalization_9/batchnorm/ReadVariableOp_12d
0batch_normalization_9/batchnorm/ReadVariableOp_20batch_normalization_9/batchnorm/ReadVariableOp_22h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2?
?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?

D__inference_Encoder_layer_call_and_return_conditional_losses_8543006
input_5
dense_9_8542673!
batch_normalization_9_8542702!
batch_normalization_9_8542704!
batch_normalization_9_8542706!
batch_normalization_9_8542708
conv1d_transpose_6_8542745"
batch_normalization_10_8542774"
batch_normalization_10_8542776"
batch_normalization_10_8542778"
batch_normalization_10_8542780
conv1d_transpose_7_8542796"
batch_normalization_11_8542825"
batch_normalization_11_8542827"
batch_normalization_11_8542829"
batch_normalization_11_8542831
conv1d_transpose_8_8542847"
batch_normalization_12_8542876"
batch_normalization_12_8542878"
batch_normalization_12_8542880"
batch_normalization_12_8542882
dense_10_8542935
dense_11_8542955
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_6/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_9_8542673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_85426642!
dense_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_9_8542702batch_normalization_9_8542704batch_normalization_9_8542706batch_normalization_9_8542708*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420542/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_85427162
re_lu_7/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_85427372
reshape_2/PartitionedCall?
*conv1d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_transpose_6_8542745*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_85421352,
*conv1d_transpose_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_10_8542774batch_normalization_10_8542776batch_normalization_10_8542778batch_normalization_10_8542780*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_854223920
.batch_normalization_10/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_85427882
re_lu_8/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv1d_transpose_7_8542796*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_85423202,
*conv1d_transpose_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_11_8542825batch_normalization_11_8542827batch_normalization_11_8542829batch_normalization_11_8542831*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_854242420
.batch_normalization_11/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_85428392
re_lu_9/PartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv1d_transpose_8_8542847*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_85425052,
*conv1d_transpose_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_12_8542876batch_normalization_12_8542878batch_normalization_12_8542880batch_normalization_12_8542882*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_854260920
.batch_normalization_12/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_85428902
re_lu_10/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall!re_lu_10/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_85429102
flatten_4/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_10_8542935*
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
GPU 2J 8? *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_85429262"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_11_8542955*
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
GPU 2J 8? *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_85429462"
 dense_11/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0)dense_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429742"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_6/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_6/StatefulPartitionedCall*conv1d_transpose_6/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
E__inference_dense_11_layer_call_and_return_conditional_losses_8544355

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
?U
?

D__inference_Encoder_layer_call_and_return_conditional_losses_8543139

inputs
dense_9_8543077!
batch_normalization_9_8543080!
batch_normalization_9_8543082!
batch_normalization_9_8543084!
batch_normalization_9_8543086
conv1d_transpose_6_8543091"
batch_normalization_10_8543094"
batch_normalization_10_8543096"
batch_normalization_10_8543098"
batch_normalization_10_8543100
conv1d_transpose_7_8543104"
batch_normalization_11_8543107"
batch_normalization_11_8543109"
batch_normalization_11_8543111"
batch_normalization_11_8543113
conv1d_transpose_8_8543117"
batch_normalization_12_8543120"
batch_normalization_12_8543122"
batch_normalization_12_8543124"
batch_normalization_12_8543126
dense_10_8543131
dense_11_8543134
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_6/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_8543077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_85426642!
dense_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_9_8543080batch_normalization_9_8543082batch_normalization_9_8543084batch_normalization_9_8543086*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420542/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_85427162
re_lu_7/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_85427372
reshape_2/PartitionedCall?
*conv1d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_transpose_6_8543091*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_85421352,
*conv1d_transpose_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_10_8543094batch_normalization_10_8543096batch_normalization_10_8543098batch_normalization_10_8543100*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_854223920
.batch_normalization_10/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_85427882
re_lu_8/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv1d_transpose_7_8543104*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_85423202,
*conv1d_transpose_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_11_8543107batch_normalization_11_8543109batch_normalization_11_8543111batch_normalization_11_8543113*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_854242420
.batch_normalization_11/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_85428392
re_lu_9/PartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv1d_transpose_8_8543117*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_85425052,
*conv1d_transpose_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_12_8543120batch_normalization_12_8543122batch_normalization_12_8543124batch_normalization_12_8543126*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_854260920
.batch_normalization_12/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_85428902
re_lu_10/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall!re_lu_10/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_85429102
flatten_4/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_10_8543131*
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
GPU 2J 8? *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_85429262"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_11_8543134*
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
GPU 2J 8? *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_85429462"
 dense_11/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0)dense_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429742"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_6/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_6/StatefulPartitionedCall*conv1d_transpose_6/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8542609

inputs
assignmovingavg_8542584
assignmovingavg_1_8542590)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542584*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8542584*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542584*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542584*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8542584AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542584*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542590*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8542590*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542590*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542590*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8542590AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542590*
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
?
?
)__inference_Encoder_layer_call_fn_8543915

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

unknown_20
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_85432532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_8542716

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_4_layer_call_fn_8544332

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
GPU 2J 8? *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_85429102
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
?
?
)__inference_Encoder_layer_call_fn_8543186
input_5
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_85431392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?1
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8542424

inputs
assignmovingavg_8542399
assignmovingavg_1_8542405)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542399*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8542399*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542399*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542399*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8542399AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542399*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542405*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8542405*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542405*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542405*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8542405AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542405*
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
?
s
*__inference_lambda_1_layer_call_fn_8544406
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429902
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
?0
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543965

inputs
assignmovingavg_8543940
assignmovingavg_1_8543946)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8543940*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8543940*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8543940*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8543940*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8543940AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8543940*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8543946*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8543946*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8543946*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8543946*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8543946AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8543946*
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
t
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544394
inputs_0
inputs_1
identity?F
ShapeShapeinputs_0*
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
seed2??M2$
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
random_normal[
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
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulX
addAddV2inputs_0mul:z:0*
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
?
`
D__inference_re_lu_9_layer_call_and_return_conditional_losses_8544218

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
?
r
E__inference_lambda_1_layer_call_and_return_conditional_losses_8542974

inputs
inputs_1
identity?D
ShapeShapeinputs*
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
seed2?כ2$
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
random_normal[
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
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulV
addAddV2inputsmul:z:0*
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
?
E
)__inference_re_lu_8_layer_call_fn_8544131

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
GPU 2J 8? *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_85427882
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
?0
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8542054

inputs
assignmovingavg_8542029
assignmovingavg_1_8542035)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542029*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8542029*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542029*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542029*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8542029AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542029*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542035*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8542035*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542035*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542035*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8542035AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542035*
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
?
?
D__inference_dense_9_layer_call_and_return_conditional_losses_8542664

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_8542505

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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_8544034

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
value	B :2
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
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_11_layer_call_fn_8544200

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_85424242
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
?
?
%__inference_signature_wrapper_8543351
input_5
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_85419582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?U
?

D__inference_Encoder_layer_call_and_return_conditional_losses_8543071
input_5
dense_9_8543009!
batch_normalization_9_8543012!
batch_normalization_9_8543014!
batch_normalization_9_8543016!
batch_normalization_9_8543018
conv1d_transpose_6_8543023"
batch_normalization_10_8543026"
batch_normalization_10_8543028"
batch_normalization_10_8543030"
batch_normalization_10_8543032
conv1d_transpose_7_8543036"
batch_normalization_11_8543039"
batch_normalization_11_8543041"
batch_normalization_11_8543043"
batch_normalization_11_8543045
conv1d_transpose_8_8543049"
batch_normalization_12_8543052"
batch_normalization_12_8543054"
batch_normalization_12_8543056"
batch_normalization_12_8543058
dense_10_8543063
dense_11_8543066
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_6/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_9_8543009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_85426642!
dense_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_9_8543012batch_normalization_9_8543014batch_normalization_9_8543016batch_normalization_9_8543018*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420872/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_85427162
re_lu_7/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_85427372
reshape_2/PartitionedCall?
*conv1d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_transpose_6_8543023*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_85421352,
*conv1d_transpose_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_10_8543026batch_normalization_10_8543028batch_normalization_10_8543030batch_normalization_10_8543032*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_854227220
.batch_normalization_10/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_85427882
re_lu_8/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv1d_transpose_7_8543036*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_85423202,
*conv1d_transpose_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_11_8543039batch_normalization_11_8543041batch_normalization_11_8543043batch_normalization_11_8543045*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_854245720
.batch_normalization_11/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_85428392
re_lu_9/PartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv1d_transpose_8_8543049*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_85425052,
*conv1d_transpose_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_12_8543052batch_normalization_12_8543054batch_normalization_12_8543056batch_normalization_12_8543058*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_854264220
.batch_normalization_12/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_85428902
re_lu_10/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall!re_lu_10/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_85429102
flatten_4/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_10_8543063*
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
GPU 2J 8? *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_85429262"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_11_8543066*
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
GPU 2J 8? *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_85429462"
 dense_11/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0)dense_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429902"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_6/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_6/StatefulPartitionedCall*conv1d_transpose_6/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
z
4__inference_conv1d_transpose_8_layer_call_fn_8542513

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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_85425052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_8542890

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
?U
?

D__inference_Encoder_layer_call_and_return_conditional_losses_8543253

inputs
dense_9_8543191!
batch_normalization_9_8543194!
batch_normalization_9_8543196!
batch_normalization_9_8543198!
batch_normalization_9_8543200
conv1d_transpose_6_8543205"
batch_normalization_10_8543208"
batch_normalization_10_8543210"
batch_normalization_10_8543212"
batch_normalization_10_8543214
conv1d_transpose_7_8543218"
batch_normalization_11_8543221"
batch_normalization_11_8543223"
batch_normalization_11_8543225"
batch_normalization_11_8543227
conv1d_transpose_8_8543231"
batch_normalization_12_8543234"
batch_normalization_12_8543236"
batch_normalization_12_8543238"
batch_normalization_12_8543240
dense_10_8543245
dense_11_8543248
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?.batch_normalization_12/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_6/StatefulPartitionedCall?*conv1d_transpose_7/StatefulPartitionedCall?*conv1d_transpose_8/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_8543191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_85426642!
dense_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_9_8543194batch_normalization_9_8543196batch_normalization_9_8543198batch_normalization_9_8543200*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420872/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_85427162
re_lu_7/PartitionedCall?
reshape_2/PartitionedCallPartitionedCall re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_85427372
reshape_2/PartitionedCall?
*conv1d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_transpose_6_8543205*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_85421352,
*conv1d_transpose_6/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_6/StatefulPartitionedCall:output:0batch_normalization_10_8543208batch_normalization_10_8543210batch_normalization_10_8543212batch_normalization_10_8543214*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_854227220
.batch_normalization_10/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_85427882
re_lu_8/PartitionedCall?
*conv1d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv1d_transpose_7_8543218*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_85423202,
*conv1d_transpose_7/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_7/StatefulPartitionedCall:output:0batch_normalization_11_8543221batch_normalization_11_8543223batch_normalization_11_8543225batch_normalization_11_8543227*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_854245720
.batch_normalization_11/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_85428392
re_lu_9/PartitionedCall?
*conv1d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0conv1d_transpose_8_8543231*
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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_85425052,
*conv1d_transpose_8/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_8/StatefulPartitionedCall:output:0batch_normalization_12_8543234batch_normalization_12_8543236batch_normalization_12_8543238batch_normalization_12_8543240*
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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_854264220
.batch_normalization_12/StatefulPartitionedCall?
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_85428902
re_lu_10/PartitionedCall?
flatten_4/PartitionedCallPartitionedCall!re_lu_10/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_85429102
flatten_4/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_10_8543245*
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
GPU 2J 8? *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_85429262"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_11_8543248*
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
GPU 2J 8? *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_85429462"
 dense_11/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0)dense_11/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429902"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_6/StatefulPartitionedCall+^conv1d_transpose_7/StatefulPartitionedCall+^conv1d_transpose_8/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_6/StatefulPartitionedCall*conv1d_transpose_6/StatefulPartitionedCall2X
*conv1d_transpose_7/StatefulPartitionedCall*conv1d_transpose_7/StatefulPartitionedCall2X
*conv1d_transpose_8/StatefulPartitionedCall*conv1d_transpose_8/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544279

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
?
o
)__inference_dense_9_layer_call_fn_8543929

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_85426642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_8544310

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
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543985

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
?
z
4__inference_conv1d_transpose_6_layer_call_fn_8542143

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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_85421352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?9
?
 __inference__traced_save_8544495
file_prefix-
)savev2_dense_9_kernel_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_6_kernel_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_7_kernel_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_8_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop
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
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop4savev2_conv1d_transpose_6_kernel_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop4savev2_conv1d_transpose_7_kernel_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop4savev2_conv1d_transpose_8_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop*savev2_dense_11_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
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
?: ::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
?
?
8__inference_batch_normalization_12_layer_call_fn_8544305

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_85426422
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
?
?
D__inference_dense_9_layer_call_and_return_conditional_losses_8543922

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544167

inputs
assignmovingavg_8544142
assignmovingavg_1_8544148)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544142*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8544142*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544142*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544142*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8544142AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544142*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544148*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8544148*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544148*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544148*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8544148AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544148*
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
?
F
*__inference_re_lu_10_layer_call_fn_8544315

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
GPU 2J 8? *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_85428902
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
?
t
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544378
inputs_0
inputs_1
identity?F
ShapeShapeinputs_0*
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
seed2?ì2$
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
random_normal[
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
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulX
addAddV2inputs_0mul:z:0*
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8542457

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8542272

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
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_8542737

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
value	B :2
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
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_Encoder_layer_call_fn_8543300
input_5
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

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_85432532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
E
)__inference_re_lu_9_layer_call_fn_8544223

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
GPU 2J 8? *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_85428392
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
?
?
E__inference_dense_10_layer_call_and_return_conditional_losses_8544340

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
7__inference_batch_normalization_9_layer_call_fn_8543998

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_85420542
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
?
G
+__inference_reshape_2_layer_call_fn_8544039

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_85427372
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_8542910

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
?
p
*__inference_dense_11_layer_call_fn_8544362

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
GPU 2J 8? *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_85429462
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
?1
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8542239

inputs
assignmovingavg_8542214
assignmovingavg_1_8542220)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542214*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8542214*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542214*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8542214*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8542214AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8542214*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542220*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8542220*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542220*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8542220*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8542220AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8542220*
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
?
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_8544016

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
*__inference_lambda_1_layer_call_fn_8544400
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
GPU 2J 8? *N
fIRG
E__inference_lambda_1_layer_call_and_return_conditional_losses_85429742
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8542087

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
?1
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544075

inputs
assignmovingavg_8544050
assignmovingavg_1_8544056)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544050*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8544050*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544050*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544050*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8544050AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544050*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544056*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8544056*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544056*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544056*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8544056AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544056*
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
?
?
8__inference_batch_normalization_11_layer_call_fn_8544213

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_85424572
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
?-
?
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_8542320

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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:??????????????????:2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_8544126

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
8__inference_batch_normalization_10_layer_call_fn_8544121

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
GPU 2J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_85422722
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
?
?
)__inference_Encoder_layer_call_fn_8543866

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

unknown_20
identity??StatefulPartitionedCall?
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_85431392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_8541958
input_52
.encoder_dense_9_matmul_readvariableop_resourceC
?encoder_batch_normalization_9_batchnorm_readvariableop_resourceG
Cencoder_batch_normalization_9_batchnorm_mul_readvariableop_resourceE
Aencoder_batch_normalization_9_batchnorm_readvariableop_1_resourceE
Aencoder_batch_normalization_9_batchnorm_readvariableop_2_resourceT
Pencoder_conv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resourceD
@encoder_batch_normalization_10_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_10_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_10_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_10_batchnorm_readvariableop_2_resourceT
Pencoder_conv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resourceD
@encoder_batch_normalization_11_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_11_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_11_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_11_batchnorm_readvariableop_2_resourceT
Pencoder_conv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resourceD
@encoder_batch_normalization_12_batchnorm_readvariableop_resourceH
Dencoder_batch_normalization_12_batchnorm_mul_readvariableop_resourceF
Bencoder_batch_normalization_12_batchnorm_readvariableop_1_resourceF
Bencoder_batch_normalization_12_batchnorm_readvariableop_2_resource3
/encoder_dense_10_matmul_readvariableop_resource3
/encoder_dense_11_matmul_readvariableop_resource
identity??7Encoder/batch_normalization_10/batchnorm/ReadVariableOp?9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_11/batchnorm/ReadVariableOp?9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOp?7Encoder/batch_normalization_12/batchnorm/ReadVariableOp?9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_1?9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_2?;Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOp?6Encoder/batch_normalization_9/batchnorm/ReadVariableOp?8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_1?8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_2?:Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOp?GEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?GEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?GEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?&Encoder/dense_10/MatMul/ReadVariableOp?&Encoder/dense_11/MatMul/ReadVariableOp?%Encoder/dense_9/MatMul/ReadVariableOp?
%Encoder/dense_9/MatMul/ReadVariableOpReadVariableOp.encoder_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%Encoder/dense_9/MatMul/ReadVariableOp?
Encoder/dense_9/MatMulMatMulinput_5-Encoder/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_9/MatMul?
6Encoder/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp?encoder_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6Encoder/batch_normalization_9/batchnorm/ReadVariableOp?
-Encoder/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2/
-Encoder/batch_normalization_9/batchnorm/add/y?
+Encoder/batch_normalization_9/batchnorm/addAddV2>Encoder/batch_normalization_9/batchnorm/ReadVariableOp:value:06Encoder/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+Encoder/batch_normalization_9/batchnorm/add?
-Encoder/batch_normalization_9/batchnorm/RsqrtRsqrt/Encoder/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-Encoder/batch_normalization_9/batchnorm/Rsqrt?
:Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpCencoder_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOp?
+Encoder/batch_normalization_9/batchnorm/mulMul1Encoder/batch_normalization_9/batchnorm/Rsqrt:y:0BEncoder/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+Encoder/batch_normalization_9/batchnorm/mul?
-Encoder/batch_normalization_9/batchnorm/mul_1Mul Encoder/dense_9/MatMul:product:0/Encoder/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2/
-Encoder/batch_normalization_9/batchnorm/mul_1?
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpAencoder_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_1?
-Encoder/batch_normalization_9/batchnorm/mul_2Mul@Encoder/batch_normalization_9/batchnorm/ReadVariableOp_1:value:0/Encoder/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-Encoder/batch_normalization_9/batchnorm/mul_2?
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpAencoder_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_2?
+Encoder/batch_normalization_9/batchnorm/subSub@Encoder/batch_normalization_9/batchnorm/ReadVariableOp_2:value:01Encoder/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+Encoder/batch_normalization_9/batchnorm/sub?
-Encoder/batch_normalization_9/batchnorm/add_1AddV21Encoder/batch_normalization_9/batchnorm/mul_1:z:0/Encoder/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2/
-Encoder/batch_normalization_9/batchnorm/add_1?
Encoder/re_lu_7/ReluRelu1Encoder/batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Encoder/re_lu_7/Relu?
Encoder/reshape_2/ShapeShape"Encoder/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
Encoder/reshape_2/Shape?
%Encoder/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Encoder/reshape_2/strided_slice/stack?
'Encoder/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Encoder/reshape_2/strided_slice/stack_1?
'Encoder/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Encoder/reshape_2/strided_slice/stack_2?
Encoder/reshape_2/strided_sliceStridedSlice Encoder/reshape_2/Shape:output:0.Encoder/reshape_2/strided_slice/stack:output:00Encoder/reshape_2/strided_slice/stack_1:output:00Encoder/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Encoder/reshape_2/strided_slice?
!Encoder/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!Encoder/reshape_2/Reshape/shape/1?
!Encoder/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!Encoder/reshape_2/Reshape/shape/2?
Encoder/reshape_2/Reshape/shapePack(Encoder/reshape_2/strided_slice:output:0*Encoder/reshape_2/Reshape/shape/1:output:0*Encoder/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2!
Encoder/reshape_2/Reshape/shape?
Encoder/reshape_2/ReshapeReshape"Encoder/re_lu_7/Relu:activations:0(Encoder/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
Encoder/reshape_2/Reshape?
 Encoder/conv1d_transpose_6/ShapeShape"Encoder/reshape_2/Reshape:output:0*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_6/Shape?
.Encoder/conv1d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Encoder/conv1d_transpose_6/strided_slice/stack?
0Encoder/conv1d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_6/strided_slice/stack_1?
0Encoder/conv1d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_6/strided_slice/stack_2?
(Encoder/conv1d_transpose_6/strided_sliceStridedSlice)Encoder/conv1d_transpose_6/Shape:output:07Encoder/conv1d_transpose_6/strided_slice/stack:output:09Encoder/conv1d_transpose_6/strided_slice/stack_1:output:09Encoder/conv1d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Encoder/conv1d_transpose_6/strided_slice?
0Encoder/conv1d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_6/strided_slice_1/stack?
2Encoder/conv1d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_6/strided_slice_1/stack_1?
2Encoder/conv1d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_6/strided_slice_1/stack_2?
*Encoder/conv1d_transpose_6/strided_slice_1StridedSlice)Encoder/conv1d_transpose_6/Shape:output:09Encoder/conv1d_transpose_6/strided_slice_1/stack:output:0;Encoder/conv1d_transpose_6/strided_slice_1/stack_1:output:0;Encoder/conv1d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Encoder/conv1d_transpose_6/strided_slice_1?
 Encoder/conv1d_transpose_6/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Encoder/conv1d_transpose_6/mul/y?
Encoder/conv1d_transpose_6/mulMul3Encoder/conv1d_transpose_6/strided_slice_1:output:0)Encoder/conv1d_transpose_6/mul/y:output:0*
T0*
_output_shapes
: 2 
Encoder/conv1d_transpose_6/mul?
"Encoder/conv1d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Encoder/conv1d_transpose_6/stack/2?
 Encoder/conv1d_transpose_6/stackPack1Encoder/conv1d_transpose_6/strided_slice:output:0"Encoder/conv1d_transpose_6/mul:z:0+Encoder/conv1d_transpose_6/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_6/stack?
:Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims/dim?
6Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims
ExpandDims"Encoder/reshape_2/Reshape:output:0CEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims?
GEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPencoder_conv1d_transpose_6_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim?
8Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1
ExpandDimsOEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1?
?Encoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Encoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack?
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1?
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2?
9Encoder/conv1d_transpose_6/conv1d_transpose/strided_sliceStridedSlice)Encoder/conv1d_transpose_6/stack:output:0HEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack:output:0JEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_1:output:0JEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Encoder/conv1d_transpose_6/conv1d_transpose/strided_slice?
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack?
CEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1?
CEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2?
;Encoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1StridedSlice)Encoder/conv1d_transpose_6/stack:output:0JEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack:output:0LEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_1:output:0LEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Encoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1?
;Encoder/conv1d_transpose_6/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Encoder/conv1d_transpose_6/conv1d_transpose/concat/values_1?
7Encoder/conv1d_transpose_6/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Encoder/conv1d_transpose_6/conv1d_transpose/concat/axis?
2Encoder/conv1d_transpose_6/conv1d_transpose/concatConcatV2BEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice:output:0DEncoder/conv1d_transpose_6/conv1d_transpose/concat/values_1:output:0DEncoder/conv1d_transpose_6/conv1d_transpose/strided_slice_1:output:0@Encoder/conv1d_transpose_6/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Encoder/conv1d_transpose_6/conv1d_transpose/concat?
+Encoder/conv1d_transpose_6/conv1d_transposeConv2DBackpropInput;Encoder/conv1d_transpose_6/conv1d_transpose/concat:output:0AEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1:output:0?Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Encoder/conv1d_transpose_6/conv1d_transpose?
3Encoder/conv1d_transpose_6/conv1d_transpose/SqueezeSqueeze4Encoder/conv1d_transpose_6/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Encoder/conv1d_transpose_6/conv1d_transpose/Squeeze?
7Encoder/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_10/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_10/batchnorm/add/y?
,Encoder/batch_normalization_10/batchnorm/addAddV2?Encoder/batch_normalization_10/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_10/batchnorm/add?
.Encoder/batch_normalization_10/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_10/batchnorm/Rsqrt?
;Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_10/batchnorm/mulMul2Encoder/batch_normalization_10/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_10/batchnorm/mul?
.Encoder/batch_normalization_10/batchnorm/mul_1Mul<Encoder/conv1d_transpose_6/conv1d_transpose/Squeeze:output:00Encoder/batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_10/batchnorm/mul_1?
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_10/batchnorm/mul_2MulAEncoder/batch_normalization_10/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_10/batchnorm/mul_2?
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_10/batchnorm/subSubAEncoder/batch_normalization_10/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_10/batchnorm/sub?
.Encoder/batch_normalization_10/batchnorm/add_1AddV22Encoder/batch_normalization_10/batchnorm/mul_1:z:00Encoder/batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_10/batchnorm/add_1?
Encoder/re_lu_8/ReluRelu2Encoder/batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
Encoder/re_lu_8/Relu?
 Encoder/conv1d_transpose_7/ShapeShape"Encoder/re_lu_8/Relu:activations:0*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_7/Shape?
.Encoder/conv1d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Encoder/conv1d_transpose_7/strided_slice/stack?
0Encoder/conv1d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_7/strided_slice/stack_1?
0Encoder/conv1d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_7/strided_slice/stack_2?
(Encoder/conv1d_transpose_7/strided_sliceStridedSlice)Encoder/conv1d_transpose_7/Shape:output:07Encoder/conv1d_transpose_7/strided_slice/stack:output:09Encoder/conv1d_transpose_7/strided_slice/stack_1:output:09Encoder/conv1d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Encoder/conv1d_transpose_7/strided_slice?
0Encoder/conv1d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_7/strided_slice_1/stack?
2Encoder/conv1d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_7/strided_slice_1/stack_1?
2Encoder/conv1d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_7/strided_slice_1/stack_2?
*Encoder/conv1d_transpose_7/strided_slice_1StridedSlice)Encoder/conv1d_transpose_7/Shape:output:09Encoder/conv1d_transpose_7/strided_slice_1/stack:output:0;Encoder/conv1d_transpose_7/strided_slice_1/stack_1:output:0;Encoder/conv1d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Encoder/conv1d_transpose_7/strided_slice_1?
 Encoder/conv1d_transpose_7/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Encoder/conv1d_transpose_7/mul/y?
Encoder/conv1d_transpose_7/mulMul3Encoder/conv1d_transpose_7/strided_slice_1:output:0)Encoder/conv1d_transpose_7/mul/y:output:0*
T0*
_output_shapes
: 2 
Encoder/conv1d_transpose_7/mul?
"Encoder/conv1d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Encoder/conv1d_transpose_7/stack/2?
 Encoder/conv1d_transpose_7/stackPack1Encoder/conv1d_transpose_7/strided_slice:output:0"Encoder/conv1d_transpose_7/mul:z:0+Encoder/conv1d_transpose_7/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_7/stack?
:Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dim?
6Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims
ExpandDims"Encoder/re_lu_8/Relu:activations:0CEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims?
GEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPencoder_conv1d_transpose_7_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim?
8Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1
ExpandDimsOEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1?
?Encoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Encoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack?
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1?
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2?
9Encoder/conv1d_transpose_7/conv1d_transpose/strided_sliceStridedSlice)Encoder/conv1d_transpose_7/stack:output:0HEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack:output:0JEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_1:output:0JEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Encoder/conv1d_transpose_7/conv1d_transpose/strided_slice?
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack?
CEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1?
CEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2?
;Encoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1StridedSlice)Encoder/conv1d_transpose_7/stack:output:0JEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack:output:0LEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_1:output:0LEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Encoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1?
;Encoder/conv1d_transpose_7/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Encoder/conv1d_transpose_7/conv1d_transpose/concat/values_1?
7Encoder/conv1d_transpose_7/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Encoder/conv1d_transpose_7/conv1d_transpose/concat/axis?
2Encoder/conv1d_transpose_7/conv1d_transpose/concatConcatV2BEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice:output:0DEncoder/conv1d_transpose_7/conv1d_transpose/concat/values_1:output:0DEncoder/conv1d_transpose_7/conv1d_transpose/strided_slice_1:output:0@Encoder/conv1d_transpose_7/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Encoder/conv1d_transpose_7/conv1d_transpose/concat?
+Encoder/conv1d_transpose_7/conv1d_transposeConv2DBackpropInput;Encoder/conv1d_transpose_7/conv1d_transpose/concat:output:0AEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1:output:0?Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Encoder/conv1d_transpose_7/conv1d_transpose?
3Encoder/conv1d_transpose_7/conv1d_transpose/SqueezeSqueeze4Encoder/conv1d_transpose_7/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Encoder/conv1d_transpose_7/conv1d_transpose/Squeeze?
7Encoder/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_11/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_11/batchnorm/add/y?
,Encoder/batch_normalization_11/batchnorm/addAddV2?Encoder/batch_normalization_11/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_11/batchnorm/add?
.Encoder/batch_normalization_11/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_11/batchnorm/Rsqrt?
;Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_11/batchnorm/mulMul2Encoder/batch_normalization_11/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_11/batchnorm/mul?
.Encoder/batch_normalization_11/batchnorm/mul_1Mul<Encoder/conv1d_transpose_7/conv1d_transpose/Squeeze:output:00Encoder/batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_11/batchnorm/mul_1?
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_11/batchnorm/mul_2MulAEncoder/batch_normalization_11/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_11/batchnorm/mul_2?
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_11/batchnorm/subSubAEncoder/batch_normalization_11/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_11/batchnorm/sub?
.Encoder/batch_normalization_11/batchnorm/add_1AddV22Encoder/batch_normalization_11/batchnorm/mul_1:z:00Encoder/batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_11/batchnorm/add_1?
Encoder/re_lu_9/ReluRelu2Encoder/batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
Encoder/re_lu_9/Relu?
 Encoder/conv1d_transpose_8/ShapeShape"Encoder/re_lu_9/Relu:activations:0*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_8/Shape?
.Encoder/conv1d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.Encoder/conv1d_transpose_8/strided_slice/stack?
0Encoder/conv1d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_8/strided_slice/stack_1?
0Encoder/conv1d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_8/strided_slice/stack_2?
(Encoder/conv1d_transpose_8/strided_sliceStridedSlice)Encoder/conv1d_transpose_8/Shape:output:07Encoder/conv1d_transpose_8/strided_slice/stack:output:09Encoder/conv1d_transpose_8/strided_slice/stack_1:output:09Encoder/conv1d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(Encoder/conv1d_transpose_8/strided_slice?
0Encoder/conv1d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0Encoder/conv1d_transpose_8/strided_slice_1/stack?
2Encoder/conv1d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_8/strided_slice_1/stack_1?
2Encoder/conv1d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2Encoder/conv1d_transpose_8/strided_slice_1/stack_2?
*Encoder/conv1d_transpose_8/strided_slice_1StridedSlice)Encoder/conv1d_transpose_8/Shape:output:09Encoder/conv1d_transpose_8/strided_slice_1/stack:output:0;Encoder/conv1d_transpose_8/strided_slice_1/stack_1:output:0;Encoder/conv1d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*Encoder/conv1d_transpose_8/strided_slice_1?
 Encoder/conv1d_transpose_8/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 Encoder/conv1d_transpose_8/mul/y?
Encoder/conv1d_transpose_8/mulMul3Encoder/conv1d_transpose_8/strided_slice_1:output:0)Encoder/conv1d_transpose_8/mul/y:output:0*
T0*
_output_shapes
: 2 
Encoder/conv1d_transpose_8/mul?
"Encoder/conv1d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"Encoder/conv1d_transpose_8/stack/2?
 Encoder/conv1d_transpose_8/stackPack1Encoder/conv1d_transpose_8/strided_slice:output:0"Encoder/conv1d_transpose_8/mul:z:0+Encoder/conv1d_transpose_8/stack/2:output:0*
N*
T0*
_output_shapes
:2"
 Encoder/conv1d_transpose_8/stack?
:Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dim?
6Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims
ExpandDims"Encoder/re_lu_9/Relu:activations:0CEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????28
6Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims?
GEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPencoder_conv1d_transpose_8_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02I
GEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp?
<Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2>
<Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim?
8Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1
ExpandDimsOEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0EEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2:
8Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1?
?Encoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?Encoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack?
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1?
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2?
9Encoder/conv1d_transpose_8/conv1d_transpose/strided_sliceStridedSlice)Encoder/conv1d_transpose_8/stack:output:0HEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack:output:0JEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_1:output:0JEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2;
9Encoder/conv1d_transpose_8/conv1d_transpose/strided_slice?
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack?
CEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1?
CEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2?
;Encoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1StridedSlice)Encoder/conv1d_transpose_8/stack:output:0JEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack:output:0LEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_1:output:0LEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;Encoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1?
;Encoder/conv1d_transpose_8/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Encoder/conv1d_transpose_8/conv1d_transpose/concat/values_1?
7Encoder/conv1d_transpose_8/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7Encoder/conv1d_transpose_8/conv1d_transpose/concat/axis?
2Encoder/conv1d_transpose_8/conv1d_transpose/concatConcatV2BEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice:output:0DEncoder/conv1d_transpose_8/conv1d_transpose/concat/values_1:output:0DEncoder/conv1d_transpose_8/conv1d_transpose/strided_slice_1:output:0@Encoder/conv1d_transpose_8/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2Encoder/conv1d_transpose_8/conv1d_transpose/concat?
+Encoder/conv1d_transpose_8/conv1d_transposeConv2DBackpropInput;Encoder/conv1d_transpose_8/conv1d_transpose/concat:output:0AEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1:output:0?Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2-
+Encoder/conv1d_transpose_8/conv1d_transpose?
3Encoder/conv1d_transpose_8/conv1d_transpose/SqueezeSqueeze4Encoder/conv1d_transpose_8/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
25
3Encoder/conv1d_transpose_8/conv1d_transpose/Squeeze?
7Encoder/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp@encoder_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Encoder/batch_normalization_12/batchnorm/ReadVariableOp?
.Encoder/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.Encoder/batch_normalization_12/batchnorm/add/y?
,Encoder/batch_normalization_12/batchnorm/addAddV2?Encoder/batch_normalization_12/batchnorm/ReadVariableOp:value:07Encoder/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_12/batchnorm/add?
.Encoder/batch_normalization_12/batchnorm/RsqrtRsqrt0Encoder/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_12/batchnorm/Rsqrt?
;Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpDencoder_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOp?
,Encoder/batch_normalization_12/batchnorm/mulMul2Encoder/batch_normalization_12/batchnorm/Rsqrt:y:0CEncoder/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_12/batchnorm/mul?
.Encoder/batch_normalization_12/batchnorm/mul_1Mul<Encoder/conv1d_transpose_8/conv1d_transpose/Squeeze:output:00Encoder/batch_normalization_12/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_12/batchnorm/mul_1?
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpBencoder_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_1?
.Encoder/batch_normalization_12/batchnorm/mul_2MulAEncoder/batch_normalization_12/batchnorm/ReadVariableOp_1:value:00Encoder/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Encoder/batch_normalization_12/batchnorm/mul_2?
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpBencoder_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_2?
,Encoder/batch_normalization_12/batchnorm/subSubAEncoder/batch_normalization_12/batchnorm/ReadVariableOp_2:value:02Encoder/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Encoder/batch_normalization_12/batchnorm/sub?
.Encoder/batch_normalization_12/batchnorm/add_1AddV22Encoder/batch_normalization_12/batchnorm/mul_1:z:00Encoder/batch_normalization_12/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????20
.Encoder/batch_normalization_12/batchnorm/add_1?
Encoder/re_lu_10/ReluRelu2Encoder/batch_normalization_12/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
Encoder/re_lu_10/Relu?
Encoder/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Encoder/flatten_4/Const?
Encoder/flatten_4/ReshapeReshape#Encoder/re_lu_10/Relu:activations:0 Encoder/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
Encoder/flatten_4/Reshape?
&Encoder/dense_10/MatMul/ReadVariableOpReadVariableOp/encoder_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Encoder/dense_10/MatMul/ReadVariableOp?
Encoder/dense_10/MatMulMatMul"Encoder/flatten_4/Reshape:output:0.Encoder/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_10/MatMul?
Encoder/dense_10/TanhTanh!Encoder/dense_10/MatMul:product:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_10/Tanh?
&Encoder/dense_11/MatMul/ReadVariableOpReadVariableOp/encoder_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Encoder/dense_11/MatMul/ReadVariableOp?
Encoder/dense_11/MatMulMatMul"Encoder/flatten_4/Reshape:output:0.Encoder/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_11/MatMul?
Encoder/dense_11/TanhTanh!Encoder/dense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_11/Tanhy
Encoder/lambda_1/ShapeShapeEncoder/dense_10/Tanh:y:0*
T0*
_output_shapes
:2
Encoder/lambda_1/Shape?
#Encoder/lambda_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#Encoder/lambda_1/random_normal/mean?
%Encoder/lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%Encoder/lambda_1/random_normal/stddev?
3Encoder/lambda_1/random_normal/RandomStandardNormalRandomStandardNormalEncoder/lambda_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???25
3Encoder/lambda_1/random_normal/RandomStandardNormal?
"Encoder/lambda_1/random_normal/mulMul<Encoder/lambda_1/random_normal/RandomStandardNormal:output:0.Encoder/lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2$
"Encoder/lambda_1/random_normal/mul?
Encoder/lambda_1/random_normalAdd&Encoder/lambda_1/random_normal/mul:z:0,Encoder/lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2 
Encoder/lambda_1/random_normal}
Encoder/lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Encoder/lambda_1/truediv/y?
Encoder/lambda_1/truedivRealDivEncoder/dense_11/Tanh:y:0#Encoder/lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/truediv?
Encoder/lambda_1/ExpExpEncoder/lambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/Exp?
Encoder/lambda_1/mulMul"Encoder/lambda_1/random_normal:z:0Encoder/lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/mul?
Encoder/lambda_1/addAddV2Encoder/dense_10/Tanh:y:0Encoder/lambda_1/mul:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/add?
IdentityIdentityEncoder/lambda_1/add:z:08^Encoder/batch_normalization_10/batchnorm/ReadVariableOp:^Encoder/batch_normalization_10/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_10/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_11/batchnorm/ReadVariableOp:^Encoder/batch_normalization_11/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_11/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOp8^Encoder/batch_normalization_12/batchnorm/ReadVariableOp:^Encoder/batch_normalization_12/batchnorm/ReadVariableOp_1:^Encoder/batch_normalization_12/batchnorm/ReadVariableOp_2<^Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOp7^Encoder/batch_normalization_9/batchnorm/ReadVariableOp9^Encoder/batch_normalization_9/batchnorm/ReadVariableOp_19^Encoder/batch_normalization_9/batchnorm/ReadVariableOp_2;^Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOpH^Encoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOpH^Encoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpH^Encoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp'^Encoder/dense_10/MatMul/ReadVariableOp'^Encoder/dense_11/MatMul/ReadVariableOp&^Encoder/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????::::::::::::::::::::::2r
7Encoder/batch_normalization_10/batchnorm/ReadVariableOp7Encoder/batch_normalization_10/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_19Encoder/batch_normalization_10/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_10/batchnorm/ReadVariableOp_29Encoder/batch_normalization_10/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_10/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_11/batchnorm/ReadVariableOp7Encoder/batch_normalization_11/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_19Encoder/batch_normalization_11/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_11/batchnorm/ReadVariableOp_29Encoder/batch_normalization_11/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_11/batchnorm/mul/ReadVariableOp2r
7Encoder/batch_normalization_12/batchnorm/ReadVariableOp7Encoder/batch_normalization_12/batchnorm/ReadVariableOp2v
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_19Encoder/batch_normalization_12/batchnorm/ReadVariableOp_12v
9Encoder/batch_normalization_12/batchnorm/ReadVariableOp_29Encoder/batch_normalization_12/batchnorm/ReadVariableOp_22z
;Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOp;Encoder/batch_normalization_12/batchnorm/mul/ReadVariableOp2p
6Encoder/batch_normalization_9/batchnorm/ReadVariableOp6Encoder/batch_normalization_9/batchnorm/ReadVariableOp2t
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_18Encoder/batch_normalization_9/batchnorm/ReadVariableOp_12t
8Encoder/batch_normalization_9/batchnorm/ReadVariableOp_28Encoder/batch_normalization_9/batchnorm/ReadVariableOp_22x
:Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOp:Encoder/batch_normalization_9/batchnorm/mul/ReadVariableOp2?
GEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOpGEncoder/conv1d_transpose_6/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
GEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOpGEncoder/conv1d_transpose_7/conv1d_transpose/ExpandDims_1/ReadVariableOp2?
GEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOpGEncoder/conv1d_transpose_8/conv1d_transpose/ExpandDims_1/ReadVariableOp2P
&Encoder/dense_10/MatMul/ReadVariableOp&Encoder/dense_10/MatMul/ReadVariableOp2P
&Encoder/dense_11/MatMul/ReadVariableOp&Encoder/dense_11/MatMul/ReadVariableOp2N
%Encoder/dense_9/MatMul/ReadVariableOp%Encoder/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?a
?
#__inference__traced_restore_8544571
file_prefix#
assignvariableop_dense_9_kernel2
.assignvariableop_1_batch_normalization_9_gamma1
-assignvariableop_2_batch_normalization_9_beta8
4assignvariableop_3_batch_normalization_9_moving_mean<
8assignvariableop_4_batch_normalization_9_moving_variance0
,assignvariableop_5_conv1d_transpose_6_kernel3
/assignvariableop_6_batch_normalization_10_gamma2
.assignvariableop_7_batch_normalization_10_beta9
5assignvariableop_8_batch_normalization_10_moving_mean=
9assignvariableop_9_batch_normalization_10_moving_variance1
-assignvariableop_10_conv1d_transpose_7_kernel4
0assignvariableop_11_batch_normalization_11_gamma3
/assignvariableop_12_batch_normalization_11_beta:
6assignvariableop_13_batch_normalization_11_moving_mean>
:assignvariableop_14_batch_normalization_11_moving_variance1
-assignvariableop_15_conv1d_transpose_8_kernel4
0assignvariableop_16_batch_normalization_12_gamma3
/assignvariableop_17_batch_normalization_12_beta:
6assignvariableop_18_batch_normalization_12_moving_mean>
:assignvariableop_19_batch_normalization_12_moving_variance'
#assignvariableop_20_dense_10_kernel'
#assignvariableop_21_dense_11_kernel
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_9_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batch_normalization_9_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp4assignvariableop_3_batch_normalization_9_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_normalization_9_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_conv1d_transpose_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_10_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_10_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_10_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_10_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv1d_transpose_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_11_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_11_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_11_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_11_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_conv1d_transpose_8_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_12_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_12_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_12_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_12_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_11_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212(
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
?1
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544259

inputs
assignmovingavg_8544234
assignmovingavg_1_8544240)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544234*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8544234*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544234*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/8544234*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8544234AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/8544234*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544240*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8544240*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544240*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/8544240*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8544240AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/8544240*
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
?
E
)__inference_re_lu_7_layer_call_fn_8544021

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_85427162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_10_layer_call_and_return_conditional_losses_8542926

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
z
4__inference_conv1d_transpose_7_layer_call_fn_8542328

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
GPU 2J 8? *X
fSRQ
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_85423202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8542642

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
input_50
serving_default_input_5:0?????????<
lambda_10
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
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}, "name": "reshape_2", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_6", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv1d_transpose_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_7", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv1d_transpose_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_8", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_transpose_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_10", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["re_lu_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+msvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA2MTYvb3Jp\nZ2luYWwtYWFlL25ldHdvcmtfUmVMVS5wedoIPGxhbWJkYT5HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network_ReLU", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_10", 0, 0, {}], ["dense_11", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}, "name": "reshape_2", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_6", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv1d_transpose_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_7", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv1d_transpose_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_8", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_transpose_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_10", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["re_lu_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+msvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA2MTYvb3Jp\nZ2luYWwtYWFlL25ldHdvcmtfUmVMVS5wedoIPGxhbWJkYT5HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network_ReLU", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_10", 0, 0, {}], ["dense_11", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?

kernel
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?	
axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?


.kernel
/trainable_variables
0	variables
1regularization_losses
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_6", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
?	
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}}
?
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


@kernel
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_7", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}}
?	
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2]}}
?
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


Rkernel
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_8", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2]}}
?	
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
?
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

hkernel
itrainable_variables
j	variables
kregularization_losses
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?

mkernel
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+msvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA2MTYvb3Jp\nZ2luYWwtYWFlL25ldHdvcmtfUmVMVS5wedoIPGxhbWJkYT5HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network_ReLU", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
0
1
2
.3
44
55
@6
F7
G8
R9
X10
Y11
h12
m13"
trackable_list_wrapper
?
0
1
2
 3
!4
.5
46
57
68
79
@10
F11
G12
H13
I14
R15
X16
Y17
Z18
[19
h20
m21"
trackable_list_wrapper
 "
trackable_list_wrapper
?

vlayers
trainable_variables
wnon_trainable_variables
xlayer_regularization_losses
ymetrics
	variables
zlayer_metrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 :2dense_9/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{layer_metrics
|layer_regularization_losses

}layers
trainable_variables
~metrics
	variables
non_trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
"trainable_variables
?metrics
#	variables
?non_trainable_variables
$regularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
&trainable_variables
?metrics
'	variables
?non_trainable_variables
(regularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
*trainable_variables
?metrics
+	variables
?non_trainable_variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_6/kernel
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
/trainable_variables
?metrics
0	variables
?non_trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_10/gamma
):'2batch_normalization_10/beta
2:0 (2"batch_normalization_10/moving_mean
6:4 (2&batch_normalization_10/moving_variance
.
40
51"
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
8trainable_variables
?metrics
9	variables
?non_trainable_variables
:regularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
<trainable_variables
?metrics
=	variables
?non_trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_7/kernel
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
Atrainable_variables
?metrics
B	variables
?non_trainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
.
F0
G1"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
Jtrainable_variables
?metrics
K	variables
?non_trainable_variables
Lregularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
Ntrainable_variables
?metrics
O	variables
?non_trainable_variables
Pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_8/kernel
'
R0"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
Strainable_variables
?metrics
T	variables
?non_trainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
.
X0
Y1"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
\trainable_variables
?metrics
]	variables
?non_trainable_variables
^regularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
`trainable_variables
?metrics
a	variables
?non_trainable_variables
bregularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
dtrainable_variables
?metrics
e	variables
?non_trainable_variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_10/kernel
'
h0"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
itrainable_variables
?metrics
j	variables
?non_trainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_11/kernel
'
m0"
trackable_list_wrapper
'
m0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
?layers
ntrainable_variables
?metrics
o	variables
?non_trainable_variables
pregularization_losses
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
?layer_metrics
 ?layer_regularization_losses
?layers
rtrainable_variables
?metrics
s	variables
?non_trainable_variables
tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
16
17"
trackable_list_wrapper
X
 0
!1
62
73
H4
I5
Z6
[7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 0
!1"
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
60
71"
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
H0
I1"
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
Z0
[1"
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
?2?
D__inference_Encoder_layer_call_and_return_conditional_losses_8543616
D__inference_Encoder_layer_call_and_return_conditional_losses_8543071
D__inference_Encoder_layer_call_and_return_conditional_losses_8543006
D__inference_Encoder_layer_call_and_return_conditional_losses_8543817?
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
"__inference__wrapped_model_8541958?
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
input_5?????????
?2?
)__inference_Encoder_layer_call_fn_8543866
)__inference_Encoder_layer_call_fn_8543300
)__inference_Encoder_layer_call_fn_8543915
)__inference_Encoder_layer_call_fn_8543186?
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
D__inference_dense_9_layer_call_and_return_conditional_losses_8543922?
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
)__inference_dense_9_layer_call_fn_8543929?
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543965
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543985?
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
7__inference_batch_normalization_9_layer_call_fn_8543998
7__inference_batch_normalization_9_layer_call_fn_8544011?
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
D__inference_re_lu_7_layer_call_and_return_conditional_losses_8544016?
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
)__inference_re_lu_7_layer_call_fn_8544021?
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
F__inference_reshape_2_layer_call_and_return_conditional_losses_8544034?
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
+__inference_reshape_2_layer_call_fn_8544039?
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
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_8542135?
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
%?"??????????????????
?2?
4__inference_conv1d_transpose_6_layer_call_fn_8542143?
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
%?"??????????????????
?2?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544095
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544075?
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
8__inference_batch_normalization_10_layer_call_fn_8544108
8__inference_batch_normalization_10_layer_call_fn_8544121?
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
D__inference_re_lu_8_layer_call_and_return_conditional_losses_8544126?
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
)__inference_re_lu_8_layer_call_fn_8544131?
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
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_8542320?
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
4__inference_conv1d_transpose_7_layer_call_fn_8542328?
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
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544187
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544167?
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
8__inference_batch_normalization_11_layer_call_fn_8544200
8__inference_batch_normalization_11_layer_call_fn_8544213?
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
D__inference_re_lu_9_layer_call_and_return_conditional_losses_8544218?
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
)__inference_re_lu_9_layer_call_fn_8544223?
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
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_8542505?
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
4__inference_conv1d_transpose_8_layer_call_fn_8542513?
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544259
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544279?
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
8__inference_batch_normalization_12_layer_call_fn_8544292
8__inference_batch_normalization_12_layer_call_fn_8544305?
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
E__inference_re_lu_10_layer_call_and_return_conditional_losses_8544310?
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
*__inference_re_lu_10_layer_call_fn_8544315?
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
F__inference_flatten_4_layer_call_and_return_conditional_losses_8544327?
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
+__inference_flatten_4_layer_call_fn_8544332?
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
E__inference_dense_10_layer_call_and_return_conditional_losses_8544340?
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
*__inference_dense_10_layer_call_fn_8544347?
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
E__inference_dense_11_layer_call_and_return_conditional_losses_8544355?
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
*__inference_dense_11_layer_call_fn_8544362?
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
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544394
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544378?
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
*__inference_lambda_1_layer_call_fn_8544400
*__inference_lambda_1_layer_call_fn_8544406?
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
?B?
%__inference_signature_wrapper_8543351input_5"?
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
D__inference_Encoder_layer_call_and_return_conditional_losses_8543006y !.6745@HIFGRZ[XYhm8?5
.?+
!?
input_5?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_Encoder_layer_call_and_return_conditional_losses_8543071y! .7465@IFHGR[XZYhm8?5
.?+
!?
input_5?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_Encoder_layer_call_and_return_conditional_losses_8543616x !.6745@HIFGRZ[XYhm7?4
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
D__inference_Encoder_layer_call_and_return_conditional_losses_8543817x! .7465@IFHGR[XZYhm7?4
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
)__inference_Encoder_layer_call_fn_8543186l !.6745@HIFGRZ[XYhm8?5
.?+
!?
input_5?????????
p

 
? "???????????
)__inference_Encoder_layer_call_fn_8543300l! .7465@IFHGR[XZYhm8?5
.?+
!?
input_5?????????
p 

 
? "???????????
)__inference_Encoder_layer_call_fn_8543866k !.6745@HIFGRZ[XYhm7?4
-?*
 ?
inputs?????????
p

 
? "???????????
)__inference_Encoder_layer_call_fn_8543915k! .7465@IFHGR[XZYhm7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
"__inference__wrapped_model_8541958! .7465@IFHGR[XZYhm0?-
&?#
!?
input_5?????????
? "3?0
.
lambda_1"?
lambda_1??????????
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544075|6745@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_8544095|7465@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_10_layer_call_fn_8544108o6745@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
8__inference_batch_normalization_10_layer_call_fn_8544121o7465@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544167|HIFG@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_8544187|IFHG@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_11_layer_call_fn_8544200oHIFG@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
8__inference_batch_normalization_11_layer_call_fn_8544213oIFHG@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544259|Z[XY@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_8544279|[XZY@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_12_layer_call_fn_8544292oZ[XY@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
8__inference_batch_normalization_12_layer_call_fn_8544305o[XZY@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543965b !3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_8543985b! 3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
7__inference_batch_normalization_9_layer_call_fn_8543998U !3?0
)?&
 ?
inputs?????????
p
? "???????????
7__inference_batch_normalization_9_layer_call_fn_8544011U! 3?0
)?&
 ?
inputs?????????
p 
? "???????????
O__inference_conv1d_transpose_6_layer_call_and_return_conditional_losses_8542135u.<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
4__inference_conv1d_transpose_6_layer_call_fn_8542143h.<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
O__inference_conv1d_transpose_7_layer_call_and_return_conditional_losses_8542320u@<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
4__inference_conv1d_transpose_7_layer_call_fn_8542328h@<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
O__inference_conv1d_transpose_8_layer_call_and_return_conditional_losses_8542505uR<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
4__inference_conv1d_transpose_8_layer_call_fn_8542513hR<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
E__inference_dense_10_layer_call_and_return_conditional_losses_8544340dh8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
*__inference_dense_10_layer_call_fn_8544347Wh8?5
.?+
)?&
inputs??????????????????
? "???????????
E__inference_dense_11_layer_call_and_return_conditional_losses_8544355dm8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
*__inference_dense_11_layer_call_fn_8544362Wm8?5
.?+
)?&
inputs??????????????????
? "???????????
D__inference_dense_9_layer_call_and_return_conditional_losses_8543922[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
)__inference_dense_9_layer_call_fn_8543929N/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_flatten_4_layer_call_and_return_conditional_losses_8544327n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
+__inference_flatten_4_layer_call_fn_8544332a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544378?b?_
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
E__inference_lambda_1_layer_call_and_return_conditional_losses_8544394?b?_
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
*__inference_lambda_1_layer_call_fn_8544400~b?_
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
*__inference_lambda_1_layer_call_fn_8544406~b?_
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
E__inference_re_lu_10_layer_call_and_return_conditional_losses_8544310r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
*__inference_re_lu_10_layer_call_fn_8544315e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
D__inference_re_lu_7_layer_call_and_return_conditional_losses_8544016X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
)__inference_re_lu_7_layer_call_fn_8544021K/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_re_lu_8_layer_call_and_return_conditional_losses_8544126r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
)__inference_re_lu_8_layer_call_fn_8544131e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
D__inference_re_lu_9_layer_call_and_return_conditional_losses_8544218r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
)__inference_re_lu_9_layer_call_fn_8544223e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
F__inference_reshape_2_layer_call_and_return_conditional_losses_8544034\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ~
+__inference_reshape_2_layer_call_fn_8544039O/?,
%?"
 ?
inputs?????????
? "???????????
%__inference_signature_wrapper_8543351?! .7465@IFHGR[XZYhm;?8
? 
1?.
,
input_5!?
input_5?????????"3?0
.
lambda_1"?
lambda_1?????????