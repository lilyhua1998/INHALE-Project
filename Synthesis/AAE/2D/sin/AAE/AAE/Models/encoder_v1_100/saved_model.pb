??
??
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
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	?*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:?*
dtype0
?
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_normalization_4/gamma
?
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_normalization_4/beta
?
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes	
:?*
dtype0
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
??*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:?*
dtype0
?
layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_normalization_5/gamma
?
/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes	
:?*
dtype0
?
layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_normalization_5/beta
?
.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes	
:?*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	?*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	?*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
?
layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_6/gamma
?
/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:*
dtype0
?
layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_6/beta
?
.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:*
dtype0
?
layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_7/gamma
?
/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
:*
dtype0
?
layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_7/beta
?
.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
:*
dtype0

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
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
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
q
axis
	gamma
beta
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
q
.axis
	/gamma
0beta
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
q
Eaxis
	Fgamma
Gbeta
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
q
Laxis
	Mgamma
Nbeta
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
R
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
v
0
1
2
3
(4
)5
/6
07
98
:9
?10
@11
F12
G13
M14
N15
v
0
1
2
3
(4
)5
/6
07
98
:9
?10
@11
F12
G13
M14
N15
 
?
Wmetrics
trainable_variables
	variables
regularization_losses
Xlayer_metrics

Ylayers
Znon_trainable_variables
[layer_regularization_losses
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
\metrics
trainable_variables
	variables
regularization_losses
]layer_metrics

^layers
_non_trainable_variables
`layer_regularization_losses
 
 
 
?
ametrics
trainable_variables
	variables
regularization_losses
blayer_metrics

clayers
dnon_trainable_variables
elayer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_4/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_4/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
fmetrics
 trainable_variables
!	variables
"regularization_losses
glayer_metrics

hlayers
inon_trainable_variables
jlayer_regularization_losses
 
 
 
?
kmetrics
$trainable_variables
%	variables
&regularization_losses
llayer_metrics

mlayers
nnon_trainable_variables
olayer_regularization_losses
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
?
pmetrics
*trainable_variables
+	variables
,regularization_losses
qlayer_metrics

rlayers
snon_trainable_variables
tlayer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_5/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_5/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
umetrics
1trainable_variables
2	variables
3regularization_losses
vlayer_metrics

wlayers
xnon_trainable_variables
ylayer_regularization_losses
 
 
 
?
zmetrics
5trainable_variables
6	variables
7regularization_losses
{layer_metrics

|layers
}non_trainable_variables
~layer_regularization_losses
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
metrics
;trainable_variables
<	variables
=regularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
?metrics
Atrainable_variables
B	variables
Cregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_6/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_6/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
?
?metrics
Htrainable_variables
I	variables
Jregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_7/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_7/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
?
?metrics
Otrainable_variables
P	variables
Qregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
Strainable_variables
T	variables
Uregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
^
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
 
 
z
serving_default_input_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5dense_14/kerneldense_14/biaslayer_normalization_4/gammalayer_normalization_4/betadense_15/kerneldense_15/biaslayer_normalization_5/gammalayer_normalization_5/betadense_17/kerneldense_17/biasdense_16/kerneldense_16/biaslayer_normalization_6/gammalayer_normalization_6/betalayer_normalization_7/gammalayer_normalization_7/beta*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_142531
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_143448
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biaslayer_normalization_4/gammalayer_normalization_4/betadense_15/kerneldense_15/biaslayer_normalization_5/gammalayer_normalization_5/betadense_16/kerneldense_16/biasdense_17/kerneldense_17/biaslayer_normalization_6/gammalayer_normalization_6/betalayer_normalization_7/gammalayer_normalization_7/beta*
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_143506??
?"
?
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_143273

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_143040

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_141841

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_141968

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_Encoder_layer_call_fn_142407
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1423722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
ͤ
?
!__inference__wrapped_model_141827
input_53
/encoder_dense_14_matmul_readvariableop_resource4
0encoder_dense_14_biasadd_readvariableop_resource?
;encoder_layer_normalization_4_mul_2_readvariableop_resource=
9encoder_layer_normalization_4_add_readvariableop_resource3
/encoder_dense_15_matmul_readvariableop_resource4
0encoder_dense_15_biasadd_readvariableop_resource?
;encoder_layer_normalization_5_mul_2_readvariableop_resource=
9encoder_layer_normalization_5_add_readvariableop_resource3
/encoder_dense_17_matmul_readvariableop_resource4
0encoder_dense_17_biasadd_readvariableop_resource3
/encoder_dense_16_matmul_readvariableop_resource4
0encoder_dense_16_biasadd_readvariableop_resource?
;encoder_layer_normalization_6_mul_2_readvariableop_resource=
9encoder_layer_normalization_6_add_readvariableop_resource?
;encoder_layer_normalization_7_mul_2_readvariableop_resource=
9encoder_layer_normalization_7_add_readvariableop_resource
identity??'Encoder/dense_14/BiasAdd/ReadVariableOp?&Encoder/dense_14/MatMul/ReadVariableOp?'Encoder/dense_15/BiasAdd/ReadVariableOp?&Encoder/dense_15/MatMul/ReadVariableOp?'Encoder/dense_16/BiasAdd/ReadVariableOp?&Encoder/dense_16/MatMul/ReadVariableOp?'Encoder/dense_17/BiasAdd/ReadVariableOp?&Encoder/dense_17/MatMul/ReadVariableOp?0Encoder/layer_normalization_4/add/ReadVariableOp?2Encoder/layer_normalization_4/mul_2/ReadVariableOp?0Encoder/layer_normalization_5/add/ReadVariableOp?2Encoder/layer_normalization_5/mul_2/ReadVariableOp?0Encoder/layer_normalization_6/add/ReadVariableOp?2Encoder/layer_normalization_6/mul_2/ReadVariableOp?0Encoder/layer_normalization_7/add/ReadVariableOp?2Encoder/layer_normalization_7/mul_2/ReadVariableOp?
&Encoder/dense_14/MatMul/ReadVariableOpReadVariableOp/encoder_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Encoder/dense_14/MatMul/ReadVariableOp?
Encoder/dense_14/MatMulMatMulinput_5.Encoder/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Encoder/dense_14/MatMul?
'Encoder/dense_14/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'Encoder/dense_14/BiasAdd/ReadVariableOp?
Encoder/dense_14/BiasAddBiasAdd!Encoder/dense_14/MatMul:product:0/Encoder/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Encoder/dense_14/BiasAdd?
Encoder/dropout_1/IdentityIdentity!Encoder/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Encoder/dropout_1/Identity?
#Encoder/layer_normalization_4/ShapeShape#Encoder/dropout_1/Identity:output:0*
T0*
_output_shapes
:2%
#Encoder/layer_normalization_4/Shape?
1Encoder/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1Encoder/layer_normalization_4/strided_slice/stack?
3Encoder/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_4/strided_slice/stack_1?
3Encoder/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_4/strided_slice/stack_2?
+Encoder/layer_normalization_4/strided_sliceStridedSlice,Encoder/layer_normalization_4/Shape:output:0:Encoder/layer_normalization_4/strided_slice/stack:output:0<Encoder/layer_normalization_4/strided_slice/stack_1:output:0<Encoder/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/layer_normalization_4/strided_slice?
#Encoder/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/layer_normalization_4/mul/x?
!Encoder/layer_normalization_4/mulMul,Encoder/layer_normalization_4/mul/x:output:04Encoder/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: 2#
!Encoder/layer_normalization_4/mul?
3Encoder/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_4/strided_slice_1/stack?
5Encoder/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_4/strided_slice_1/stack_1?
5Encoder/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_4/strided_slice_1/stack_2?
-Encoder/layer_normalization_4/strided_slice_1StridedSlice,Encoder/layer_normalization_4/Shape:output:0<Encoder/layer_normalization_4/strided_slice_1/stack:output:0>Encoder/layer_normalization_4/strided_slice_1/stack_1:output:0>Encoder/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Encoder/layer_normalization_4/strided_slice_1?
%Encoder/layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%Encoder/layer_normalization_4/mul_1/x?
#Encoder/layer_normalization_4/mul_1Mul.Encoder/layer_normalization_4/mul_1/x:output:06Encoder/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#Encoder/layer_normalization_4/mul_1?
-Encoder/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_4/Reshape/shape/0?
-Encoder/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_4/Reshape/shape/3?
+Encoder/layer_normalization_4/Reshape/shapePack6Encoder/layer_normalization_4/Reshape/shape/0:output:0%Encoder/layer_normalization_4/mul:z:0'Encoder/layer_normalization_4/mul_1:z:06Encoder/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+Encoder/layer_normalization_4/Reshape/shape?
%Encoder/layer_normalization_4/ReshapeReshape#Encoder/dropout_1/Identity:output:04Encoder/layer_normalization_4/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%Encoder/layer_normalization_4/Reshape?
#Encoder/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#Encoder/layer_normalization_4/Const?
'Encoder/layer_normalization_4/Fill/dimsPack%Encoder/layer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2)
'Encoder/layer_normalization_4/Fill/dims?
"Encoder/layer_normalization_4/FillFill0Encoder/layer_normalization_4/Fill/dims:output:0,Encoder/layer_normalization_4/Const:output:0*
T0*#
_output_shapes
:?????????2$
"Encoder/layer_normalization_4/Fill?
%Encoder/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/layer_normalization_4/Const_1?
)Encoder/layer_normalization_4/Fill_1/dimsPack%Encoder/layer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2+
)Encoder/layer_normalization_4/Fill_1/dims?
$Encoder/layer_normalization_4/Fill_1Fill2Encoder/layer_normalization_4/Fill_1/dims:output:0.Encoder/layer_normalization_4/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$Encoder/layer_normalization_4/Fill_1?
%Encoder/layer_normalization_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_4/Const_2?
%Encoder/layer_normalization_4/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_4/Const_3?
.Encoder/layer_normalization_4/FusedBatchNormV3FusedBatchNormV3.Encoder/layer_normalization_4/Reshape:output:0+Encoder/layer_normalization_4/Fill:output:0-Encoder/layer_normalization_4/Fill_1:output:0.Encoder/layer_normalization_4/Const_2:output:0.Encoder/layer_normalization_4/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.Encoder/layer_normalization_4/FusedBatchNormV3?
'Encoder/layer_normalization_4/Reshape_1Reshape2Encoder/layer_normalization_4/FusedBatchNormV3:y:0,Encoder/layer_normalization_4/Shape:output:0*
T0*(
_output_shapes
:??????????2)
'Encoder/layer_normalization_4/Reshape_1?
2Encoder/layer_normalization_4/mul_2/ReadVariableOpReadVariableOp;encoder_layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype024
2Encoder/layer_normalization_4/mul_2/ReadVariableOp?
#Encoder/layer_normalization_4/mul_2Mul0Encoder/layer_normalization_4/Reshape_1:output:0:Encoder/layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#Encoder/layer_normalization_4/mul_2?
0Encoder/layer_normalization_4/add/ReadVariableOpReadVariableOp9encoder_layer_normalization_4_add_readvariableop_resource*
_output_shapes	
:?*
dtype022
0Encoder/layer_normalization_4/add/ReadVariableOp?
!Encoder/layer_normalization_4/addAddV2'Encoder/layer_normalization_4/mul_2:z:08Encoder/layer_normalization_4/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!Encoder/layer_normalization_4/add?
Encoder/re_lu_8/ReluRelu%Encoder/layer_normalization_4/add:z:0*
T0*(
_output_shapes
:??????????2
Encoder/re_lu_8/Relu?
&Encoder/dense_15/MatMul/ReadVariableOpReadVariableOp/encoder_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&Encoder/dense_15/MatMul/ReadVariableOp?
Encoder/dense_15/MatMulMatMul"Encoder/re_lu_8/Relu:activations:0.Encoder/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Encoder/dense_15/MatMul?
'Encoder/dense_15/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'Encoder/dense_15/BiasAdd/ReadVariableOp?
Encoder/dense_15/BiasAddBiasAdd!Encoder/dense_15/MatMul:product:0/Encoder/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Encoder/dense_15/BiasAdd?
#Encoder/layer_normalization_5/ShapeShape!Encoder/dense_15/BiasAdd:output:0*
T0*
_output_shapes
:2%
#Encoder/layer_normalization_5/Shape?
1Encoder/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1Encoder/layer_normalization_5/strided_slice/stack?
3Encoder/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_5/strided_slice/stack_1?
3Encoder/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_5/strided_slice/stack_2?
+Encoder/layer_normalization_5/strided_sliceStridedSlice,Encoder/layer_normalization_5/Shape:output:0:Encoder/layer_normalization_5/strided_slice/stack:output:0<Encoder/layer_normalization_5/strided_slice/stack_1:output:0<Encoder/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/layer_normalization_5/strided_slice?
#Encoder/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/layer_normalization_5/mul/x?
!Encoder/layer_normalization_5/mulMul,Encoder/layer_normalization_5/mul/x:output:04Encoder/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: 2#
!Encoder/layer_normalization_5/mul?
3Encoder/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_5/strided_slice_1/stack?
5Encoder/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_5/strided_slice_1/stack_1?
5Encoder/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_5/strided_slice_1/stack_2?
-Encoder/layer_normalization_5/strided_slice_1StridedSlice,Encoder/layer_normalization_5/Shape:output:0<Encoder/layer_normalization_5/strided_slice_1/stack:output:0>Encoder/layer_normalization_5/strided_slice_1/stack_1:output:0>Encoder/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Encoder/layer_normalization_5/strided_slice_1?
%Encoder/layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%Encoder/layer_normalization_5/mul_1/x?
#Encoder/layer_normalization_5/mul_1Mul.Encoder/layer_normalization_5/mul_1/x:output:06Encoder/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#Encoder/layer_normalization_5/mul_1?
-Encoder/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_5/Reshape/shape/0?
-Encoder/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_5/Reshape/shape/3?
+Encoder/layer_normalization_5/Reshape/shapePack6Encoder/layer_normalization_5/Reshape/shape/0:output:0%Encoder/layer_normalization_5/mul:z:0'Encoder/layer_normalization_5/mul_1:z:06Encoder/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+Encoder/layer_normalization_5/Reshape/shape?
%Encoder/layer_normalization_5/ReshapeReshape!Encoder/dense_15/BiasAdd:output:04Encoder/layer_normalization_5/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%Encoder/layer_normalization_5/Reshape?
#Encoder/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#Encoder/layer_normalization_5/Const?
'Encoder/layer_normalization_5/Fill/dimsPack%Encoder/layer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2)
'Encoder/layer_normalization_5/Fill/dims?
"Encoder/layer_normalization_5/FillFill0Encoder/layer_normalization_5/Fill/dims:output:0,Encoder/layer_normalization_5/Const:output:0*
T0*#
_output_shapes
:?????????2$
"Encoder/layer_normalization_5/Fill?
%Encoder/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/layer_normalization_5/Const_1?
)Encoder/layer_normalization_5/Fill_1/dimsPack%Encoder/layer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2+
)Encoder/layer_normalization_5/Fill_1/dims?
$Encoder/layer_normalization_5/Fill_1Fill2Encoder/layer_normalization_5/Fill_1/dims:output:0.Encoder/layer_normalization_5/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$Encoder/layer_normalization_5/Fill_1?
%Encoder/layer_normalization_5/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_5/Const_2?
%Encoder/layer_normalization_5/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_5/Const_3?
.Encoder/layer_normalization_5/FusedBatchNormV3FusedBatchNormV3.Encoder/layer_normalization_5/Reshape:output:0+Encoder/layer_normalization_5/Fill:output:0-Encoder/layer_normalization_5/Fill_1:output:0.Encoder/layer_normalization_5/Const_2:output:0.Encoder/layer_normalization_5/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.Encoder/layer_normalization_5/FusedBatchNormV3?
'Encoder/layer_normalization_5/Reshape_1Reshape2Encoder/layer_normalization_5/FusedBatchNormV3:y:0,Encoder/layer_normalization_5/Shape:output:0*
T0*(
_output_shapes
:??????????2)
'Encoder/layer_normalization_5/Reshape_1?
2Encoder/layer_normalization_5/mul_2/ReadVariableOpReadVariableOp;encoder_layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype024
2Encoder/layer_normalization_5/mul_2/ReadVariableOp?
#Encoder/layer_normalization_5/mul_2Mul0Encoder/layer_normalization_5/Reshape_1:output:0:Encoder/layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#Encoder/layer_normalization_5/mul_2?
0Encoder/layer_normalization_5/add/ReadVariableOpReadVariableOp9encoder_layer_normalization_5_add_readvariableop_resource*
_output_shapes	
:?*
dtype022
0Encoder/layer_normalization_5/add/ReadVariableOp?
!Encoder/layer_normalization_5/addAddV2'Encoder/layer_normalization_5/mul_2:z:08Encoder/layer_normalization_5/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!Encoder/layer_normalization_5/add?
Encoder/re_lu_9/ReluRelu%Encoder/layer_normalization_5/add:z:0*
T0*(
_output_shapes
:??????????2
Encoder/re_lu_9/Relu?
&Encoder/dense_17/MatMul/ReadVariableOpReadVariableOp/encoder_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Encoder/dense_17/MatMul/ReadVariableOp?
Encoder/dense_17/MatMulMatMul"Encoder/re_lu_9/Relu:activations:0.Encoder/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_17/MatMul?
'Encoder/dense_17/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Encoder/dense_17/BiasAdd/ReadVariableOp?
Encoder/dense_17/BiasAddBiasAdd!Encoder/dense_17/MatMul:product:0/Encoder/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_17/BiasAdd?
Encoder/dense_17/TanhTanh!Encoder/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_17/Tanh?
&Encoder/dense_16/MatMul/ReadVariableOpReadVariableOp/encoder_dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Encoder/dense_16/MatMul/ReadVariableOp?
Encoder/dense_16/MatMulMatMul"Encoder/re_lu_9/Relu:activations:0.Encoder/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_16/MatMul?
'Encoder/dense_16/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Encoder/dense_16/BiasAdd/ReadVariableOp?
Encoder/dense_16/BiasAddBiasAdd!Encoder/dense_16/MatMul:product:0/Encoder/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_16/BiasAdd?
Encoder/dense_16/TanhTanh!Encoder/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/dense_16/Tanh?
#Encoder/layer_normalization_6/ShapeShapeEncoder/dense_16/Tanh:y:0*
T0*
_output_shapes
:2%
#Encoder/layer_normalization_6/Shape?
1Encoder/layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1Encoder/layer_normalization_6/strided_slice/stack?
3Encoder/layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_6/strided_slice/stack_1?
3Encoder/layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_6/strided_slice/stack_2?
+Encoder/layer_normalization_6/strided_sliceStridedSlice,Encoder/layer_normalization_6/Shape:output:0:Encoder/layer_normalization_6/strided_slice/stack:output:0<Encoder/layer_normalization_6/strided_slice/stack_1:output:0<Encoder/layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/layer_normalization_6/strided_slice?
#Encoder/layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/layer_normalization_6/mul/x?
!Encoder/layer_normalization_6/mulMul,Encoder/layer_normalization_6/mul/x:output:04Encoder/layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2#
!Encoder/layer_normalization_6/mul?
3Encoder/layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_6/strided_slice_1/stack?
5Encoder/layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_6/strided_slice_1/stack_1?
5Encoder/layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_6/strided_slice_1/stack_2?
-Encoder/layer_normalization_6/strided_slice_1StridedSlice,Encoder/layer_normalization_6/Shape:output:0<Encoder/layer_normalization_6/strided_slice_1/stack:output:0>Encoder/layer_normalization_6/strided_slice_1/stack_1:output:0>Encoder/layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Encoder/layer_normalization_6/strided_slice_1?
%Encoder/layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%Encoder/layer_normalization_6/mul_1/x?
#Encoder/layer_normalization_6/mul_1Mul.Encoder/layer_normalization_6/mul_1/x:output:06Encoder/layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#Encoder/layer_normalization_6/mul_1?
-Encoder/layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_6/Reshape/shape/0?
-Encoder/layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_6/Reshape/shape/3?
+Encoder/layer_normalization_6/Reshape/shapePack6Encoder/layer_normalization_6/Reshape/shape/0:output:0%Encoder/layer_normalization_6/mul:z:0'Encoder/layer_normalization_6/mul_1:z:06Encoder/layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+Encoder/layer_normalization_6/Reshape/shape?
%Encoder/layer_normalization_6/ReshapeReshapeEncoder/dense_16/Tanh:y:04Encoder/layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%Encoder/layer_normalization_6/Reshape?
#Encoder/layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#Encoder/layer_normalization_6/Const?
'Encoder/layer_normalization_6/Fill/dimsPack%Encoder/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2)
'Encoder/layer_normalization_6/Fill/dims?
"Encoder/layer_normalization_6/FillFill0Encoder/layer_normalization_6/Fill/dims:output:0,Encoder/layer_normalization_6/Const:output:0*
T0*#
_output_shapes
:?????????2$
"Encoder/layer_normalization_6/Fill?
%Encoder/layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/layer_normalization_6/Const_1?
)Encoder/layer_normalization_6/Fill_1/dimsPack%Encoder/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2+
)Encoder/layer_normalization_6/Fill_1/dims?
$Encoder/layer_normalization_6/Fill_1Fill2Encoder/layer_normalization_6/Fill_1/dims:output:0.Encoder/layer_normalization_6/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$Encoder/layer_normalization_6/Fill_1?
%Encoder/layer_normalization_6/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_6/Const_2?
%Encoder/layer_normalization_6/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_6/Const_3?
.Encoder/layer_normalization_6/FusedBatchNormV3FusedBatchNormV3.Encoder/layer_normalization_6/Reshape:output:0+Encoder/layer_normalization_6/Fill:output:0-Encoder/layer_normalization_6/Fill_1:output:0.Encoder/layer_normalization_6/Const_2:output:0.Encoder/layer_normalization_6/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.Encoder/layer_normalization_6/FusedBatchNormV3?
'Encoder/layer_normalization_6/Reshape_1Reshape2Encoder/layer_normalization_6/FusedBatchNormV3:y:0,Encoder/layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2)
'Encoder/layer_normalization_6/Reshape_1?
2Encoder/layer_normalization_6/mul_2/ReadVariableOpReadVariableOp;encoder_layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype024
2Encoder/layer_normalization_6/mul_2/ReadVariableOp?
#Encoder/layer_normalization_6/mul_2Mul0Encoder/layer_normalization_6/Reshape_1:output:0:Encoder/layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#Encoder/layer_normalization_6/mul_2?
0Encoder/layer_normalization_6/add/ReadVariableOpReadVariableOp9encoder_layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype022
0Encoder/layer_normalization_6/add/ReadVariableOp?
!Encoder/layer_normalization_6/addAddV2'Encoder/layer_normalization_6/mul_2:z:08Encoder/layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!Encoder/layer_normalization_6/add?
#Encoder/layer_normalization_7/ShapeShapeEncoder/dense_17/Tanh:y:0*
T0*
_output_shapes
:2%
#Encoder/layer_normalization_7/Shape?
1Encoder/layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1Encoder/layer_normalization_7/strided_slice/stack?
3Encoder/layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_7/strided_slice/stack_1?
3Encoder/layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_7/strided_slice/stack_2?
+Encoder/layer_normalization_7/strided_sliceStridedSlice,Encoder/layer_normalization_7/Shape:output:0:Encoder/layer_normalization_7/strided_slice/stack:output:0<Encoder/layer_normalization_7/strided_slice/stack_1:output:0<Encoder/layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+Encoder/layer_normalization_7/strided_slice?
#Encoder/layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#Encoder/layer_normalization_7/mul/x?
!Encoder/layer_normalization_7/mulMul,Encoder/layer_normalization_7/mul/x:output:04Encoder/layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2#
!Encoder/layer_normalization_7/mul?
3Encoder/layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3Encoder/layer_normalization_7/strided_slice_1/stack?
5Encoder/layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_7/strided_slice_1/stack_1?
5Encoder/layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5Encoder/layer_normalization_7/strided_slice_1/stack_2?
-Encoder/layer_normalization_7/strided_slice_1StridedSlice,Encoder/layer_normalization_7/Shape:output:0<Encoder/layer_normalization_7/strided_slice_1/stack:output:0>Encoder/layer_normalization_7/strided_slice_1/stack_1:output:0>Encoder/layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-Encoder/layer_normalization_7/strided_slice_1?
%Encoder/layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%Encoder/layer_normalization_7/mul_1/x?
#Encoder/layer_normalization_7/mul_1Mul.Encoder/layer_normalization_7/mul_1/x:output:06Encoder/layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#Encoder/layer_normalization_7/mul_1?
-Encoder/layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_7/Reshape/shape/0?
-Encoder/layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-Encoder/layer_normalization_7/Reshape/shape/3?
+Encoder/layer_normalization_7/Reshape/shapePack6Encoder/layer_normalization_7/Reshape/shape/0:output:0%Encoder/layer_normalization_7/mul:z:0'Encoder/layer_normalization_7/mul_1:z:06Encoder/layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+Encoder/layer_normalization_7/Reshape/shape?
%Encoder/layer_normalization_7/ReshapeReshapeEncoder/dense_17/Tanh:y:04Encoder/layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%Encoder/layer_normalization_7/Reshape?
#Encoder/layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#Encoder/layer_normalization_7/Const?
'Encoder/layer_normalization_7/Fill/dimsPack%Encoder/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2)
'Encoder/layer_normalization_7/Fill/dims?
"Encoder/layer_normalization_7/FillFill0Encoder/layer_normalization_7/Fill/dims:output:0,Encoder/layer_normalization_7/Const:output:0*
T0*#
_output_shapes
:?????????2$
"Encoder/layer_normalization_7/Fill?
%Encoder/layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/layer_normalization_7/Const_1?
)Encoder/layer_normalization_7/Fill_1/dimsPack%Encoder/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2+
)Encoder/layer_normalization_7/Fill_1/dims?
$Encoder/layer_normalization_7/Fill_1Fill2Encoder/layer_normalization_7/Fill_1/dims:output:0.Encoder/layer_normalization_7/Const_1:output:0*
T0*#
_output_shapes
:?????????2&
$Encoder/layer_normalization_7/Fill_1?
%Encoder/layer_normalization_7/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_7/Const_2?
%Encoder/layer_normalization_7/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2'
%Encoder/layer_normalization_7/Const_3?
.Encoder/layer_normalization_7/FusedBatchNormV3FusedBatchNormV3.Encoder/layer_normalization_7/Reshape:output:0+Encoder/layer_normalization_7/Fill:output:0-Encoder/layer_normalization_7/Fill_1:output:0.Encoder/layer_normalization_7/Const_2:output:0.Encoder/layer_normalization_7/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.Encoder/layer_normalization_7/FusedBatchNormV3?
'Encoder/layer_normalization_7/Reshape_1Reshape2Encoder/layer_normalization_7/FusedBatchNormV3:y:0,Encoder/layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????2)
'Encoder/layer_normalization_7/Reshape_1?
2Encoder/layer_normalization_7/mul_2/ReadVariableOpReadVariableOp;encoder_layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:*
dtype024
2Encoder/layer_normalization_7/mul_2/ReadVariableOp?
#Encoder/layer_normalization_7/mul_2Mul0Encoder/layer_normalization_7/Reshape_1:output:0:Encoder/layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#Encoder/layer_normalization_7/mul_2?
0Encoder/layer_normalization_7/add/ReadVariableOpReadVariableOp9encoder_layer_normalization_7_add_readvariableop_resource*
_output_shapes
:*
dtype022
0Encoder/layer_normalization_7/add/ReadVariableOp?
!Encoder/layer_normalization_7/addAddV2'Encoder/layer_normalization_7/mul_2:z:08Encoder/layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!Encoder/layer_normalization_7/add}
Encoder/lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Encoder/lambda_1/truediv/y?
Encoder/lambda_1/truedivRealDiv%Encoder/layer_normalization_7/add:z:0#Encoder/lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/truediv?
Encoder/lambda_1/ExpExpEncoder/lambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/Exp?
Encoder/lambda_1/mulMul%Encoder/layer_normalization_6/add:z:0Encoder/lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/mulx
Encoder/lambda_1/ShapeShapeEncoder/lambda_1/mul:z:0*
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
:?????????*
dtype0*
seed???)*
seed2??}25
3Encoder/lambda_1/random_normal/RandomStandardNormal?
"Encoder/lambda_1/random_normal/mulMul<Encoder/lambda_1/random_normal/RandomStandardNormal:output:0.Encoder/lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2$
"Encoder/lambda_1/random_normal/mul?
Encoder/lambda_1/random_normalAdd&Encoder/lambda_1/random_normal/mul:z:0,Encoder/lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2 
Encoder/lambda_1/random_normal?
Encoder/lambda_1/addAddV2%Encoder/layer_normalization_6/add:z:0"Encoder/lambda_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/lambda_1/add?
IdentityIdentityEncoder/lambda_1/add:z:0(^Encoder/dense_14/BiasAdd/ReadVariableOp'^Encoder/dense_14/MatMul/ReadVariableOp(^Encoder/dense_15/BiasAdd/ReadVariableOp'^Encoder/dense_15/MatMul/ReadVariableOp(^Encoder/dense_16/BiasAdd/ReadVariableOp'^Encoder/dense_16/MatMul/ReadVariableOp(^Encoder/dense_17/BiasAdd/ReadVariableOp'^Encoder/dense_17/MatMul/ReadVariableOp1^Encoder/layer_normalization_4/add/ReadVariableOp3^Encoder/layer_normalization_4/mul_2/ReadVariableOp1^Encoder/layer_normalization_5/add/ReadVariableOp3^Encoder/layer_normalization_5/mul_2/ReadVariableOp1^Encoder/layer_normalization_6/add/ReadVariableOp3^Encoder/layer_normalization_6/mul_2/ReadVariableOp1^Encoder/layer_normalization_7/add/ReadVariableOp3^Encoder/layer_normalization_7/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2R
'Encoder/dense_14/BiasAdd/ReadVariableOp'Encoder/dense_14/BiasAdd/ReadVariableOp2P
&Encoder/dense_14/MatMul/ReadVariableOp&Encoder/dense_14/MatMul/ReadVariableOp2R
'Encoder/dense_15/BiasAdd/ReadVariableOp'Encoder/dense_15/BiasAdd/ReadVariableOp2P
&Encoder/dense_15/MatMul/ReadVariableOp&Encoder/dense_15/MatMul/ReadVariableOp2R
'Encoder/dense_16/BiasAdd/ReadVariableOp'Encoder/dense_16/BiasAdd/ReadVariableOp2P
&Encoder/dense_16/MatMul/ReadVariableOp&Encoder/dense_16/MatMul/ReadVariableOp2R
'Encoder/dense_17/BiasAdd/ReadVariableOp'Encoder/dense_17/BiasAdd/ReadVariableOp2P
&Encoder/dense_17/MatMul/ReadVariableOp&Encoder/dense_17/MatMul/ReadVariableOp2d
0Encoder/layer_normalization_4/add/ReadVariableOp0Encoder/layer_normalization_4/add/ReadVariableOp2h
2Encoder/layer_normalization_4/mul_2/ReadVariableOp2Encoder/layer_normalization_4/mul_2/ReadVariableOp2d
0Encoder/layer_normalization_5/add/ReadVariableOp0Encoder/layer_normalization_5/add/ReadVariableOp2h
2Encoder/layer_normalization_5/mul_2/ReadVariableOp2Encoder/layer_normalization_5/mul_2/ReadVariableOp2d
0Encoder/layer_normalization_6/add/ReadVariableOp0Encoder/layer_normalization_6/add/ReadVariableOp2h
2Encoder/layer_normalization_6/mul_2/ReadVariableOp2Encoder/layer_normalization_6/mul_2/ReadVariableOp2d
0Encoder/layer_normalization_7/add/ReadVariableOp0Encoder/layer_normalization_7/add/ReadVariableOp2h
2Encoder/layer_normalization_7/mul_2/ReadVariableOp2Encoder/layer_normalization_7/mul_2/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?"
?
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_141929

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
q
D__inference_lambda_1_layer_call_and_return_conditional_losses_142257

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
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpT
mulMulinputsExp:y:0*
T0*'
_output_shapes
:?????????2
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
:?????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal`
addAddV2inputsrandom_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_143121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
q
D__inference_lambda_1_layer_call_and_return_conditional_losses_142241

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
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpT
mulMulinputsExp:y:0*
T0*'
_output_shapes
:?????????2
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
:?????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normal`
addAddV2inputsrandom_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
(__inference_Encoder_layer_call_fn_142967

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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1423722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_142531
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1418272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_142093

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_143222

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_142209

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142372

inputs
dense_14_142327
dense_14_142329 
layer_normalization_4_142333 
layer_normalization_4_142335
dense_15_142339
dense_15_142341 
layer_normalization_5_142344 
layer_normalization_5_142346
dense_17_142350
dense_17_142352
dense_16_142355
dense_16_142357 
layer_normalization_6_142360 
layer_normalization_6_142362 
layer_normalization_7_142365 
layer_normalization_7_142367
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?-layer_normalization_4/StatefulPartitionedCall?-layer_normalization_5/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_142327dense_14_142329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1418412"
 dense_14/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418692#
!dropout_1/StatefulPartitionedCall?
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0layer_normalization_4_142333layer_normalization_4_142335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_1419292/
-layer_normalization_4/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_8_layer_call_and_return_conditional_losses_1419502
re_lu_8/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0dense_15_142339dense_15_142341*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1419682"
 dense_15/StatefulPartitionedCall?
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0layer_normalization_5_142344layer_normalization_5_142346*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_1420262/
-layer_normalization_5/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_9_layer_call_and_return_conditional_losses_1420472
re_lu_9/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_17_142350dense_17_142352*
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
GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1420662"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_16_142355dense_16_142357*
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
GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1420932"
 dense_16/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0layer_normalization_6_142360layer_normalization_6_142362*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_1421512/
-layer_normalization_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0layer_normalization_7_142365layer_normalization_7_142367*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_1422092/
-layer_normalization_7/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_7/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422412"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_142026

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_1_layer_call_fn_143045

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_15_layer_call_fn_143130

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
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1419682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_4_layer_call_fn_143101

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_1419292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_6_layer_call_fn_143282

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_1421512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_8_layer_call_and_return_conditional_losses_143106

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_143092

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_Encoder_layer_call_fn_143004

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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1424572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_5_layer_call_fn_143181

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_1420262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?F
?
"__inference__traced_restore_143506
file_prefix$
 assignvariableop_dense_14_kernel$
 assignvariableop_1_dense_14_bias2
.assignvariableop_2_layer_normalization_4_gamma1
-assignvariableop_3_layer_normalization_4_beta&
"assignvariableop_4_dense_15_kernel$
 assignvariableop_5_dense_15_bias2
.assignvariableop_6_layer_normalization_5_gamma1
-assignvariableop_7_layer_normalization_5_beta&
"assignvariableop_8_dense_16_kernel$
 assignvariableop_9_dense_16_bias'
#assignvariableop_10_dense_17_kernel%
!assignvariableop_11_dense_17_bias3
/assignvariableop_12_layer_normalization_6_gamma2
.assignvariableop_13_layer_normalization_6_beta3
/assignvariableop_14_layer_normalization_7_gamma2
.assignvariableop_15_layer_normalization_7_beta
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_4_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_4_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_15_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_15_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_5_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_5_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_16_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_16_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_17_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_17_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_6_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_layer_normalization_6_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_layer_normalization_7_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_layer_normalization_7_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
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
?
_
C__inference_re_lu_9_layer_call_and_return_conditional_losses_143186

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_141869

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_143172

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3z
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*(
_output_shapes
:??????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
mul_2/ReadVariableOpz
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpm
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_14_layer_call_fn_143023

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
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1418412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_143202

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_7_layer_call_fn_143333

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_1422092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_143014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_142151

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
D__inference_lambda_1_layer_call_and_return_conditional_losses_143365
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
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpV
mulMulinputs_0Exp:y:0*
T0*'
_output_shapes
:?????????2
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
:?????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normalb
addAddV2inputs_0random_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
s
D__inference_lambda_1_layer_call_and_return_conditional_losses_143349
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
:?????????2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:?????????2
ExpV
mulMulinputs_0Exp:y:0*
T0*'
_output_shapes
:?????????2
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
:?????????*
dtype0*
seed???)*
seed2ȩ{2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
random_normalb
addAddV2inputs_0random_normal:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142930

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3layer_normalization_4_mul_2_readvariableop_resource5
1layer_normalization_4_add_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource7
3layer_normalization_5_mul_2_readvariableop_resource5
1layer_normalization_5_add_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource7
3layer_normalization_6_mul_2_readvariableop_resource5
1layer_normalization_6_add_readvariableop_resource7
3layer_normalization_7_mul_2_readvariableop_resource5
1layer_normalization_7_add_readvariableop_resource
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?(layer_normalization_4/add/ReadVariableOp?*layer_normalization_4/mul_2/ReadVariableOp?(layer_normalization_5/add/ReadVariableOp?*layer_normalization_5/mul_2/ReadVariableOp?(layer_normalization_6/add/ReadVariableOp?*layer_normalization_6/mul_2/ReadVariableOp?(layer_normalization_7/add/ReadVariableOp?*layer_normalization_7/mul_2/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/BiasAdd?
dropout_1/IdentityIdentitydense_14/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Identity?
layer_normalization_4/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
layer_normalization_4/Shape?
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_4/strided_slice/stack?
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice/stack_1?
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice/stack_2?
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_4/strided_slice|
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_4/mul/x?
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_4/mul?
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice_1/stack?
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_4/strided_slice_1/stack_1?
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_4/strided_slice_1/stack_2?
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_4/strided_slice_1?
layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_4/mul_1/x?
layer_normalization_4/mul_1Mul&layer_normalization_4/mul_1/x:output:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_4/mul_1?
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_4/Reshape/shape/0?
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_4/Reshape/shape/3?
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul:z:0layer_normalization_4/mul_1:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_4/Reshape/shape?
layer_normalization_4/ReshapeReshapedropout_1/Identity:output:0,layer_normalization_4/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_4/Reshape
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_4/Const?
layer_normalization_4/Fill/dimsPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_4/Fill/dims?
layer_normalization_4/FillFill(layer_normalization_4/Fill/dims:output:0$layer_normalization_4/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_4/Fill?
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_4/Const_1?
!layer_normalization_4/Fill_1/dimsPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_4/Fill_1/dims?
layer_normalization_4/Fill_1Fill*layer_normalization_4/Fill_1/dims:output:0&layer_normalization_4/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_4/Fill_1?
layer_normalization_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_4/Const_2?
layer_normalization_4/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_4/Const_3?
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/Fill:output:0%layer_normalization_4/Fill_1:output:0&layer_normalization_4/Const_2:output:0&layer_normalization_4/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_4/FusedBatchNormV3?
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_4/Reshape_1?
*layer_normalization_4/mul_2/ReadVariableOpReadVariableOp3layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_4/mul_2/ReadVariableOp?
layer_normalization_4/mul_2Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_4/mul_2?
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_4/add/ReadVariableOp?
layer_normalization_4/addAddV2layer_normalization_4/mul_2:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_4/addv
re_lu_8/ReluRelulayer_normalization_4/add:z:0*
T0*(
_output_shapes
:??????????2
re_lu_8/Relu?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulre_lu_8/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/BiasAdd?
layer_normalization_5/ShapeShapedense_15/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization_5/Shape?
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_5/strided_slice/stack?
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice/stack_1?
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice/stack_2?
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_5/strided_slice|
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_5/mul/x?
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_5/mul?
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice_1/stack?
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_5/strided_slice_1/stack_1?
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_5/strided_slice_1/stack_2?
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_5/strided_slice_1?
layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_5/mul_1/x?
layer_normalization_5/mul_1Mul&layer_normalization_5/mul_1/x:output:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_5/mul_1?
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_5/Reshape/shape/0?
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_5/Reshape/shape/3?
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul:z:0layer_normalization_5/mul_1:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_5/Reshape/shape?
layer_normalization_5/ReshapeReshapedense_15/BiasAdd:output:0,layer_normalization_5/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_5/Reshape
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_5/Const?
layer_normalization_5/Fill/dimsPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_5/Fill/dims?
layer_normalization_5/FillFill(layer_normalization_5/Fill/dims:output:0$layer_normalization_5/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_5/Fill?
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_5/Const_1?
!layer_normalization_5/Fill_1/dimsPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_5/Fill_1/dims?
layer_normalization_5/Fill_1Fill*layer_normalization_5/Fill_1/dims:output:0&layer_normalization_5/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_5/Fill_1?
layer_normalization_5/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_5/Const_2?
layer_normalization_5/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_5/Const_3?
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/Fill:output:0%layer_normalization_5/Fill_1:output:0&layer_normalization_5/Const_2:output:0&layer_normalization_5/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_5/FusedBatchNormV3?
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_5/Reshape_1?
*layer_normalization_5/mul_2/ReadVariableOpReadVariableOp3layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_5/mul_2/ReadVariableOp?
layer_normalization_5/mul_2Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_5/mul_2?
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_5/add/ReadVariableOp?
layer_normalization_5/addAddV2layer_normalization_5/mul_2:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_5/addv
re_lu_9/ReluRelulayer_normalization_5/add:z:0*
T0*(
_output_shapes
:??????????2
re_lu_9/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulre_lu_9/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdds
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Tanh?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulre_lu_9/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Tanh{
layer_normalization_6/ShapeShapedense_16/Tanh:y:0*
T0*
_output_shapes
:2
layer_normalization_6/Shape?
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_6/strided_slice/stack?
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_1?
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_2?
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_6/strided_slice|
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul/x?
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul?
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice_1/stack?
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_1?
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_2?
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_6/strided_slice_1?
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul_1/x?
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul_1?
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/0?
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/3?
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_6/Reshape/shape?
layer_normalization_6/ReshapeReshapedense_16/Tanh:y:0,layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_6/Reshape
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_6/Const?
layer_normalization_6/Fill/dimsPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_6/Fill/dims?
layer_normalization_6/FillFill(layer_normalization_6/Fill/dims:output:0$layer_normalization_6/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/Fill?
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_6/Const_1?
!layer_normalization_6/Fill_1/dimsPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_6/Fill_1/dims?
layer_normalization_6/Fill_1Fill*layer_normalization_6/Fill_1/dims:output:0&layer_normalization_6/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/Fill_1?
layer_normalization_6/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_2?
layer_normalization_6/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_3?
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/Fill:output:0%layer_normalization_6/Fill_1:output:0&layer_normalization_6/Const_2:output:0&layer_normalization_6/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_6/FusedBatchNormV3?
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_6/Reshape_1?
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_6/mul_2/ReadVariableOp?
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/mul_2?
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_6/add/ReadVariableOp?
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/add{
layer_normalization_7/ShapeShapedense_17/Tanh:y:0*
T0*
_output_shapes
:2
layer_normalization_7/Shape?
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_7/strided_slice/stack?
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_1?
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_2?
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_7/strided_slice|
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul/x?
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul?
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice_1/stack?
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_1?
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_2?
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_7/strided_slice_1?
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul_1/x?
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul_1?
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/0?
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/3?
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_7/Reshape/shape?
layer_normalization_7/ReshapeReshapedense_17/Tanh:y:0,layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_7/Reshape
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_7/Const?
layer_normalization_7/Fill/dimsPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_7/Fill/dims?
layer_normalization_7/FillFill(layer_normalization_7/Fill/dims:output:0$layer_normalization_7/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/Fill?
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_7/Const_1?
!layer_normalization_7/Fill_1/dimsPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_7/Fill_1/dims?
layer_normalization_7/Fill_1Fill*layer_normalization_7/Fill_1/dims:output:0&layer_normalization_7/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/Fill_1?
layer_normalization_7/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_2?
layer_normalization_7/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_3?
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/Fill:output:0%layer_normalization_7/Fill_1:output:0&layer_normalization_7/Const_2:output:0&layer_normalization_7/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_7/FusedBatchNormV3?
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_7/Reshape_1?
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_7/mul_2/ReadVariableOp?
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_7/mul_2?
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_7/add/ReadVariableOp?
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_7/addm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y?
lambda_1/truedivRealDivlayer_normalization_7/add:z:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Exp?
lambda_1/mulMullayer_normalization_6/add:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_1/mul`
lambda_1/ShapeShapelambda_1/mul:z:0*
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
:?????????*
dtype0*
seed???)*
seed2??L2-
+lambda_1/random_normal/RandomStandardNormal?
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal/mul?
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal?
lambda_1/addAddV2layer_normalization_6/add:z:0lambda_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/add?
IdentityIdentitylambda_1/add:z:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_2/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_2/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_2/ReadVariableOp*layer_normalization_4/mul_2/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_2/ReadVariableOp*layer_normalization_5/mul_2/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
(__inference_Encoder_layer_call_fn_142492
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

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1424572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
~
)__inference_dense_16_layer_call_fn_143211

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
GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1420932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
)__inference_lambda_1_layer_call_fn_143377
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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?<
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142321
input_5
dense_14_142276
dense_14_142278 
layer_normalization_4_142282 
layer_normalization_4_142284
dense_15_142288
dense_15_142290 
layer_normalization_5_142293 
layer_normalization_5_142295
dense_17_142299
dense_17_142301
dense_16_142304
dense_16_142306 
layer_normalization_6_142309 
layer_normalization_6_142311 
layer_normalization_7_142314 
layer_normalization_7_142316
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?-layer_normalization_4/StatefulPartitionedCall?-layer_normalization_5/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_14_142276dense_14_142278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1418412"
 dense_14/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418742
dropout_1/PartitionedCall?
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0layer_normalization_4_142282layer_normalization_4_142284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_1419292/
-layer_normalization_4/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_8_layer_call_and_return_conditional_losses_1419502
re_lu_8/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0dense_15_142288dense_15_142290*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1419682"
 dense_15/StatefulPartitionedCall?
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0layer_normalization_5_142293layer_normalization_5_142295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_1420262/
-layer_normalization_5/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_9_layer_call_and_return_conditional_losses_1420472
re_lu_9/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_17_142299dense_17_142301*
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
GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1420662"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_16_142304dense_16_142306*
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
GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1420932"
 dense_16/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0layer_normalization_6_142309layer_normalization_6_142311*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_1421512/
-layer_normalization_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0layer_normalization_7_142314layer_normalization_7_142316*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_1422092/
-layer_normalization_7/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422572"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
_
C__inference_re_lu_9_layer_call_and_return_conditional_losses_142047

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_17_layer_call_fn_143231

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
GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1420662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_142066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_8_layer_call_and_return_conditional_losses_141950

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_141874

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_143035

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142457

inputs
dense_14_142412
dense_14_142414 
layer_normalization_4_142418 
layer_normalization_4_142420
dense_15_142424
dense_15_142426 
layer_normalization_5_142429 
layer_normalization_5_142431
dense_17_142435
dense_17_142437
dense_16_142440
dense_16_142442 
layer_normalization_6_142445 
layer_normalization_6_142447 
layer_normalization_7_142450 
layer_normalization_7_142452
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?-layer_normalization_4/StatefulPartitionedCall?-layer_normalization_5/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_142412dense_14_142414*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1418412"
 dense_14/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418742
dropout_1/PartitionedCall?
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0layer_normalization_4_142418layer_normalization_4_142420*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_1419292/
-layer_normalization_4/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_8_layer_call_and_return_conditional_losses_1419502
re_lu_8/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0dense_15_142424dense_15_142426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1419682"
 dense_15/StatefulPartitionedCall?
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0layer_normalization_5_142429layer_normalization_5_142431*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_1420262/
-layer_normalization_5/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_9_layer_call_and_return_conditional_losses_1420472
re_lu_9/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_17_142435dense_17_142437*
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
GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1420662"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_16_142440dense_16_142442*
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
GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1420932"
 dense_16/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0layer_normalization_6_142445layer_normalization_6_142447*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_1421512/
-layer_normalization_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0layer_normalization_7_142450layer_normalization_7_142452*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_1422092/
-layer_normalization_7/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422572"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
)__inference_lambda_1_layer_call_fn_143371
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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
D
(__inference_re_lu_9_layer_call_fn_143191

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_9_layer_call_and_return_conditional_losses_1420472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142734

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3layer_normalization_4_mul_2_readvariableop_resource5
1layer_normalization_4_add_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource7
3layer_normalization_5_mul_2_readvariableop_resource5
1layer_normalization_5_add_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource7
3layer_normalization_6_mul_2_readvariableop_resource5
1layer_normalization_6_add_readvariableop_resource7
3layer_normalization_7_mul_2_readvariableop_resource5
1layer_normalization_7_add_readvariableop_resource
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?(layer_normalization_4/add/ReadVariableOp?*layer_normalization_4/mul_2/ReadVariableOp?(layer_normalization_5/add/ReadVariableOp?*layer_normalization_5/mul_2/ReadVariableOp?(layer_normalization_6/add/ReadVariableOp?*layer_normalization_6/mul_2/ReadVariableOp?(layer_normalization_7/add/ReadVariableOp?*layer_normalization_7/mul_2/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_14/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul{
dropout_1/dropout/ShapeShapedense_14/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
layer_normalization_4/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
layer_normalization_4/Shape?
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_4/strided_slice/stack?
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice/stack_1?
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice/stack_2?
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_4/strided_slice|
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_4/mul/x?
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_4/mul?
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_4/strided_slice_1/stack?
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_4/strided_slice_1/stack_1?
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_4/strided_slice_1/stack_2?
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_4/strided_slice_1?
layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_4/mul_1/x?
layer_normalization_4/mul_1Mul&layer_normalization_4/mul_1/x:output:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_4/mul_1?
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_4/Reshape/shape/0?
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_4/Reshape/shape/3?
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul:z:0layer_normalization_4/mul_1:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_4/Reshape/shape?
layer_normalization_4/ReshapeReshapedropout_1/dropout/Mul_1:z:0,layer_normalization_4/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_4/Reshape
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_4/Const?
layer_normalization_4/Fill/dimsPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_4/Fill/dims?
layer_normalization_4/FillFill(layer_normalization_4/Fill/dims:output:0$layer_normalization_4/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_4/Fill?
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_4/Const_1?
!layer_normalization_4/Fill_1/dimsPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_4/Fill_1/dims?
layer_normalization_4/Fill_1Fill*layer_normalization_4/Fill_1/dims:output:0&layer_normalization_4/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_4/Fill_1?
layer_normalization_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_4/Const_2?
layer_normalization_4/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_4/Const_3?
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/Fill:output:0%layer_normalization_4/Fill_1:output:0&layer_normalization_4/Const_2:output:0&layer_normalization_4/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_4/FusedBatchNormV3?
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_4/Reshape_1?
*layer_normalization_4/mul_2/ReadVariableOpReadVariableOp3layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_4/mul_2/ReadVariableOp?
layer_normalization_4/mul_2Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_4/mul_2?
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_4/add/ReadVariableOp?
layer_normalization_4/addAddV2layer_normalization_4/mul_2:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_4/addv
re_lu_8/ReluRelulayer_normalization_4/add:z:0*
T0*(
_output_shapes
:??????????2
re_lu_8/Relu?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulre_lu_8/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/BiasAdd?
layer_normalization_5/ShapeShapedense_15/BiasAdd:output:0*
T0*
_output_shapes
:2
layer_normalization_5/Shape?
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_5/strided_slice/stack?
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice/stack_1?
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice/stack_2?
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_5/strided_slice|
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_5/mul/x?
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_5/mul?
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_5/strided_slice_1/stack?
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_5/strided_slice_1/stack_1?
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_5/strided_slice_1/stack_2?
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_5/strided_slice_1?
layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_5/mul_1/x?
layer_normalization_5/mul_1Mul&layer_normalization_5/mul_1/x:output:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_5/mul_1?
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_5/Reshape/shape/0?
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_5/Reshape/shape/3?
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul:z:0layer_normalization_5/mul_1:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_5/Reshape/shape?
layer_normalization_5/ReshapeReshapedense_15/BiasAdd:output:0,layer_normalization_5/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_5/Reshape
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_5/Const?
layer_normalization_5/Fill/dimsPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_5/Fill/dims?
layer_normalization_5/FillFill(layer_normalization_5/Fill/dims:output:0$layer_normalization_5/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_5/Fill?
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_5/Const_1?
!layer_normalization_5/Fill_1/dimsPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_5/Fill_1/dims?
layer_normalization_5/Fill_1Fill*layer_normalization_5/Fill_1/dims:output:0&layer_normalization_5/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_5/Fill_1?
layer_normalization_5/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_5/Const_2?
layer_normalization_5/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_5/Const_3?
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/Fill:output:0%layer_normalization_5/Fill_1:output:0&layer_normalization_5/Const_2:output:0&layer_normalization_5/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_5/FusedBatchNormV3?
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*(
_output_shapes
:??????????2!
layer_normalization_5/Reshape_1?
*layer_normalization_5/mul_2/ReadVariableOpReadVariableOp3layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*layer_normalization_5/mul_2/ReadVariableOp?
layer_normalization_5/mul_2Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_5/mul_2?
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(layer_normalization_5/add/ReadVariableOp?
layer_normalization_5/addAddV2layer_normalization_5/mul_2:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_normalization_5/addv
re_lu_9/ReluRelulayer_normalization_5/add:z:0*
T0*(
_output_shapes
:??????????2
re_lu_9/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulre_lu_9/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdds
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Tanh?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulre_lu_9/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Tanh{
layer_normalization_6/ShapeShapedense_16/Tanh:y:0*
T0*
_output_shapes
:2
layer_normalization_6/Shape?
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_6/strided_slice/stack?
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_1?
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_2?
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_6/strided_slice|
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul/x?
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul?
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice_1/stack?
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_1?
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_2?
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_6/strided_slice_1?
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul_1/x?
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul_1?
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/0?
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/3?
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_6/Reshape/shape?
layer_normalization_6/ReshapeReshapedense_16/Tanh:y:0,layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_6/Reshape
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_6/Const?
layer_normalization_6/Fill/dimsPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_6/Fill/dims?
layer_normalization_6/FillFill(layer_normalization_6/Fill/dims:output:0$layer_normalization_6/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/Fill?
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_6/Const_1?
!layer_normalization_6/Fill_1/dimsPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_6/Fill_1/dims?
layer_normalization_6/Fill_1Fill*layer_normalization_6/Fill_1/dims:output:0&layer_normalization_6/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/Fill_1?
layer_normalization_6/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_2?
layer_normalization_6/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_3?
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/Fill:output:0%layer_normalization_6/Fill_1:output:0&layer_normalization_6/Const_2:output:0&layer_normalization_6/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_6/FusedBatchNormV3?
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_6/Reshape_1?
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_6/mul_2/ReadVariableOp?
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/mul_2?
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_6/add/ReadVariableOp?
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/add{
layer_normalization_7/ShapeShapedense_17/Tanh:y:0*
T0*
_output_shapes
:2
layer_normalization_7/Shape?
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_7/strided_slice/stack?
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_1?
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_2?
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_7/strided_slice|
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul/x?
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul?
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice_1/stack?
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_1?
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_2?
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_7/strided_slice_1?
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul_1/x?
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul_1?
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/0?
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/3?
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_7/Reshape/shape?
layer_normalization_7/ReshapeReshapedense_17/Tanh:y:0,layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_7/Reshape
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization_7/Const?
layer_normalization_7/Fill/dimsPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_7/Fill/dims?
layer_normalization_7/FillFill(layer_normalization_7/Fill/dims:output:0$layer_normalization_7/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/Fill?
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_7/Const_1?
!layer_normalization_7/Fill_1/dimsPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_7/Fill_1/dims?
layer_normalization_7/Fill_1Fill*layer_normalization_7/Fill_1/dims:output:0&layer_normalization_7/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/Fill_1?
layer_normalization_7/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_2?
layer_normalization_7/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_3?
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/Fill:output:0%layer_normalization_7/Fill_1:output:0&layer_normalization_7/Const_2:output:0&layer_normalization_7/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_7/FusedBatchNormV3?
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_7/Reshape_1?
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_7/mul_2/ReadVariableOp?
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_7/mul_2?
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_7/add/ReadVariableOp?
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_7/addm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y?
lambda_1/truedivRealDivlayer_normalization_7/add:z:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Exp?
lambda_1/mulMullayer_normalization_6/add:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
lambda_1/mul`
lambda_1/ShapeShapelambda_1/mul:z:0*
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
:?????????*
dtype0*
seed???)*
seed2???2-
+lambda_1/random_normal/RandomStandardNormal?
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal/mul?
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/random_normal?
lambda_1/addAddV2layer_normalization_6/add:z:0lambda_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/add?
IdentityIdentitylambda_1/add:z:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_2/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_2/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_2/ReadVariableOp*layer_normalization_4/mul_2/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_2/ReadVariableOp*layer_normalization_5/mul_2/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_143324

inputs!
mul_2_readvariableop_resource
add_readvariableop_resource
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
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
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
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
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add?
IdentityIdentityadd:z:0^add/ReadVariableOp^mul_2/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_143050

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?,
?
__inference__traced_save_143448
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop6savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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
: :	?:?:?:?:
??:?:?:?:	?::	?:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::%!

_output_shapes
:	?: 
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
:: 

_output_shapes
::

_output_shapes
: 
?=
?
C__inference_Encoder_layer_call_and_return_conditional_losses_142273
input_5
dense_14_141852
dense_14_141854 
layer_normalization_4_141940 
layer_normalization_4_141942
dense_15_141979
dense_15_141981 
layer_normalization_5_142037 
layer_normalization_5_142039
dense_17_142077
dense_17_142079
dense_16_142104
dense_16_142106 
layer_normalization_6_142162 
layer_normalization_6_142164 
layer_normalization_7_142220 
layer_normalization_7_142222
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall? lambda_1/StatefulPartitionedCall?-layer_normalization_4/StatefulPartitionedCall?-layer_normalization_5/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_14_141852dense_14_141854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_1418412"
 dense_14/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1418692#
!dropout_1/StatefulPartitionedCall?
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0layer_normalization_4_141940layer_normalization_4_141942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_1419292/
-layer_normalization_4/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_8_layer_call_and_return_conditional_losses_1419502
re_lu_8/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0dense_15_141979dense_15_141981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1419682"
 dense_15/StatefulPartitionedCall?
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0layer_normalization_5_142037layer_normalization_5_142039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_1420262/
-layer_normalization_5/StatefulPartitionedCall?
re_lu_9/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_9_layer_call_and_return_conditional_losses_1420472
re_lu_9/PartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_17_142077dense_17_142079*
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
GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1420662"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_9/PartitionedCall:output:0dense_16_142104dense_16_142106*
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
GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1420932"
 dense_16/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0layer_normalization_6_142162layer_normalization_6_142164*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_1421512/
-layer_normalization_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0layer_normalization_7_142220layer_normalization_7_142222*
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
GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_1422092/
-layer_normalization_7/StatefulPartitionedCall?
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:06layer_normalization_7/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_1422412"
 lambda_1/StatefulPartitionedCall?
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????::::::::::::::::2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5
?
D
(__inference_re_lu_8_layer_call_fn_143111

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_8_layer_call_and_return_conditional_losses_1419502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?b
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
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?^
_tf_keras_network?]{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_5", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["layer_normalization_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_6", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_7", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA9WsAAAAvVXNlcnMvbGlseWh1YS9PbmVEcml2\nZSAtIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9B\nQUUwNjI4L0FBReaLt+iynTIvbmV0d29yay5wedoIPGxhbWJkYT5AAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["layer_normalization_6", 0, 0, {}], ["layer_normalization_7", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_5", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_9", "inbound_nodes": [[["layer_normalization_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["re_lu_9", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_6", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_7", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA9WsAAAAvVXNlcnMvbGlseWh1YS9PbmVEcml2\nZSAtIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9B\nQUUwNjI4L0FBReaLt+iynTIvbmV0d29yay5wedoIPGxhbWJkYT5AAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["layer_normalization_6", 0, 0, {}], ["layer_normalization_7", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
axis
	gamma
beta
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?
$trainable_variables
%	variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?
.axis
	/gamma
0beta
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?
Eaxis
	Fgamma
Gbeta
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
Laxis
	Mgamma
Nbeta
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAKAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQB0AKADfABk\nAhkAZAMbAKEBFAChAaEBFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA9WsAAAAvVXNlcnMvbGlseWh1YS9PbmVEcml2\nZSAtIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFLzJEL3Npbi9B\nQUUwNjI4L0FBReaLt+iynTIvbmV0d29yay5wedoIPGxhbWJkYT5AAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "network", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
0
1
2
3
(4
)5
/6
07
98
:9
?10
@11
F12
G13
M14
N15"
trackable_list_wrapper
?
0
1
2
3
(4
)5
/6
07
98
:9
?10
@11
F12
G13
M14
N15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wmetrics
trainable_variables
	variables
regularization_losses
Xlayer_metrics

Ylayers
Znon_trainable_variables
[layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
": 	?2dense_14/kernel
:?2dense_14/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\metrics
trainable_variables
	variables
regularization_losses
]layer_metrics

^layers
_non_trainable_variables
`layer_regularization_losses
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
ametrics
trainable_variables
	variables
regularization_losses
blayer_metrics

clayers
dnon_trainable_variables
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer_normalization_4/gamma
):'?2layer_normalization_4/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fmetrics
 trainable_variables
!	variables
"regularization_losses
glayer_metrics

hlayers
inon_trainable_variables
jlayer_regularization_losses
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
kmetrics
$trainable_variables
%	variables
&regularization_losses
llayer_metrics

mlayers
nnon_trainable_variables
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_15/kernel
:?2dense_15/bias
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
?
pmetrics
*trainable_variables
+	variables
,regularization_losses
qlayer_metrics

rlayers
snon_trainable_variables
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2layer_normalization_5/gamma
):'?2layer_normalization_5/beta
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
umetrics
1trainable_variables
2	variables
3regularization_losses
vlayer_metrics

wlayers
xnon_trainable_variables
ylayer_regularization_losses
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
zmetrics
5trainable_variables
6	variables
7regularization_losses
{layer_metrics

|layers
}non_trainable_variables
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_16/kernel
:2dense_16/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
;trainable_variables
<	variables
=regularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_17/kernel
:2dense_17/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Atrainable_variables
B	variables
Cregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_6/gamma
(:&2layer_normalization_6/beta
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Htrainable_variables
I	variables
Jregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_7/gamma
(:&2layer_normalization_7/beta
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Otrainable_variables
P	variables
Qregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?metrics
Strainable_variables
T	variables
Uregularization_losses
?layer_metrics
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
12"
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
?2?
(__inference_Encoder_layer_call_fn_142407
(__inference_Encoder_layer_call_fn_142492
(__inference_Encoder_layer_call_fn_143004
(__inference_Encoder_layer_call_fn_142967?
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
?2?
C__inference_Encoder_layer_call_and_return_conditional_losses_142734
C__inference_Encoder_layer_call_and_return_conditional_losses_142321
C__inference_Encoder_layer_call_and_return_conditional_losses_142930
C__inference_Encoder_layer_call_and_return_conditional_losses_142273?
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
!__inference__wrapped_model_141827?
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
?2?
)__inference_dense_14_layer_call_fn_143023?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_143014?
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
*__inference_dropout_1_layer_call_fn_143050
*__inference_dropout_1_layer_call_fn_143045?
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
E__inference_dropout_1_layer_call_and_return_conditional_losses_143035
E__inference_dropout_1_layer_call_and_return_conditional_losses_143040?
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
6__inference_layer_normalization_4_layer_call_fn_143101?
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_143092?
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
(__inference_re_lu_8_layer_call_fn_143111?
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
C__inference_re_lu_8_layer_call_and_return_conditional_losses_143106?
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
)__inference_dense_15_layer_call_fn_143130?
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
D__inference_dense_15_layer_call_and_return_conditional_losses_143121?
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
6__inference_layer_normalization_5_layer_call_fn_143181?
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
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_143172?
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
(__inference_re_lu_9_layer_call_fn_143191?
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
C__inference_re_lu_9_layer_call_and_return_conditional_losses_143186?
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
)__inference_dense_16_layer_call_fn_143211?
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
D__inference_dense_16_layer_call_and_return_conditional_losses_143202?
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
)__inference_dense_17_layer_call_fn_143231?
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
D__inference_dense_17_layer_call_and_return_conditional_losses_143222?
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
6__inference_layer_normalization_6_layer_call_fn_143282?
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
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_143273?
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
6__inference_layer_normalization_7_layer_call_fn_143333?
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
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_143324?
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
)__inference_lambda_1_layer_call_fn_143371
)__inference_lambda_1_layer_call_fn_143377?
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_143349
D__inference_lambda_1_layer_call_and_return_conditional_losses_143365?
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
$__inference_signature_wrapper_142531input_5"?
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
C__inference_Encoder_layer_call_and_return_conditional_losses_142273s()/0?@9:FGMN8?5
.?+
!?
input_5?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_Encoder_layer_call_and_return_conditional_losses_142321s()/0?@9:FGMN8?5
.?+
!?
input_5?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_Encoder_layer_call_and_return_conditional_losses_142734r()/0?@9:FGMN7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_Encoder_layer_call_and_return_conditional_losses_142930r()/0?@9:FGMN7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
(__inference_Encoder_layer_call_fn_142407f()/0?@9:FGMN8?5
.?+
!?
input_5?????????
p

 
? "???????????
(__inference_Encoder_layer_call_fn_142492f()/0?@9:FGMN8?5
.?+
!?
input_5?????????
p 

 
? "???????????
(__inference_Encoder_layer_call_fn_142967e()/0?@9:FGMN7?4
-?*
 ?
inputs?????????
p

 
? "???????????
(__inference_Encoder_layer_call_fn_143004e()/0?@9:FGMN7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
!__inference__wrapped_model_141827y()/0?@9:FGMN0?-
&?#
!?
input_5?????????
? "3?0
.
lambda_1"?
lambda_1??????????
D__inference_dense_14_layer_call_and_return_conditional_losses_143014]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? }
)__inference_dense_14_layer_call_fn_143023P/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_dense_15_layer_call_and_return_conditional_losses_143121^()0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_15_layer_call_fn_143130Q()0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_16_layer_call_and_return_conditional_losses_143202]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_16_layer_call_fn_143211P9:0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dense_17_layer_call_and_return_conditional_losses_143222]?@0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_17_layer_call_fn_143231P?@0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_143035^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_143040^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? 
*__inference_dropout_1_layer_call_fn_143045Q4?1
*?'
!?
inputs??????????
p
? "???????????
*__inference_dropout_1_layer_call_fn_143050Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_lambda_1_layer_call_and_return_conditional_losses_143349?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "%?"
?
0?????????
? ?
D__inference_lambda_1_layer_call_and_return_conditional_losses_143365?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "%?"
?
0?????????
? ?
)__inference_lambda_1_layer_call_fn_143371~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "???????????
)__inference_lambda_1_layer_call_fn_143377~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "???????????
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_143092^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_layer_normalization_4_layer_call_fn_143101Q0?-
&?#
!?
inputs??????????
? "????????????
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_143172^/00?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_layer_normalization_5_layer_call_fn_143181Q/00?-
&?#
!?
inputs??????????
? "????????????
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_143273\FG/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
6__inference_layer_normalization_6_layer_call_fn_143282OFG/?,
%?"
 ?
inputs?????????
? "???????????
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_143324\MN/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
6__inference_layer_normalization_7_layer_call_fn_143333OMN/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_re_lu_8_layer_call_and_return_conditional_losses_143106Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
(__inference_re_lu_8_layer_call_fn_143111M0?-
&?#
!?
inputs??????????
? "????????????
C__inference_re_lu_9_layer_call_and_return_conditional_losses_143186Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
(__inference_re_lu_9_layer_call_fn_143191M0?-
&?#
!?
inputs??????????
? "????????????
$__inference_signature_wrapper_142531?()/0?@9:FGMN;?8
? 
1?.
,
input_5!?
input_5?????????"3?0
.
lambda_1"?
lambda_1?????????