??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
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
conv1d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_2/kernel
?
-conv1d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_2/kernel*"
_output_shapes
:*
dtype0
?
conv1d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv1d_transpose_2/bias

+conv1d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_2/bias*
_output_shapes
:*
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0
?
conv1d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_3/kernel
?
-conv1d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_3/kernel*"
_output_shapes
:*
dtype0
?
conv1d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv1d_transpose_3/bias

+conv1d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_3/bias*
_output_shapes
:*
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?5
value?5B?5 B?5
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
regularization_losses
trainable_variables
	keras_api

signatures
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
^

Rkernel
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
?
0
1
2
3
4
(5
)6
/7
08
19
210
;11
<12
B13
C14
D15
E16
R17
 
V
0
1
2
(3
)4
/5
06
;7
<8
B9
C10
R11
?
Wnon_trainable_variables
Xlayer_metrics
Ymetrics
	variables
Zlayer_regularization_losses
regularization_losses
trainable_variables

[layers
 
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
\non_trainable_variables
]layer_metrics
^metrics
	variables
_layer_regularization_losses
regularization_losses
trainable_variables

`layers
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
?
anon_trainable_variables
blayer_metrics
cmetrics
	variables
dlayer_regularization_losses
regularization_losses
trainable_variables

elayers
 
 
 
?
fnon_trainable_variables
glayer_metrics
hmetrics
 	variables
ilayer_regularization_losses
!regularization_losses
"trainable_variables

jlayers
 
 
 
?
knon_trainable_variables
llayer_metrics
mmetrics
$	variables
nlayer_regularization_losses
%regularization_losses
&trainable_variables

olayers
ec
VARIABLE_VALUEconv1d_transpose_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv1d_transpose_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
pnon_trainable_variables
qlayer_metrics
rmetrics
*	variables
slayer_regularization_losses
+regularization_losses
,trainable_variables

tlayers
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23
 

/0
01
?
unon_trainable_variables
vlayer_metrics
wmetrics
3	variables
xlayer_regularization_losses
4regularization_losses
5trainable_variables

ylayers
 
 
 
?
znon_trainable_variables
{layer_metrics
|metrics
7	variables
}layer_regularization_losses
8regularization_losses
9trainable_variables

~layers
ec
VARIABLE_VALUEconv1d_transpose_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv1d_transpose_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
?
non_trainable_variables
?layer_metrics
?metrics
=	variables
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
?layers
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3
 

B0
C1
?
?non_trainable_variables
?layer_metrics
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
Htrainable_variables
?layers
 
 
 
?
?non_trainable_variables
?layer_metrics
?metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
Ltrainable_variables
?layers
 
 
 
?
?non_trainable_variables
?layer_metrics
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
Ptrainable_variables
?layers
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

R0
 

R0
?
?non_trainable_variables
?layer_metrics
?metrics
S	variables
 ?layer_regularization_losses
Tregularization_losses
Utrainable_variables
?layers
*
0
1
12
23
D4
E5
 
 
 
V
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
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
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
10
21
 
 
 
 
 
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
D0
E1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_dense_6_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_6_inputdense_6/kernel%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/betaconv1d_transpose_2/kernelconv1d_transpose_2/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betaconv1d_transpose_3/kernelconv1d_transpose_3/bias&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betadense_7/kernel*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_151970
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp-conv1d_transpose_2/kernel/Read/ReadVariableOp+conv1d_transpose_2/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp-conv1d_transpose_3/kernel/Read/ReadVariableOp+conv1d_transpose_3/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOpConst*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_152803
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv1d_transpose_2/kernelconv1d_transpose_2/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv1d_transpose_3/kernelconv1d_transpose_3/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_7/kernel*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_152867??
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152658

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_151460

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_151109

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOpD
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
value	B :2	
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
:*
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
:2
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
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_10_layer_call_fn_152579

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
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1512152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_3_layer_call_fn_151309

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
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_1512992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?<
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_151688
dense_6_input
dense_6_151469 
batch_normalization_9_151498 
batch_normalization_9_151500 
batch_normalization_9_151502 
batch_normalization_9_151504
conv1d_transpose_2_151541
conv1d_transpose_2_151543!
batch_normalization_10_151572!
batch_normalization_10_151574!
batch_normalization_10_151576!
batch_normalization_10_151578
conv1d_transpose_3_151594
conv1d_transpose_3_151596!
batch_normalization_11_151625!
batch_normalization_11_151627!
batch_normalization_11_151629!
batch_normalization_11_151631
dense_7_151684
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCalldense_6_inputdense_6_151469*
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
GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1514602!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_151498batch_normalization_9_151500batch_normalization_9_151502batch_normalization_9_151504*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510252/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_1515122
re_lu_3/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1515332
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_151541conv1d_transpose_2_151543*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_1511092,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_151572batch_normalization_10_151574batch_normalization_10_151576batch_normalization_10_151578*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15121520
.batch_normalization_10/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_1515862
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_151594conv1d_transpose_3_151596*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_1512992,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_151625batch_normalization_11_151627batch_normalization_11_151629batch_normalization_11_151631*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15140520
.batch_normalization_11/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_1516392
re_lu_5/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1516592
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_151684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1516752!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
?
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_151533

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
?0
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152546

inputs
assignmovingavg_152521
assignmovingavg_1_152527)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152521*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_152521*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152521*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152521*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_152521AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152521*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152527*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_152527*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152527*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152527*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_152527AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152527*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_152505

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
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152566

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_152706

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
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_9_layer_call_fn_152469

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510252
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
?
?
7__inference_batch_normalization_11_layer_call_fn_152684

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
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1514382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_151659

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
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_151586

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_2_layer_call_fn_151119

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
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_1511092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_151970
dense_6_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_1509292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
?<
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_151795

inputs
dense_6_151746 
batch_normalization_9_151749 
batch_normalization_9_151751 
batch_normalization_9_151753 
batch_normalization_9_151755
conv1d_transpose_2_151760
conv1d_transpose_2_151762!
batch_normalization_10_151765!
batch_normalization_10_151767!
batch_normalization_10_151769!
batch_normalization_10_151771
conv1d_transpose_3_151775
conv1d_transpose_3_151777!
batch_normalization_11_151780!
batch_normalization_11_151782!
batch_normalization_11_151784!
batch_normalization_11_151786
dense_7_151791
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_151746*
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
GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1514602!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_151749batch_normalization_9_151751batch_normalization_9_151753batch_normalization_9_151755*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510252/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_1515122
re_lu_3/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1515332
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_151760conv1d_transpose_2_151762*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_1511092,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_151765batch_normalization_10_151767batch_normalization_10_151769batch_normalization_10_151771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15121520
.batch_normalization_10/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_1515862
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_151775conv1d_transpose_3_151777*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_1512992,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_151780batch_normalization_11_151782batch_normalization_11_151784batch_normalization_11_151786*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15140520
.batch_normalization_11/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_1516392
re_lu_5/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1516592
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_151791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1516752!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_151438

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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_152689

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_152487

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
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_151248

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
D
(__inference_re_lu_4_layer_call_fn_152602

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
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_1515862
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_3_layer_call_fn_152386

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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1518882
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
?
?
-__inference_sequential_3_layer_call_fn_151834
dense_6_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1517952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
?
F
*__inference_flatten_3_layer_call_fn_152711

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
GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1516592
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_3_layer_call_fn_152345

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

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1517952
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
?0
?
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_151299

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOpD
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
value	B :2	
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
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
$:"??????????????????*
paddingSAME*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_7_layer_call_and_return_conditional_losses_152719

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?0
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152436

inputs
assignmovingavg_152411
assignmovingavg_1_152417)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152411*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_152411*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152411*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152411*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_152411AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152411*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152417*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_152417*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152417*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152417*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_152417AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152417*
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
?
?
C__inference_dense_7_layer_call_and_return_conditional_losses_151675

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_151058

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
?0
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_151215

inputs
assignmovingavg_151190
assignmovingavg_1_151196)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151190*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_151190*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151190*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151190*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_151190AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151190*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151196*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_151196*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151196*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151196*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_151196AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151196*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152456

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
?
D
(__inference_re_lu_5_layer_call_fn_152694

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
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_1516392
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_151927
dense_6_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1518882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
?
?
7__inference_batch_normalization_10_layer_call_fn_152592

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
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1512482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?Q
?
"__inference__traced_restore_152867
file_prefix#
assignvariableop_dense_6_kernel2
.assignvariableop_1_batch_normalization_9_gamma1
-assignvariableop_2_batch_normalization_9_beta8
4assignvariableop_3_batch_normalization_9_moving_mean<
8assignvariableop_4_batch_normalization_9_moving_variance0
,assignvariableop_5_conv1d_transpose_2_kernel.
*assignvariableop_6_conv1d_transpose_2_bias3
/assignvariableop_7_batch_normalization_10_gamma2
.assignvariableop_8_batch_normalization_10_beta9
5assignvariableop_9_batch_normalization_10_moving_mean>
:assignvariableop_10_batch_normalization_10_moving_variance1
-assignvariableop_11_conv1d_transpose_3_kernel/
+assignvariableop_12_conv1d_transpose_3_bias4
0assignvariableop_13_batch_normalization_11_gamma3
/assignvariableop_14_batch_normalization_11_beta:
6assignvariableop_15_batch_normalization_11_moving_mean>
:assignvariableop_16_batch_normalization_11_moving_variance&
"assignvariableop_17_dense_7_kernel
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
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
AssignVariableOp_5AssignVariableOp,assignvariableop_5_conv1d_transpose_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_conv1d_transpose_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_10_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_10_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_10_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_10_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_conv1d_transpose_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv1d_transpose_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_11_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_11_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp6assignvariableop_15_batch_normalization_11_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_normalization_11_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_7_kernelIdentity_17:output:0"/device:CPU:0*
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
?<
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_151888

inputs
dense_6_151839 
batch_normalization_9_151842 
batch_normalization_9_151844 
batch_normalization_9_151846 
batch_normalization_9_151848
conv1d_transpose_2_151853
conv1d_transpose_2_151855!
batch_normalization_10_151858!
batch_normalization_10_151860!
batch_normalization_10_151862!
batch_normalization_10_151864
conv1d_transpose_3_151868
conv1d_transpose_3_151870!
batch_normalization_11_151873!
batch_normalization_11_151875!
batch_normalization_11_151877!
batch_normalization_11_151879
dense_7_151884
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_151839*
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
GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1514602!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_151842batch_normalization_9_151844batch_normalization_9_151846batch_normalization_9_151848*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510582/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_1515122
re_lu_3/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1515332
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_151853conv1d_transpose_2_151855*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_1511092,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_151858batch_normalization_10_151860batch_normalization_10_151862batch_normalization_10_151864*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15124820
.batch_normalization_10/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_1515862
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_151868conv1d_transpose_3_151870*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_1512992,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_151873batch_normalization_11_151875batch_normalization_11_151877batch_normalization_11_151879*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15143820
.batch_normalization_11/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_1516392
re_lu_5/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1516592
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_151884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1516752!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_151405

inputs
assignmovingavg_151380
assignmovingavg_1_151386)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151380*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_151380*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151380*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151380*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_151380AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151380*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151386*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_151386*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151386*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151386*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_151386AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151386*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_11_layer_call_fn_152671

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
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1514052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
n
(__inference_dense_7_layer_call_fn_152726

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1516752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_151512

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
F
*__inference_reshape_3_layer_call_fn_152510

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
GPU 2J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1515332
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
?0
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_151025

inputs
assignmovingavg_151000
assignmovingavg_1_151006)
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
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151000*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_151000*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151000*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/151000*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_151000AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/151000*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151006*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_151006*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151006*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/151006*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_151006AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/151006*
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
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152638

inputs
assignmovingavg_152613
assignmovingavg_1_152619)
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
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
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
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152613*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_152613*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152613*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/152613*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_152613AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/152613*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152619*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_152619*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152619*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/152619*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_152619AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/152619*
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
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
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
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
n
(__inference_dense_6_layer_call_fn_152400

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
GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1514602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_152597

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
D
(__inference_re_lu_3_layer_call_fn_152492

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
GPU 2J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_1515122
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
?<
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_151740
dense_6_input
dense_6_151691 
batch_normalization_9_151694 
batch_normalization_9_151696 
batch_normalization_9_151698 
batch_normalization_9_151700
conv1d_transpose_2_151705
conv1d_transpose_2_151707!
batch_normalization_10_151710!
batch_normalization_10_151712!
batch_normalization_10_151714!
batch_normalization_10_151716
conv1d_transpose_3_151720
conv1d_transpose_3_151722!
batch_normalization_11_151725!
batch_normalization_11_151727!
batch_normalization_11_151729!
batch_normalization_11_151731
dense_7_151736
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall?*conv1d_transpose_2/StatefulPartitionedCall?*conv1d_transpose_3/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCalldense_6_inputdense_6_151691*
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
GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1514602!
dense_6/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_9_151694batch_normalization_9_151696batch_normalization_9_151698batch_normalization_9_151700*
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510582/
-batch_normalization_9/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_1515122
re_lu_3/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1515332
reshape_3/PartitionedCall?
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv1d_transpose_2_151705conv1d_transpose_2_151707*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_1511092,
*conv1d_transpose_2/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_10_151710batch_normalization_10_151712batch_normalization_10_151714batch_normalization_10_151716*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_15124820
.batch_normalization_10/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_1515862
re_lu_4/PartitionedCall?
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv1d_transpose_3_151720conv1d_transpose_3_151722*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_1512992,
*conv1d_transpose_3/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_11_151725batch_normalization_11_151727batch_normalization_11_151729batch_normalization_11_151731*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_15143820
.batch_normalization_11/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_1516392
re_lu_5/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1516592
flatten_3/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_7_151736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1516752!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
?
?
6__inference_batch_normalization_9_layer_call_fn_152482

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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1510582
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
?1
?	
__inference__traced_save_152803
file_prefix-
)savev2_dense_6_kernel_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_2_kernel_read_readvariableop6
2savev2_conv1d_transpose_2_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop8
4savev2_conv1d_transpose_3_kernel_read_readvariableop6
2savev2_conv1d_transpose_3_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop
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
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop4savev2_conv1d_transpose_2_kernel_read_readvariableop2savev2_conv1d_transpose_2_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop4savev2_conv1d_transpose_3_kernel_read_readvariableop2savev2_conv1d_transpose_3_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop)savev2_dense_7_kernel_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 
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
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
??
?
!__inference__wrapped_model_150929
dense_6_input7
3sequential_3_dense_6_matmul_readvariableop_resourceH
Dsequential_3_batch_normalization_9_batchnorm_readvariableop_resourceL
Hsequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resourceJ
Fsequential_3_batch_normalization_9_batchnorm_readvariableop_1_resourceJ
Fsequential_3_batch_normalization_9_batchnorm_readvariableop_2_resourceY
Usequential_3_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_3_conv1d_transpose_2_biasadd_readvariableop_resourceI
Esequential_3_batch_normalization_10_batchnorm_readvariableop_resourceM
Isequential_3_batch_normalization_10_batchnorm_mul_readvariableop_resourceK
Gsequential_3_batch_normalization_10_batchnorm_readvariableop_1_resourceK
Gsequential_3_batch_normalization_10_batchnorm_readvariableop_2_resourceY
Usequential_3_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resourceC
?sequential_3_conv1d_transpose_3_biasadd_readvariableop_resourceI
Esequential_3_batch_normalization_11_batchnorm_readvariableop_resourceM
Isequential_3_batch_normalization_11_batchnorm_mul_readvariableop_resourceK
Gsequential_3_batch_normalization_11_batchnorm_readvariableop_1_resourceK
Gsequential_3_batch_normalization_11_batchnorm_readvariableop_2_resource7
3sequential_3_dense_7_matmul_readvariableop_resource
identity??<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2?@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2?@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1?=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2??sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?*sequential_3/dense_6/MatMul/ReadVariableOp?*sequential_3/dense_7/MatMul/ReadVariableOp?
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp?
sequential_3/dense_6/MatMulMatMuldense_6_input2sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_6/MatMul?
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOpDsequential_3_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp?
2sequential_3/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_3/batch_normalization_9/batchnorm/add/y?
0sequential_3/batch_normalization_9/batchnorm/addAddV2Csequential_3/batch_normalization_9/batchnorm/ReadVariableOp:value:0;sequential_3/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_3/batch_normalization_9/batchnorm/add?
2sequential_3/batch_normalization_9/batchnorm/RsqrtRsqrt4sequential_3/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_3/batch_normalization_9/batchnorm/Rsqrt?
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_3_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?
0sequential_3/batch_normalization_9/batchnorm/mulMul6sequential_3/batch_normalization_9/batchnorm/Rsqrt:y:0Gsequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_3/batch_normalization_9/batchnorm/mul?
2sequential_3/batch_normalization_9/batchnorm/mul_1Mul%sequential_3/dense_6/MatMul:product:04sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????24
2sequential_3/batch_normalization_9/batchnorm/mul_1?
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_3_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1?
2sequential_3/batch_normalization_9/batchnorm/mul_2MulEsequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1:value:04sequential_3/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_3/batch_normalization_9/batchnorm/mul_2?
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_3_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2?
0sequential_3/batch_normalization_9/batchnorm/subSubEsequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2:value:06sequential_3/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_3/batch_normalization_9/batchnorm/sub?
2sequential_3/batch_normalization_9/batchnorm/add_1AddV26sequential_3/batch_normalization_9/batchnorm/mul_1:z:04sequential_3/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????24
2sequential_3/batch_normalization_9/batchnorm/add_1?
sequential_3/re_lu_3/ReluRelu6sequential_3/batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
sequential_3/re_lu_3/Relu?
sequential_3/reshape_3/ShapeShape'sequential_3/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
sequential_3/reshape_3/Shape?
*sequential_3/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_3/reshape_3/strided_slice/stack?
,sequential_3/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_3/strided_slice/stack_1?
,sequential_3/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_3/strided_slice/stack_2?
$sequential_3/reshape_3/strided_sliceStridedSlice%sequential_3/reshape_3/Shape:output:03sequential_3/reshape_3/strided_slice/stack:output:05sequential_3/reshape_3/strided_slice/stack_1:output:05sequential_3/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_3/reshape_3/strided_slice?
&sequential_3/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_3/Reshape/shape/1?
&sequential_3/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_3/Reshape/shape/2?
$sequential_3/reshape_3/Reshape/shapePack-sequential_3/reshape_3/strided_slice:output:0/sequential_3/reshape_3/Reshape/shape/1:output:0/sequential_3/reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/reshape_3/Reshape/shape?
sequential_3/reshape_3/ReshapeReshape'sequential_3/re_lu_3/Relu:activations:0-sequential_3/reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2 
sequential_3/reshape_3/Reshape?
%sequential_3/conv1d_transpose_2/ShapeShape'sequential_3/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2'
%sequential_3/conv1d_transpose_2/Shape?
3sequential_3/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_3/conv1d_transpose_2/strided_slice/stack?
5sequential_3/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_2/strided_slice/stack_1?
5sequential_3/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_2/strided_slice/stack_2?
-sequential_3/conv1d_transpose_2/strided_sliceStridedSlice.sequential_3/conv1d_transpose_2/Shape:output:0<sequential_3/conv1d_transpose_2/strided_slice/stack:output:0>sequential_3/conv1d_transpose_2/strided_slice/stack_1:output:0>sequential_3/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_3/conv1d_transpose_2/strided_slice?
5sequential_3/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_2/strided_slice_1/stack?
7sequential_3/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_3/conv1d_transpose_2/strided_slice_1/stack_1?
7sequential_3/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_3/conv1d_transpose_2/strided_slice_1/stack_2?
/sequential_3/conv1d_transpose_2/strided_slice_1StridedSlice.sequential_3/conv1d_transpose_2/Shape:output:0>sequential_3/conv1d_transpose_2/strided_slice_1/stack:output:0@sequential_3/conv1d_transpose_2/strided_slice_1/stack_1:output:0@sequential_3/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_3/conv1d_transpose_2/strided_slice_1?
%sequential_3/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_3/conv1d_transpose_2/mul/y?
#sequential_3/conv1d_transpose_2/mulMul8sequential_3/conv1d_transpose_2/strided_slice_1:output:0.sequential_3/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_3/conv1d_transpose_2/mul?
'sequential_3/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_3/conv1d_transpose_2/stack/2?
%sequential_3/conv1d_transpose_2/stackPack6sequential_3/conv1d_transpose_2/strided_slice:output:0'sequential_3/conv1d_transpose_2/mul:z:00sequential_3/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/conv1d_transpose_2/stack?
?sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
;sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims'sequential_3/reshape_3/Reshape:output:0Hsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims?
Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_3_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
=sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
Dsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
>sequential_3/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice.sequential_3/conv1d_transpose_2/stack:output:0Msequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Osequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Osequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice?
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
Hsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
@sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice.sequential_3/conv1d_transpose_2/stack:output:0Osequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1?
@sequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1?
<sequential_3/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_3/conv1d_transpose_2/conv1d_transpose/concat/axis?
7sequential_3/conv1d_transpose_2/conv1d_transpose/concatConcatV2Gsequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Isequential_3/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Isequential_3/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0Esequential_3/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_3/conv1d_transpose_2/conv1d_transpose/concat?
0sequential_3/conv1d_transpose_2/conv1d_transposeConv2DBackpropInput@sequential_3/conv1d_transpose_2/conv1d_transpose/concat:output:0Fsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0Dsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_3/conv1d_transpose_2/conv1d_transpose?
8sequential_3/conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze9sequential_3/conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_3/conv1d_transpose_2/conv1d_transpose/Squeeze?
6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_3_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp?
'sequential_3/conv1d_transpose_2/BiasAddBiasAddAsequential_3/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0>sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_3/conv1d_transpose_2/BiasAdd?
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?
3sequential_3/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_3/batch_normalization_10/batchnorm/add/y?
1sequential_3/batch_normalization_10/batchnorm/addAddV2Dsequential_3/batch_normalization_10/batchnorm/ReadVariableOp:value:0<sequential_3/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_10/batchnorm/add?
3sequential_3/batch_normalization_10/batchnorm/RsqrtRsqrt5sequential_3/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_10/batchnorm/Rsqrt?
@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_3_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp?
1sequential_3/batch_normalization_10/batchnorm/mulMul7sequential_3/batch_normalization_10/batchnorm/Rsqrt:y:0Hsequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_10/batchnorm/mul?
3sequential_3/batch_normalization_10/batchnorm/mul_1Mul0sequential_3/conv1d_transpose_2/BiasAdd:output:05sequential_3/batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_3/batch_normalization_10/batchnorm/mul_1?
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_3_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?
3sequential_3/batch_normalization_10/batchnorm/mul_2MulFsequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1:value:05sequential_3/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_10/batchnorm/mul_2?
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_3_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2?
1sequential_3/batch_normalization_10/batchnorm/subSubFsequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2:value:07sequential_3/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_10/batchnorm/sub?
3sequential_3/batch_normalization_10/batchnorm/add_1AddV27sequential_3/batch_normalization_10/batchnorm/mul_1:z:05sequential_3/batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_3/batch_normalization_10/batchnorm/add_1?
sequential_3/re_lu_4/ReluRelu7sequential_3/batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_3/re_lu_4/Relu?
%sequential_3/conv1d_transpose_3/ShapeShape'sequential_3/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/conv1d_transpose_3/Shape?
3sequential_3/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_3/conv1d_transpose_3/strided_slice/stack?
5sequential_3/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_3/strided_slice/stack_1?
5sequential_3/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_3/strided_slice/stack_2?
-sequential_3/conv1d_transpose_3/strided_sliceStridedSlice.sequential_3/conv1d_transpose_3/Shape:output:0<sequential_3/conv1d_transpose_3/strided_slice/stack:output:0>sequential_3/conv1d_transpose_3/strided_slice/stack_1:output:0>sequential_3/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_3/conv1d_transpose_3/strided_slice?
5sequential_3/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_3/conv1d_transpose_3/strided_slice_1/stack?
7sequential_3/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_3/conv1d_transpose_3/strided_slice_1/stack_1?
7sequential_3/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_3/conv1d_transpose_3/strided_slice_1/stack_2?
/sequential_3/conv1d_transpose_3/strided_slice_1StridedSlice.sequential_3/conv1d_transpose_3/Shape:output:0>sequential_3/conv1d_transpose_3/strided_slice_1/stack:output:0@sequential_3/conv1d_transpose_3/strided_slice_1/stack_1:output:0@sequential_3/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_3/conv1d_transpose_3/strided_slice_1?
%sequential_3/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_3/conv1d_transpose_3/mul/y?
#sequential_3/conv1d_transpose_3/mulMul8sequential_3/conv1d_transpose_3/strided_slice_1:output:0.sequential_3/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_3/conv1d_transpose_3/mul?
'sequential_3/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_3/conv1d_transpose_3/stack/2?
%sequential_3/conv1d_transpose_3/stackPack6sequential_3/conv1d_transpose_3/strided_slice:output:0'sequential_3/conv1d_transpose_3/mul:z:00sequential_3/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/conv1d_transpose_3/stack?
?sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
;sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims'sequential_3/re_lu_4/Relu:activations:0Hsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2=
;sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims?
Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpUsequential_3_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02N
Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
Asequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
=sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsTsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Jsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
=sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
Dsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
>sequential_3/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice.sequential_3/conv1d_transpose_3/stack:output:0Msequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Osequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Osequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2@
>sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice?
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
Hsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
Hsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
@sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice.sequential_3/conv1d_transpose_3/stack:output:0Osequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Qsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Qsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2B
@sequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1?
@sequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1?
<sequential_3/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<sequential_3/conv1d_transpose_3/conv1d_transpose/concat/axis?
7sequential_3/conv1d_transpose_3/conv1d_transpose/concatConcatV2Gsequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Isequential_3/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Isequential_3/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0Esequential_3/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7sequential_3/conv1d_transpose_3/conv1d_transpose/concat?
0sequential_3/conv1d_transpose_3/conv1d_transposeConv2DBackpropInput@sequential_3/conv1d_transpose_3/conv1d_transpose/concat:output:0Fsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0Dsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
22
0sequential_3/conv1d_transpose_3/conv1d_transpose?
8sequential_3/conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze9sequential_3/conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2:
8sequential_3/conv1d_transpose_3/conv1d_transpose/Squeeze?
6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_3_conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp?
'sequential_3/conv1d_transpose_3/BiasAddBiasAddAsequential_3/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0>sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2)
'sequential_3/conv1d_transpose_3/BiasAdd?
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?
3sequential_3/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_3/batch_normalization_11/batchnorm/add/y?
1sequential_3/batch_normalization_11/batchnorm/addAddV2Dsequential_3/batch_normalization_11/batchnorm/ReadVariableOp:value:0<sequential_3/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_11/batchnorm/add?
3sequential_3/batch_normalization_11/batchnorm/RsqrtRsqrt5sequential_3/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_11/batchnorm/Rsqrt?
@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_3_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp?
1sequential_3/batch_normalization_11/batchnorm/mulMul7sequential_3/batch_normalization_11/batchnorm/Rsqrt:y:0Hsequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_11/batchnorm/mul?
3sequential_3/batch_normalization_11/batchnorm/mul_1Mul0sequential_3/conv1d_transpose_3/BiasAdd:output:05sequential_3/batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????25
3sequential_3/batch_normalization_11/batchnorm/mul_1?
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_3_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02@
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?
3sequential_3/batch_normalization_11/batchnorm/mul_2MulFsequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1:value:05sequential_3/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:25
3sequential_3/batch_normalization_11/batchnorm/mul_2?
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_3_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02@
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2?
1sequential_3/batch_normalization_11/batchnorm/subSubFsequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2:value:07sequential_3/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:23
1sequential_3/batch_normalization_11/batchnorm/sub?
3sequential_3/batch_normalization_11/batchnorm/add_1AddV27sequential_3/batch_normalization_11/batchnorm/mul_1:z:05sequential_3/batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????25
3sequential_3/batch_normalization_11/batchnorm/add_1?
sequential_3/re_lu_5/ReluRelu7sequential_3/batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
sequential_3/re_lu_5/Relu?
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_3/flatten_3/Const?
sequential_3/flatten_3/ReshapeReshape'sequential_3/re_lu_5/Relu:activations:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2 
sequential_3/flatten_3/Reshape?
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOp?
sequential_3/dense_7/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_7/MatMul?
sequential_3/dense_7/TanhTanh%sequential_3/dense_7/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_7/Tanh?	
IdentityIdentitysequential_3/dense_7/Tanh:y:0=^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp=^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp<^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1>^sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2@^sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp7^sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp7^sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOpM^sequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2|
<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp<sequential_3/batch_normalization_10/batchnorm/ReadVariableOp2?
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_1>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_12?
>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_2>sequential_3/batch_normalization_10/batchnorm/ReadVariableOp_22?
@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_10/batchnorm/mul/ReadVariableOp2|
<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp<sequential_3/batch_normalization_11/batchnorm/ReadVariableOp2?
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_1>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_12?
>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_2>sequential_3/batch_normalization_11/batchnorm/ReadVariableOp_22?
@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_11/batchnorm/mul/ReadVariableOp2z
;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp;sequential_3/batch_normalization_9/batchnorm/ReadVariableOp2~
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_1=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_12~
=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_2=sequential_3/batch_normalization_9/batchnorm/ReadVariableOp_22?
?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp?sequential_3/batch_normalization_9/batchnorm/mul/ReadVariableOp2p
6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp6sequential_3/conv1d_transpose_2/BiasAdd/ReadVariableOp2?
Lsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_3/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2p
6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp6sequential_3/conv1d_transpose_3/BiasAdd/ReadVariableOp2?
Lsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpLsequential_3/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_6_input
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_152161

inputs*
&dense_6_matmul_readvariableop_resource0
,batch_normalization_9_assignmovingavg_1519842
.batch_normalization_9_assignmovingavg_1_151990?
;batch_normalization_9_batchnorm_mul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resourceL
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_2_biasadd_readvariableop_resource1
-batch_normalization_10_assignmovingavg_1520613
/batch_normalization_10_assignmovingavg_1_152067@
<batch_normalization_10_batchnorm_mul_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resourceL
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_3_biasadd_readvariableop_resource1
-batch_normalization_11_assignmovingavg_1521293
/batch_normalization_11_assignmovingavg_1_152135@
<batch_normalization_11_batchnorm_mul_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource*
&dense_7_matmul_readvariableop_resource
identity??:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_10/AssignMovingAvg/ReadVariableOp?<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_10/batchnorm/ReadVariableOp?3batch_normalization_10/batchnorm/mul/ReadVariableOp?:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_11/AssignMovingAvg/ReadVariableOp?<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_11/batchnorm/ReadVariableOp?3batch_normalization_11/batchnorm/mul/ReadVariableOp?9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_9/AssignMovingAvg/ReadVariableOp?;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_9/batchnorm/ReadVariableOp?2batch_normalization_9/batchnorm/mul/ReadVariableOp?)conv1d_transpose_2/BiasAdd/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_3/BiasAdd/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices?
"batch_normalization_9/moments/meanMeandense_6/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
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
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_6/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
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
+batch_normalization_9/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/151984*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_9/AssignMovingAvg/decay?
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_9_assignmovingavg_151984*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp?
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/151984*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub?
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/151984*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul?
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_9_assignmovingavg_151984-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/151984*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_9/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/151990*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_9/AssignMovingAvg_1/decay?
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_1_151990*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/151990*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub?
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/151990*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul?
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_1_151990/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/151990*
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
%batch_normalization_9/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
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
re_lu_3/ReluRelu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_3/Relul
reshape_3/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshapere_lu_3/Relu:activations:0 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_3/Reshape~
conv1d_transpose_2/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape?
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack?
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1?
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice?
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack?
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1?
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mulz
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/stack/2?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stack?
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsreshape_3/Reshape:output:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_2/conv1d_transpose/ExpandDims?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_slice?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1?
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1?
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concat?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transpose?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/Squeeze?
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_2/BiasAdd/ReadVariableOp?
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_2/BiasAdd?
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_10/moments/mean/reduction_indices?
#batch_normalization_10/moments/meanMean#conv1d_transpose_2/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_10/moments/mean?
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_10/moments/StopGradient?
0batch_normalization_10/moments/SquaredDifferenceSquaredDifference#conv1d_transpose_2/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
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
:*
	keep_dims(2)
'batch_normalization_10/moments/variance?
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze?
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1?
,batch_normalization_10/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/152061*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_10/AssignMovingAvg/decay?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_10_assignmovingavg_152061*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp?
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/152061*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/sub?
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/152061*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul?
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_10_assignmovingavg_152061.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/152061*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_10/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/152067*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_10/AssignMovingAvg_1/decay?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_1_152067*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/152067*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/sub?
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/152067*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul?
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_1_1520670batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/152067*
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
:2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt?
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Mul#conv1d_transpose_2/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/mul_1?
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2?
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp?
$batch_normalization_10/batchnorm/subSub7batch_normalization_10/batchnorm/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/add_1?
re_lu_4/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_4/Relu~
conv1d_transpose_3/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape?
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack?
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1?
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice?
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack?
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1?
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/stack/2?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stack?
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDimsre_lu_4/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_3/conv1d_transpose/ExpandDims?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_slice?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1?
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1?
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concat?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transpose?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze?
)conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_3/BiasAdd/ReadVariableOp?
conv1d_transpose_3/BiasAddBiasAdd4conv1d_transpose_3/conv1d_transpose/Squeeze:output:01conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_3/BiasAdd?
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_11/moments/mean/reduction_indices?
#batch_normalization_11/moments/meanMean#conv1d_transpose_3/BiasAdd:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_11/moments/mean?
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_11/moments/StopGradient?
0batch_normalization_11/moments/SquaredDifferenceSquaredDifference#conv1d_transpose_3/BiasAdd:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????22
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
:*
	keep_dims(2)
'batch_normalization_11/moments/variance?
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze?
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1?
,batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/152129*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_11/AssignMovingAvg/decay?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_11_assignmovingavg_152129*
_output_shapes
:*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp?
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/152129*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/sub?
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/152129*
_output_shapes
:2,
*batch_normalization_11/AssignMovingAvg/mul?
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_11_assignmovingavg_152129.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_11/AssignMovingAvg/152129*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/152135*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_11/AssignMovingAvg_1/decay?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_11_assignmovingavg_1_152135*
_output_shapes
:*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/152135*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/sub?
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/152135*
_output_shapes
:2.
,batch_normalization_11/AssignMovingAvg_1/mul?
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_11_assignmovingavg_1_1521350batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg_1/152135*
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
:2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/Rsqrt?
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOp?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Mul#conv1d_transpose_3/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/mul_1?
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/mul_2?
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_11/batchnorm/ReadVariableOp?
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/add_1?
re_lu_5/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_5/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapere_lu_5/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_3/Reshape?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulflatten_3/Reshape:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMulp
dense_7/TanhTanhdense_7/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_7/Tanh?

IdentityIdentitydense_7/Tanh:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_10/batchnorm/ReadVariableOp4^batch_normalization_10/batchnorm/mul/ReadVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp3^batch_normalization_9/batchnorm/mul/ReadVariableOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_3/BiasAdd/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2x
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
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_3/BiasAdd/ReadVariableOp)conv1d_transpose_3/BiasAdd/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_151639

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :??????????????????2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_152304

inputs*
&dense_6_matmul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resource?
;batch_normalization_9_batchnorm_mul_readvariableop_resource=
9batch_normalization_9_batchnorm_readvariableop_1_resource=
9batch_normalization_9_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_2_biasadd_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource@
<batch_normalization_10_batchnorm_mul_readvariableop_resource>
:batch_normalization_10_batchnorm_readvariableop_1_resource>
:batch_normalization_10_batchnorm_readvariableop_2_resourceL
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource6
2conv1d_transpose_3_biasadd_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource@
<batch_normalization_11_batchnorm_mul_readvariableop_resource>
:batch_normalization_11_batchnorm_readvariableop_1_resource>
:batch_normalization_11_batchnorm_readvariableop_2_resource*
&dense_7_matmul_readvariableop_resource
identity??/batch_normalization_10/batchnorm/ReadVariableOp?1batch_normalization_10/batchnorm/ReadVariableOp_1?1batch_normalization_10/batchnorm/ReadVariableOp_2?3batch_normalization_10/batchnorm/mul/ReadVariableOp?/batch_normalization_11/batchnorm/ReadVariableOp?1batch_normalization_11/batchnorm/ReadVariableOp_1?1batch_normalization_11/batchnorm/ReadVariableOp_2?3batch_normalization_11/batchnorm/mul/ReadVariableOp?.batch_normalization_9/batchnorm/ReadVariableOp?0batch_normalization_9/batchnorm/ReadVariableOp_1?0batch_normalization_9/batchnorm/ReadVariableOp_2?2batch_normalization_9/batchnorm/mul/ReadVariableOp?)conv1d_transpose_2/BiasAdd/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_3/BiasAdd/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
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
%batch_normalization_9/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
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
re_lu_3/ReluRelu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
re_lu_3/Relul
reshape_3/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshapere_lu_3/Relu:activations:0 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_3/Reshape~
conv1d_transpose_2/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape?
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack?
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1?
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice?
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack?
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1?
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mulz
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/stack/2?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stack?
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsreshape_3/Reshape:output:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_2/conv1d_transpose/ExpandDims?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_slice?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1?
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1?
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concat?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transpose?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/Squeeze?
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_2/BiasAdd/ReadVariableOp?
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_2/BiasAdd?
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt?
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Mul#conv1d_transpose_2/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/mul_1?
1batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_1?
&batch_normalization_10/batchnorm/mul_2Mul9batch_normalization_10/batchnorm/ReadVariableOp_1:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2?
1batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_2?
$batch_normalization_10/batchnorm/subSub9batch_normalization_10/batchnorm/ReadVariableOp_2:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_10/batchnorm/add_1?
re_lu_4/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_4/Relu~
conv1d_transpose_3/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape?
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack?
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1?
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice?
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack?
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1?
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/stack/2?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stack?
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDimsre_lu_4/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????20
.conv1d_transpose_3/conv1d_transpose/ExpandDims?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stack?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_slice?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1?
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1?
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concat?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transpose?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze?
)conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_3/BiasAdd/ReadVariableOp?
conv1d_transpose_3/BiasAddBiasAdd4conv1d_transpose_3/conv1d_transpose/Squeeze:output:01conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
conv1d_transpose_3/BiasAdd?
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/Rsqrt?
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOp?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Mul#conv1d_transpose_3/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/mul_1?
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_1?
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_11/batchnorm/mul_2?
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_2?
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????2(
&batch_normalization_11/batchnorm/add_1?
re_lu_5/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????2
re_lu_5/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_3/Const?
flatten_3/ReshapeReshapere_lu_5/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_3/Reshape?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulflatten_3/Reshape:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMulp
dense_7/TanhTanhdense_7/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_7/Tanh?
IdentityIdentitydense_7/Tanh:y:00^batch_normalization_10/batchnorm/ReadVariableOp2^batch_normalization_10/batchnorm/ReadVariableOp_12^batch_normalization_10/batchnorm/ReadVariableOp_24^batch_normalization_10/batchnorm/mul/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp1^batch_normalization_9/batchnorm/ReadVariableOp_11^batch_normalization_9/batchnorm/ReadVariableOp_23^batch_normalization_9/batchnorm/mul/ReadVariableOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_3/BiasAdd/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::::2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2f
1batch_normalization_10/batchnorm/ReadVariableOp_11batch_normalization_10/batchnorm/ReadVariableOp_12f
1batch_normalization_10/batchnorm/ReadVariableOp_21batch_normalization_10/batchnorm/ReadVariableOp_22j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2d
0batch_normalization_9/batchnorm/ReadVariableOp_10batch_normalization_9/batchnorm/ReadVariableOp_12d
0batch_normalization_9/batchnorm/ReadVariableOp_20batch_normalization_9/batchnorm/ReadVariableOp_22h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_3/BiasAdd/ReadVariableOp)conv1d_transpose_3/BiasAdd/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_152393

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
dense_6_input6
serving_default_dense_6_input:0?????????;
dense_70
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?V
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?R
_tf_keras_sequential?R{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?

kernel
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?	
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4]}}}
?


(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4]}}
?	
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?
7	variables
8regularization_losses
9trainable_variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?


;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 12]}}
?	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 6]}}
?
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Rkernel
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
?
0
1
2
3
4
(5
)6
/7
08
19
210
;11
<12
B13
C14
D15
E16
R17"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
(3
)4
/5
06
;7
<8
B9
C10
R11"
trackable_list_wrapper
?
Wnon_trainable_variables
Xlayer_metrics
Ymetrics
	variables
Zlayer_regularization_losses
regularization_losses
trainable_variables

[layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 :2dense_6/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
\non_trainable_variables
]layer_metrics
^metrics
	variables
_layer_regularization_losses
regularization_losses
trainable_variables

`layers
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
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics
cmetrics
	variables
dlayer_regularization_losses
regularization_losses
trainable_variables

elayers
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
fnon_trainable_variables
glayer_metrics
hmetrics
 	variables
ilayer_regularization_losses
!regularization_losses
"trainable_variables

jlayers
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
knon_trainable_variables
llayer_metrics
mmetrics
$	variables
nlayer_regularization_losses
%regularization_losses
&trainable_variables

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_2/kernel
%:#2conv1d_transpose_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics
rmetrics
*	variables
slayer_regularization_losses
+regularization_losses
,trainable_variables

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_10/gamma
):'2batch_normalization_10/beta
2:0 (2"batch_normalization_10/moving_mean
6:4 (2&batch_normalization_10/moving_variance
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics
wmetrics
3	variables
xlayer_regularization_losses
4regularization_losses
5trainable_variables

ylayers
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
znon_trainable_variables
{layer_metrics
|metrics
7	variables
}layer_regularization_losses
8regularization_losses
9trainable_variables

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_3/kernel
%:#2conv1d_transpose_3/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
non_trainable_variables
?layer_metrics
?metrics
=	variables
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?metrics
F	variables
 ?layer_regularization_losses
Gregularization_losses
Htrainable_variables
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
?non_trainable_variables
?layer_metrics
?metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
Ltrainable_variables
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
?non_trainable_variables
?layer_metrics
?metrics
N	variables
 ?layer_regularization_losses
Oregularization_losses
Ptrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_7/kernel
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?metrics
S	variables
 ?layer_regularization_losses
Tregularization_losses
Utrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
0
1
12
23
D4
E5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
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
11"
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
0
1"
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
10
21"
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
D0
E1"
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
?2?
-__inference_sequential_3_layer_call_fn_151927
-__inference_sequential_3_layer_call_fn_151834
-__inference_sequential_3_layer_call_fn_152386
-__inference_sequential_3_layer_call_fn_152345?
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
!__inference__wrapped_model_150929?
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
annotations? *,?)
'?$
dense_6_input?????????
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_152304
H__inference_sequential_3_layer_call_and_return_conditional_losses_151740
H__inference_sequential_3_layer_call_and_return_conditional_losses_152161
H__inference_sequential_3_layer_call_and_return_conditional_losses_151688?
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
(__inference_dense_6_layer_call_fn_152400?
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
C__inference_dense_6_layer_call_and_return_conditional_losses_152393?
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
6__inference_batch_normalization_9_layer_call_fn_152482
6__inference_batch_normalization_9_layer_call_fn_152469?
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
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152436
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152456?
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
(__inference_re_lu_3_layer_call_fn_152492?
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
C__inference_re_lu_3_layer_call_and_return_conditional_losses_152487?
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
*__inference_reshape_3_layer_call_fn_152510?
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
E__inference_reshape_3_layer_call_and_return_conditional_losses_152505?
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
3__inference_conv1d_transpose_2_layer_call_fn_151119?
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
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_151109?
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
7__inference_batch_normalization_10_layer_call_fn_152579
7__inference_batch_normalization_10_layer_call_fn_152592?
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
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152546
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152566?
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
(__inference_re_lu_4_layer_call_fn_152602?
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
C__inference_re_lu_4_layer_call_and_return_conditional_losses_152597?
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
3__inference_conv1d_transpose_3_layer_call_fn_151309?
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
%?"??????????????????
?2?
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_151299?
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
%?"??????????????????
?2?
7__inference_batch_normalization_11_layer_call_fn_152671
7__inference_batch_normalization_11_layer_call_fn_152684?
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
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152638
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152658?
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
(__inference_re_lu_5_layer_call_fn_152694?
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
C__inference_re_lu_5_layer_call_and_return_conditional_losses_152689?
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
*__inference_flatten_3_layer_call_fn_152711?
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_152706?
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
(__inference_dense_7_layer_call_fn_152726?
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
C__inference_dense_7_layer_call_and_return_conditional_losses_152719?
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
$__inference_signature_wrapper_151970dense_6_input"?
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
!__inference__wrapped_model_150929()2/10;<EBDCR6?3
,?)
'?$
dense_6_input?????????
? "1?.
,
dense_7!?
dense_7??????????
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152546|12/0@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_152566|2/10@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
7__inference_batch_normalization_10_layer_call_fn_152579o12/0@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
7__inference_batch_normalization_10_layer_call_fn_152592o2/10@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152638|DEBC@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_152658|EBDC@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
7__inference_batch_normalization_11_layer_call_fn_152671oDEBC@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
7__inference_batch_normalization_11_layer_call_fn_152684oEBDC@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152436b3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_152456b3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
6__inference_batch_normalization_9_layer_call_fn_152469U3?0
)?&
 ?
inputs?????????
p
? "???????????
6__inference_batch_normalization_9_layer_call_fn_152482U3?0
)?&
 ?
inputs?????????
p 
? "???????????
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_151109v()<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
3__inference_conv1d_transpose_2_layer_call_fn_151119i()<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_151299v;<<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
3__inference_conv1d_transpose_3_layer_call_fn_151309i;<<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
C__inference_dense_6_layer_call_and_return_conditional_losses_152393[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
(__inference_dense_6_layer_call_fn_152400N/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_7_layer_call_and_return_conditional_losses_152719dR8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????
? ?
(__inference_dense_7_layer_call_fn_152726WR8?5
.?+
)?&
inputs??????????????????
? "???????????
E__inference_flatten_3_layer_call_and_return_conditional_losses_152706n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
*__inference_flatten_3_layer_call_fn_152711a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
C__inference_re_lu_3_layer_call_and_return_conditional_losses_152487X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? w
(__inference_re_lu_3_layer_call_fn_152492K/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_re_lu_4_layer_call_and_return_conditional_losses_152597r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
(__inference_re_lu_4_layer_call_fn_152602e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
C__inference_re_lu_5_layer_call_and_return_conditional_losses_152689r<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
(__inference_re_lu_5_layer_call_fn_152694e<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
E__inference_reshape_3_layer_call_and_return_conditional_losses_152505\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? }
*__inference_reshape_3_layer_call_fn_152510O/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_sequential_3_layer_call_and_return_conditional_losses_151688{()12/0;<DEBCR>?;
4?1
'?$
dense_6_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_151740{()2/10;<EBDCR>?;
4?1
'?$
dense_6_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_152161t()12/0;<DEBCR7?4
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_152304t()2/10;<EBDCR7?4
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
-__inference_sequential_3_layer_call_fn_151834n()12/0;<DEBCR>?;
4?1
'?$
dense_6_input?????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_151927n()2/10;<EBDCR>?;
4?1
'?$
dense_6_input?????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_152345g()12/0;<DEBCR7?4
-?*
 ?
inputs?????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_152386g()2/10;<EBDCR7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
$__inference_signature_wrapper_151970?()2/10;<EBDCRG?D
? 
=?:
8
dense_6_input'?$
dense_6_input?????????"1?.
,
dense_7!?
dense_7?????????