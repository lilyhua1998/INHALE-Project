??
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
?
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
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:
*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:
*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:
*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%batch_normalization_4/moving_variance
?
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
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:
*
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:
*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:
*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%batch_normalization_5/moving_variance
?
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
* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:
*
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?(B?( B?(
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
^

kernel
trainable_variables
regularization_losses
	variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
?
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
?
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6regularization_losses
7	variables
8	keras_api
^

9kernel
:trainable_variables
;regularization_losses
<	variables
=	keras_api
V
0
1
2
3
4
"5
#6
*7
+8
19
210
911
 
?
0
1
2
3
4
5
6
"7
#8
$9
%10
*11
+12
113
214
315
416
917
?
trainable_variables
>metrics
	regularization_losses
?layer_metrics
@layer_regularization_losses

	variables
Anon_trainable_variables

Blayers
 
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
trainable_variables
Cmetrics
regularization_losses
Dlayer_metrics
Elayer_regularization_losses
	variables
Fnon_trainable_variables

Glayers
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
?
trainable_variables
Hmetrics
regularization_losses
Ilayer_metrics
Jlayer_regularization_losses
	variables
Knon_trainable_variables

Llayers
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
Mmetrics
regularization_losses
Nlayer_metrics
Olayer_regularization_losses
	variables
Pnon_trainable_variables

Qlayers
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
$2
%3
?
&trainable_variables
Rmetrics
'regularization_losses
Slayer_metrics
Tlayer_regularization_losses
(	variables
Unon_trainable_variables

Vlayers
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
?
,trainable_variables
Wmetrics
-regularization_losses
Xlayer_metrics
Ylayer_regularization_losses
.	variables
Znon_trainable_variables

[layers
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
32
43
?
5trainable_variables
\metrics
6regularization_losses
]layer_metrics
^layer_regularization_losses
7	variables
_non_trainable_variables

`layers
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

90
 

90
?
:trainable_variables
ametrics
;regularization_losses
blayer_metrics
clayer_regularization_losses
<	variables
dnon_trainable_variables

elayers
 
 
 
*
0
1
$2
%3
34
45
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 

0
1
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
$0
%1
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
30
41
 
 
 
 
 
 
?
serving_default_dense_8_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_8_inputdense_8/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_9/kerneldense_9/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_10/kerneldense_10/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_11/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_25010173
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOpConst*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_25010844
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_9/kerneldense_9/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_10/kerneldense_10/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_11/kernel*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_25010908??

??
?
#__inference__wrapped_model_25009282
dense_8_input7
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
4sequential_2_dense_11_matmul_readvariableop_resource
identity??;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp?;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp?;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp?=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1?=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2??sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp?,sequential_2/dense_10/BiasAdd/ReadVariableOp?+sequential_2/dense_10/MatMul/ReadVariableOp?+sequential_2/dense_11/MatMul/ReadVariableOp?*sequential_2/dense_8/MatMul/ReadVariableOp?+sequential_2/dense_9/BiasAdd/ReadVariableOp?*sequential_2/dense_9/MatMul/ReadVariableOp?
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp?
sequential_2/dense_8/MatMulMatMuldense_8_input2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_8/MatMul?
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_8/Relu?
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_3/batchnorm/add/y?
0sequential_2/batch_normalization_3/batchnorm/addAddV2Csequential_2/batch_normalization_3/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/add?
2sequential_2/batch_normalization_3/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/Rsqrt?
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_3/batchnorm/mulMul6sequential_2/batch_normalization_3/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/mul?
2sequential_2/batch_normalization_3/batchnorm/mul_1Mul'sequential_2/dense_8/Relu:activations:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_3/batchnorm/mul_1?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_3/batchnorm/mul_2MulEsequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:24
2sequential_2/batch_normalization_3/batchnorm/mul_2?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02?
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_3/batchnorm/subSubEsequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
0sequential_2/batch_normalization_3/batchnorm/sub?
2sequential_2/batch_normalization_3/batchnorm/add_1AddV26sequential_2/batch_normalization_3/batchnorm/mul_1:z:04sequential_2/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????24
2sequential_2/batch_normalization_3/batchnorm/add_1?
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOp?
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_9/MatMul?
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOp?
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_9/BiasAdd?
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_9/Relu?
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_4/batchnorm/add/y?
0sequential_2/batch_normalization_4/batchnorm/addAddV2Csequential_2/batch_normalization_4/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/add?
2sequential_2/batch_normalization_4/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/Rsqrt?
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_4/batchnorm/mulMul6sequential_2/batch_normalization_4/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/mul?
2sequential_2/batch_normalization_4/batchnorm/mul_1Mul'sequential_2/dense_9/Relu:activations:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
24
2sequential_2/batch_normalization_4/batchnorm/mul_1?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_4/batchnorm/mul_2MulEsequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_4/batchnorm/mul_2?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_4/batchnorm/subSubEsequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_4/batchnorm/sub?
2sequential_2/batch_normalization_4/batchnorm/add_1AddV26sequential_2/batch_normalization_4/batchnorm/mul_1:z:04sequential_2/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
24
2sequential_2/batch_normalization_4/batchnorm/add_1?
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02-
+sequential_2/dense_10/MatMul/ReadVariableOp?
sequential_2/dense_10/MatMulMatMul6sequential_2/batch_normalization_4/batchnorm/add_1:z:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_10/MatMul?
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOp?
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_10/BiasAdd?
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_2/dense_10/Relu?
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpDsequential_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02=
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp?
2sequential_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_2/batch_normalization_5/batchnorm/add/y?
0sequential_2/batch_normalization_5/batchnorm/addAddV2Csequential_2/batch_normalization_5/batchnorm/ReadVariableOp:value:0;sequential_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/add?
2sequential_2/batch_normalization_5/batchnorm/RsqrtRsqrt4sequential_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/Rsqrt?
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02A
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp?
0sequential_2/batch_normalization_5/batchnorm/mulMul6sequential_2/batch_normalization_5/batchnorm/Rsqrt:y:0Gsequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/mul?
2sequential_2/batch_normalization_5/batchnorm/mul_1Mul(sequential_2/dense_10/Relu:activations:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
24
2sequential_2/batch_normalization_5/batchnorm/mul_1?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_2_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1?
2sequential_2/batch_normalization_5/batchnorm/mul_2MulEsequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1:value:04sequential_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
24
2sequential_2/batch_normalization_5/batchnorm/mul_2?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02?
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2?
0sequential_2/batch_normalization_5/batchnorm/subSubEsequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:06sequential_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
22
0sequential_2/batch_normalization_5/batchnorm/sub?
2sequential_2/batch_normalization_5/batchnorm/add_1AddV26sequential_2/batch_normalization_5/batchnorm/mul_1:z:04sequential_2/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
24
2sequential_2/batch_normalization_5/batchnorm/add_1?
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOp?
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_11/MatMul?
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_11/Tanh?	
IdentityIdentitysequential_2/dense_11/Tanh:y:0<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2z
;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2z
;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp2~
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_12~
=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_22?
?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
?	
?
F__inference_dense_10_layer_call_and_return_conditional_losses_25009834

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010502

inputs
assignmovingavg_25010477
assignmovingavg_1_25010483)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010477*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25010477*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010477*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010477*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25010477AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010477*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010483*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25010483*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010483*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010483*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25010483AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010483*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009953
dense_8_input
dense_8_25009909"
batch_normalization_3_25009912"
batch_normalization_3_25009914"
batch_normalization_3_25009916"
batch_normalization_3_25009918
dense_9_25009921
dense_9_25009923"
batch_normalization_4_25009926"
batch_normalization_4_25009928"
batch_normalization_4_25009930"
batch_normalization_4_25009932
dense_10_25009935
dense_10_25009937"
batch_normalization_5_25009940"
batch_normalization_5_25009942"
batch_normalization_5_25009944"
batch_normalization_5_25009946
dense_11_25009949
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_25009909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_250097142!
dense_8/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_25009912batch_normalization_3_25009914batch_normalization_3_25009916batch_normalization_3_25009918*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250094112/
-batch_normalization_3/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_25009921dense_9_25009923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_250097722!
dense_9/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_25009926batch_normalization_4_25009928batch_normalization_4_25009930batch_normalization_4_25009932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095512/
-batch_normalization_4/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_25009935dense_10_25009937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_250098342"
 dense_10/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_25009940batch_normalization_5_25009942batch_normalization_5_25009944batch_normalization_5_25009946*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096912/
-batch_normalization_5/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_25009949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_250098932"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
?
?
+__inference_dense_10_layer_call_fn_25010670

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
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_250098342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

*__inference_dense_9_layer_call_fn_25010568

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
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_250097722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25009658

inputs
assignmovingavg_25009633
assignmovingavg_1_25009639)
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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
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

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009633*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25009633*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009633*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009633*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25009633AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009633*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009639*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25009639*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009639*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009639*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25009639AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009639*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
/__inference_sequential_2_layer_call_fn_25010451

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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_250100912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25009518

inputs
assignmovingavg_25009493
assignmovingavg_1_25009499)
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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
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

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009493*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25009493*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009493*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009493*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25009493AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009493*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009499*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25009499*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009499*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009499*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25009499AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009499*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?,
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009906
dense_8_input
dense_8_25009723"
batch_normalization_3_25009752"
batch_normalization_3_25009754"
batch_normalization_3_25009756"
batch_normalization_3_25009758
dense_9_25009783
dense_9_25009785"
batch_normalization_4_25009814"
batch_normalization_4_25009816"
batch_normalization_4_25009818"
batch_normalization_4_25009820
dense_10_25009845
dense_10_25009847"
batch_normalization_5_25009876"
batch_normalization_5_25009878"
batch_normalization_5_25009880"
batch_normalization_5_25009882
dense_11_25009902
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_25009723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_250097142!
dense_8/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_25009752batch_normalization_3_25009754batch_normalization_3_25009756batch_normalization_3_25009758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250093782/
-batch_normalization_3/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_25009783dense_9_25009785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_250097722!
dense_9/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_25009814batch_normalization_4_25009816batch_normalization_4_25009818batch_normalization_4_25009820*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095182/
-batch_normalization_4/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_25009845dense_10_25009847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_250098342"
 dense_10/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_25009876batch_normalization_5_25009878batch_normalization_5_25009880batch_normalization_5_25009882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096582/
-batch_normalization_5/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_25009902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_250098932"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
??
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010295

inputs*
&dense_8_matmul_readvariableop_resource2
.batch_normalization_3_assignmovingavg_250101884
0batch_normalization_3_assignmovingavg_1_25010194?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.batch_normalization_4_assignmovingavg_250102274
0batch_normalization_4_assignmovingavg_1_25010233?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource2
.batch_normalization_5_assignmovingavg_250102664
0batch_normalization_5_assignmovingavg_1_25010272?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource
identity??9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_3/AssignMovingAvg/ReadVariableOp?;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_3/batchnorm/ReadVariableOp?2batch_normalization_3/batchnorm/mul/ReadVariableOp?9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_4/AssignMovingAvg/ReadVariableOp?;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_4/batchnorm/ReadVariableOp?2batch_normalization_4/batchnorm/mul/ReadVariableOp?9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp?4batch_normalization_5/AssignMovingAvg/ReadVariableOp?;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp?6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_5/batchnorm/ReadVariableOp?2batch_normalization_5/batchnorm/mul/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMulp
dense_8/ReluReludense_8/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_8/Relu?
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices?
"batch_normalization_3/moments/meanMeandense_8/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_3/moments/mean?
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_3/moments/StopGradient?
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????21
/batch_normalization_3/moments/SquaredDifference?
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices?
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_3/moments/variance?
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze?
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1?
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/25010188*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_3/AssignMovingAvg/decay?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_25010188*
_output_shapes
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp?
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/25010188*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/sub?
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/25010188*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/mul?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_25010188-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/25010188*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/25010194*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_3/AssignMovingAvg_1/decay?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_3_assignmovingavg_1_25010194*
_output_shapes
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/25010194*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/sub?
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/25010194*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/mul?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_3_assignmovingavg_1_25010194/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/25010194*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_3/batchnorm/add/y?
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add?
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrt?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mul?
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/mul_1?
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp?
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/sub?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/add_1?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_9/Relu?
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indices?
"batch_normalization_4/moments/meanMeandense_9/Relu:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_4/moments/mean?
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_4/moments/StopGradient?
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
21
/batch_normalization_4/moments/SquaredDifference?
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices?
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_4/moments/variance?
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze?
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1?
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/25010227*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_4/AssignMovingAvg/decay?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_25010227*
_output_shapes
:
*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp?
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/25010227*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/sub?
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/25010227*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/mul?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_25010227-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/25010227*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/25010233*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_4/AssignMovingAvg_1/decay?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_4_assignmovingavg_1_25010233*
_output_shapes
:
*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/25010233*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/sub?
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/25010233*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/mul?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_4_assignmovingavg_1_25010233/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/25010233*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_4/batchnorm/add/y?
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add?
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrt?
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp?
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mul?
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_4/batchnorm/mul_1?
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2?
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp?
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/sub?
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_4/batchnorm/add_1?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_10/Relu?
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indices?
"batch_normalization_5/moments/meanMeandense_10/Relu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_5/moments/mean?
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_5/moments/StopGradient?
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
21
/batch_normalization_5/moments/SquaredDifference?
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indices?
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_5/moments/variance?
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze?
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1?
+batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/25010266*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_5/AssignMovingAvg/decay?
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_25010266*
_output_shapes
:
*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp?
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/25010266*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/sub?
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/25010266*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/mul?
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_25010266-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/25010266*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp?
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/25010272*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_5/AssignMovingAvg_1/decay?
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_5_assignmovingavg_1_25010272*
_output_shapes
:
*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/25010272*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/sub?
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/25010272*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/mul?
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_5_assignmovingavg_1_25010272/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/25010272*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp?
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_5/batchnorm/add/y?
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add?
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrt?
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp?
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mul?
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_5/batchnorm/mul_1?
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2?
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp?
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/sub?
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_5/batchnorm/add_1?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanh?

IdentityIdentitydense_11/Tanh:y:0:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2v
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
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25009411

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?s
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010369

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
identity??.batch_normalization_3/batchnorm/ReadVariableOp?0batch_normalization_3/batchnorm/ReadVariableOp_1?0batch_normalization_3/batchnorm/ReadVariableOp_2?2batch_normalization_3/batchnorm/mul/ReadVariableOp?.batch_normalization_4/batchnorm/ReadVariableOp?0batch_normalization_4/batchnorm/ReadVariableOp_1?0batch_normalization_4/batchnorm/ReadVariableOp_2?2batch_normalization_4/batchnorm/mul/ReadVariableOp?.batch_normalization_5/batchnorm/ReadVariableOp?0batch_normalization_5/batchnorm/ReadVariableOp_1?0batch_normalization_5/batchnorm/ReadVariableOp_2?2batch_normalization_5/batchnorm/mul/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMulp
dense_8/ReluReludense_8/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_8/Relu?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp?
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_3/batchnorm/add/y?
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add?
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrt?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mul?
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/mul_1?
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1?
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2?
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2?
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/sub?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/add_1?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_9/Relu?
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp?
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_4/batchnorm/add/y?
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add?
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrt?
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp?
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mul?
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_4/batchnorm/mul_1?
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1?
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2?
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2?
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/sub?
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_4/batchnorm/add_1?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_10/Relu?
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp?
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_5/batchnorm/add/y?
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add?
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrt?
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp?
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mul?
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_5/batchnorm/mul_1?
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1?
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2?
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2?
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/sub?
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2'
%batch_normalization_5/batchnorm/add_1?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_11/MatMuls
dense_11/TanhTanhdense_11/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_11/Tanh?
IdentityIdentitydense_11/Tanh:y:0/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2`
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
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010522

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25009551

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
p
*__inference_dense_8_layer_call_fn_25010466

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_250097142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_25010637

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
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_25010650

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
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010726

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_25010459

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:?????????2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010604

inputs
assignmovingavg_25010579
assignmovingavg_1_25010585)
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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
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

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010579*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25010579*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010579*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010579*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25010579AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010579*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010585*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25010585*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010585*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010585*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25010585AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010585*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_25010535

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250093782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_25010042
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
identity??StatefulPartitionedCall?
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
:?????????*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_250100032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
?
?
F__inference_dense_11_layer_call_and_return_conditional_losses_25010760

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_25010173
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
identity??StatefulPartitionedCall?
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_250092822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
?	
?
F__inference_dense_10_layer_call_and_return_conditional_losses_25010661

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?,
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010003

inputs
dense_8_25009959"
batch_normalization_3_25009962"
batch_normalization_3_25009964"
batch_normalization_3_25009966"
batch_normalization_3_25009968
dense_9_25009971
dense_9_25009973"
batch_normalization_4_25009976"
batch_normalization_4_25009978"
batch_normalization_4_25009980"
batch_normalization_4_25009982
dense_10_25009985
dense_10_25009987"
batch_normalization_5_25009990"
batch_normalization_5_25009992"
batch_normalization_5_25009994"
batch_normalization_5_25009996
dense_11_25009999
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_25009959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_250097142!
dense_8/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_25009962batch_normalization_3_25009964batch_normalization_3_25009966batch_normalization_3_25009968*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250093782/
-batch_normalization_3/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_25009971dense_9_25009973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_250097722!
dense_9/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_25009976batch_normalization_4_25009978batch_normalization_4_25009980batch_normalization_4_25009982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095182/
-batch_normalization_4/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_25009985dense_10_25009987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_250098342"
 dense_10/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_25009990batch_normalization_5_25009992batch_normalization_5_25009994batch_normalization_5_25009996*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096582/
-batch_normalization_5/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_25009999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_250098932"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25009691

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010706

inputs
assignmovingavg_25010681
assignmovingavg_1_25010687)
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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
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

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010681*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25010681*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010681*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25010681*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25010681AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25010681*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010687*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25010687*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010687*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25010687*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25010687AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25010687*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
q
+__inference_dense_11_layer_call_fn_25010767

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
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_250098932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?0
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25009378

inputs
assignmovingavg_25009353
assignmovingavg_1_25009359)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009353*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_25009353*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009353*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/25009353*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_25009353AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/25009353*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009359*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_25009359*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009359*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/25009359*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_25009359AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/25009359*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_11_layer_call_and_return_conditional_losses_25009893

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:?????????2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?,
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010091

inputs
dense_8_25010047"
batch_normalization_3_25010050"
batch_normalization_3_25010052"
batch_normalization_3_25010054"
batch_normalization_3_25010056
dense_9_25010059
dense_9_25010061"
batch_normalization_4_25010064"
batch_normalization_4_25010066"
batch_normalization_4_25010068"
batch_normalization_4_25010070
dense_10_25010073
dense_10_25010075"
batch_normalization_5_25010078"
batch_normalization_5_25010080"
batch_normalization_5_25010082"
batch_normalization_5_25010084
dense_11_25010087
identity??-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_25010047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_250097142!
dense_8/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_25010050batch_normalization_3_25010052batch_normalization_3_25010054batch_normalization_3_25010056*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250094112/
-batch_normalization_3/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_25010059dense_9_25010061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_250097722!
dense_9/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_25010064batch_normalization_4_25010066batch_normalization_4_25010068batch_normalization_4_25010070*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_250095512/
-batch_normalization_4/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_25010073dense_10_25010075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_250098342"
 dense_10/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_25010078batch_normalization_5_25010080batch_normalization_5_25010082batch_normalization_5_25010084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096912/
-batch_normalization_5/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_25010087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_250098932"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_25010739

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
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_25009714

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:?????????2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_25010130
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
identity??StatefulPartitionedCall?
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
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_250100912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_8_input
?0
?	
!__inference__traced_save_25010844
file_prefix-
)savev2_dense_8_kernel_read_readvariableop:
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :	?:::::
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
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 
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
::$ 

_output_shapes

:
: 

_output_shapes
:
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
:$ 

_output_shapes

:

: 

_output_shapes
:
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
:$ 

_output_shapes

:
:

_output_shapes
: 
?	
?
E__inference_dense_9_layer_call_and_return_conditional_losses_25009772

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_25010548

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250094112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010624

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
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
 *o?:2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
/__inference_sequential_2_layer_call_fn_25010410

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
:?????????*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_250100032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_9_layer_call_and_return_conditional_losses_25010559

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_25010752

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
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_250096912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?P
?

$__inference__traced_restore_25010908
file_prefix#
assignvariableop_dense_8_kernel2
.assignvariableop_1_batch_normalization_3_gamma1
-assignvariableop_2_batch_normalization_3_beta8
4assignvariableop_3_batch_normalization_3_moving_mean<
8assignvariableop_4_batch_normalization_3_moving_variance%
!assignvariableop_5_dense_9_kernel#
assignvariableop_6_dense_9_bias2
.assignvariableop_7_batch_normalization_4_gamma1
-assignvariableop_8_batch_normalization_4_beta8
4assignvariableop_9_batch_normalization_4_moving_mean=
9assignvariableop_10_batch_normalization_4_moving_variance'
#assignvariableop_11_dense_10_kernel%
!assignvariableop_12_dense_10_bias3
/assignvariableop_13_batch_normalization_5_gamma2
.assignvariableop_14_batch_normalization_5_beta9
5assignvariableop_15_batch_normalization_5_moving_mean=
9assignvariableop_16_batch_normalization_5_moving_variance'
#assignvariableop_17_dense_11_kernel
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
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_3_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batch_normalization_3_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp4assignvariableop_3_batch_normalization_3_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_normalization_3_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_9_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_9_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_4_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_4_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_4_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_4_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_10_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_10_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_5_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_5_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_5_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_5_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_11_kernelIdentity_17:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
dense_8_input7
serving_default_dense_8_input:0??????????<
dense_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?A
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
f__call__
*g&call_and_return_all_conditional_losses
h_default_save_signature"?>
_tf_keras_sequential?={"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?

kernel
trainable_variables
regularization_losses
	variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
?	
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&trainable_variables
'regularization_losses
(	variables
)	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?	
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6regularization_losses
7	variables
8	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

9kernel
:trainable_variables
;regularization_losses
<	variables
=	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 3, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
v
0
1
2
3
4
"5
#6
*7
+8
19
210
911"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
"7
#8
$9
%10
*11
+12
113
214
315
416
917"
trackable_list_wrapper
?
trainable_variables
>metrics
	regularization_losses
?layer_metrics
@layer_regularization_losses

	variables
Anon_trainable_variables

Blayers
f__call__
h_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
wserving_default"
signature_map
!:	?2dense_8/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
trainable_variables
Cmetrics
regularization_losses
Dlayer_metrics
Elayer_regularization_losses
	variables
Fnon_trainable_variables

Glayers
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
trainable_variables
Hmetrics
regularization_losses
Ilayer_metrics
Jlayer_regularization_losses
	variables
Knon_trainable_variables

Llayers
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_9/kernel
:
2dense_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
Mmetrics
regularization_losses
Nlayer_metrics
Olayer_regularization_losses
	variables
Pnon_trainable_variables

Qlayers
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'
2batch_normalization_4/gamma
(:&
2batch_normalization_4/beta
1:/
 (2!batch_normalization_4/moving_mean
5:3
 (2%batch_normalization_4/moving_variance
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
?
&trainable_variables
Rmetrics
'regularization_losses
Slayer_metrics
Tlayer_regularization_losses
(	variables
Unon_trainable_variables

Vlayers
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_10/kernel
:
2dense_10/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
,trainable_variables
Wmetrics
-regularization_losses
Xlayer_metrics
Ylayer_regularization_losses
.	variables
Znon_trainable_variables

[layers
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'
2batch_normalization_5/gamma
(:&
2batch_normalization_5/beta
1:/
 (2!batch_normalization_5/moving_mean
5:3
 (2%batch_normalization_5/moving_variance
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
?
5trainable_variables
\metrics
6regularization_losses
]layer_metrics
^layer_regularization_losses
7	variables
_non_trainable_variables

`layers
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_11/kernel
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
?
:trainable_variables
ametrics
;regularization_losses
blayer_metrics
clayer_regularization_losses
<	variables
dnon_trainable_variables

elayers
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
$2
%3
34
45"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
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
0
1"
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
$0
%1"
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
30
41"
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
?2?
/__inference_sequential_2_layer_call_fn_25010410
/__inference_sequential_2_layer_call_fn_25010042
/__inference_sequential_2_layer_call_fn_25010451
/__inference_sequential_2_layer_call_fn_25010130?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010295
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009906
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009953
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010369?
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
#__inference__wrapped_model_25009282?
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
annotations? *-?*
(?%
dense_8_input??????????
?2?
*__inference_dense_8_layer_call_fn_25010466?
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
E__inference_dense_8_layer_call_and_return_conditional_losses_25010459?
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
8__inference_batch_normalization_3_layer_call_fn_25010535
8__inference_batch_normalization_3_layer_call_fn_25010548?
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
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010522
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010502?
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
*__inference_dense_9_layer_call_fn_25010568?
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
E__inference_dense_9_layer_call_and_return_conditional_losses_25010559?
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
8__inference_batch_normalization_4_layer_call_fn_25010637
8__inference_batch_normalization_4_layer_call_fn_25010650?
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
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010624
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010604?
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
+__inference_dense_10_layer_call_fn_25010670?
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
F__inference_dense_10_layer_call_and_return_conditional_losses_25010661?
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
8__inference_batch_normalization_5_layer_call_fn_25010739
8__inference_batch_normalization_5_layer_call_fn_25010752?
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
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010706
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010726?
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
+__inference_dense_11_layer_call_fn_25010767?
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
F__inference_dense_11_layer_call_and_return_conditional_losses_25010760?
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
&__inference_signature_wrapper_25010173dense_8_input"?
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
#__inference__wrapped_model_25009282?%"$#*+413297?4
-?*
(?%
dense_8_input??????????
? "3?0
.
dense_11"?
dense_11??????????
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010502b3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25010522b3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_3_layer_call_fn_25010535U3?0
)?&
 ?
inputs?????????
p
? "???????????
8__inference_batch_normalization_3_layer_call_fn_25010548U3?0
)?&
 ?
inputs?????????
p 
? "???????????
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010604b$%"#3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25010624b%"$#3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
8__inference_batch_normalization_4_layer_call_fn_25010637U$%"#3?0
)?&
 ?
inputs?????????

p
? "??????????
?
8__inference_batch_normalization_4_layer_call_fn_25010650U%"$#3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010706b34123?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25010726b41323?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
8__inference_batch_normalization_5_layer_call_fn_25010739U34123?0
)?&
 ?
inputs?????????

p
? "??????????
?
8__inference_batch_normalization_5_layer_call_fn_25010752U41323?0
)?&
 ?
inputs?????????

p 
? "??????????
?
F__inference_dense_10_layer_call_and_return_conditional_losses_25010661\*+/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? ~
+__inference_dense_10_layer_call_fn_25010670O*+/?,
%?"
 ?
inputs?????????

? "??????????
?
F__inference_dense_11_layer_call_and_return_conditional_losses_25010760[9/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? }
+__inference_dense_11_layer_call_fn_25010767N9/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dense_8_layer_call_and_return_conditional_losses_25010459\0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
*__inference_dense_8_layer_call_fn_25010466O0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_9_layer_call_and_return_conditional_losses_25010559\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? }
*__inference_dense_9_layer_call_fn_25010568O/?,
%?"
 ?
inputs?????????
? "??????????
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009906|$%"#*+34129??<
5?2
(?%
dense_8_input??????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25009953|%"$#*+41329??<
5?2
(?%
dense_8_input??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010295u$%"#*+341298?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_25010369u%"$#*+413298?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_2_layer_call_fn_25010042o$%"#*+34129??<
5?2
(?%
dense_8_input??????????
p

 
? "???????????
/__inference_sequential_2_layer_call_fn_25010130o%"$#*+41329??<
5?2
(?%
dense_8_input??????????
p 

 
? "???????????
/__inference_sequential_2_layer_call_fn_25010410h$%"#*+341298?5
.?+
!?
inputs??????????
p

 
? "???????????
/__inference_sequential_2_layer_call_fn_25010451h%"$#*+413298?5
.?+
!?
inputs??????????
p 

 
? "???????????
&__inference_signature_wrapper_25010173?%"$#*+41329H?E
? 
>?;
9
dense_8_input(?%
dense_8_input??????????"3?0
.
dense_11"?
dense_11?????????