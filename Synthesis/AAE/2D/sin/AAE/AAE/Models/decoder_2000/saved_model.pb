’¾
Ķ¢
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8į¢
z
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_50/kernel
s
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes

:*
dtype0
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
:*
dtype0

batch_normalization_63/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_63/gamma

0batch_normalization_63/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_63/gamma*
_output_shapes
:*
dtype0

batch_normalization_63/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_63/beta

/batch_normalization_63/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_63/beta*
_output_shapes
:*
dtype0

"batch_normalization_63/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_63/moving_mean

6batch_normalization_63/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_63/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_63/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_63/moving_variance

:batch_normalization_63/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_63/moving_variance*
_output_shapes
:*
dtype0
z
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_51/kernel
s
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes

:
*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:
*
dtype0

batch_normalization_64/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_64/gamma

0batch_normalization_64/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_64/gamma*
_output_shapes
:
*
dtype0

batch_normalization_64/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_64/beta

/batch_normalization_64/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_64/beta*
_output_shapes
:
*
dtype0

"batch_normalization_64/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_64/moving_mean

6batch_normalization_64/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_64/moving_mean*
_output_shapes
:
*
dtype0
¤
&batch_normalization_64/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_64/moving_variance

:batch_normalization_64/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_64/moving_variance*
_output_shapes
:
*
dtype0
z
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_52/kernel
s
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes

:

*
dtype0
r
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_52/bias
k
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes
:
*
dtype0

batch_normalization_65/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_65/gamma

0batch_normalization_65/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_65/gamma*
_output_shapes
:
*
dtype0

batch_normalization_65/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_65/beta

/batch_normalization_65/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_65/beta*
_output_shapes
:
*
dtype0

"batch_normalization_65/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_65/moving_mean

6batch_normalization_65/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_65/moving_mean*
_output_shapes
:
*
dtype0
¤
&batch_normalization_65/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_65/moving_variance

:batch_normalization_65/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_65/moving_variance*
_output_shapes
:
*
dtype0
z
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_53/kernel
s
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes

:
*
dtype0
r
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_53/bias
k
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ń+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¬+
value¢+B+ B+

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api

#axis
	$gamma
%beta
&moving_mean
'moving_variance
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api

2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
 
f
0
1
2
3
4
5
$6
%7
,8
-9
310
411
;12
<13

0
1
2
3
4
5
6
7
$8
%9
&10
'11
,12
-13
314
415
516
617
;18
<19
­
	regularization_losses
Alayer_metrics
Bmetrics
Cnon_trainable_variables

trainable_variables
	variables
Dlayer_regularization_losses

Elayers
 
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Flayer_metrics
Gmetrics
Hnon_trainable_variables
trainable_variables
	variables
Ilayer_regularization_losses

Jlayers
 
ge
VARIABLE_VALUEbatch_normalization_63/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_63/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_63/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_63/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
­
regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
trainable_variables
	variables
Nlayer_regularization_losses

Olayers
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
 trainable_variables
!	variables
Slayer_regularization_losses

Tlayers
 
ge
VARIABLE_VALUEbatch_normalization_64/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_64/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_64/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_64/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
&2
'3
­
(regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
)trainable_variables
*	variables
Xlayer_regularization_losses

Ylayers
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
­
.regularization_losses
Zlayer_metrics
[metrics
\non_trainable_variables
/trainable_variables
0	variables
]layer_regularization_losses

^layers
 
ge
VARIABLE_VALUEbatch_normalization_65/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_65/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_65/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_65/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
52
63
­
7regularization_losses
_layer_metrics
`metrics
anon_trainable_variables
8trainable_variables
9	variables
blayer_regularization_losses

clayers
[Y
VARIABLE_VALUEdense_53/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_53/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
­
=regularization_losses
dlayer_metrics
emetrics
fnon_trainable_variables
>trainable_variables
?	variables
glayer_regularization_losses

hlayers
 
 
*
0
1
&2
'3
54
65
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 

0
1
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
50
61
 
 
 
 
 
 
 
{
serving_default_input_20Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
ņ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_20dense_50/kerneldense_50/bias&batch_normalization_63/moving_variancebatch_normalization_63/gamma"batch_normalization_63/moving_meanbatch_normalization_63/betadense_51/kerneldense_51/bias&batch_normalization_64/moving_variancebatch_normalization_64/gamma"batch_normalization_64/moving_meanbatch_normalization_64/betadense_52/kerneldense_52/bias&batch_normalization_65/moving_variancebatch_normalization_65/gamma"batch_normalization_65/moving_meanbatch_normalization_65/betadense_53/kerneldense_53/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_41914766
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ö	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp0batch_normalization_63/gamma/Read/ReadVariableOp/batch_normalization_63/beta/Read/ReadVariableOp6batch_normalization_63/moving_mean/Read/ReadVariableOp:batch_normalization_63/moving_variance/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp0batch_normalization_64/gamma/Read/ReadVariableOp/batch_normalization_64/beta/Read/ReadVariableOp6batch_normalization_64/moving_mean/Read/ReadVariableOp:batch_normalization_64/moving_variance/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp0batch_normalization_65/gamma/Read/ReadVariableOp/batch_normalization_65/beta/Read/ReadVariableOp6batch_normalization_65/moving_mean/Read/ReadVariableOp:batch_normalization_65/moving_variance/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOpConst*!
Tin
2*
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
!__inference__traced_save_41915473
Į
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_50/kerneldense_50/biasbatch_normalization_63/gammabatch_normalization_63/beta"batch_normalization_63/moving_mean&batch_normalization_63/moving_variancedense_51/kerneldense_51/biasbatch_normalization_64/gammabatch_normalization_64/beta"batch_normalization_64/moving_mean&batch_normalization_64/moving_variancedense_52/kerneldense_52/biasbatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_variancedense_53/kerneldense_53/bias* 
Tin
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_41915543³
×

T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41914246

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
š	
ß
F__inference_dense_50_layer_call_and_return_conditional_losses_41915075

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
š	
ß
F__inference_dense_51_layer_call_and_return_conditional_losses_41915177

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
×

T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41914106

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
į

+__inference_dense_53_layer_call_fn_41915390

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_419144582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

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
š	
ß
F__inference_dense_52_layer_call_and_return_conditional_losses_41915279

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
×

T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41913966

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
Ŗ-
ł
E__inference_Decoder_layer_call_and_return_conditional_losses_41914475
input_20
dense_50_41914283
dense_50_41914285#
batch_normalization_63_41914314#
batch_normalization_63_41914316#
batch_normalization_63_41914318#
batch_normalization_63_41914320
dense_51_41914345
dense_51_41914347#
batch_normalization_64_41914376#
batch_normalization_64_41914378#
batch_normalization_64_41914380#
batch_normalization_64_41914382
dense_52_41914407
dense_52_41914409#
batch_normalization_65_41914438#
batch_normalization_65_41914440#
batch_normalization_65_41914442#
batch_normalization_65_41914444
dense_53_41914469
dense_53_41914471
identity¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢.batch_normalization_65/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_50_41914283dense_50_41914285*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_419142722"
 dense_50/StatefulPartitionedCallĒ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_63_41914314batch_normalization_63_41914316batch_normalization_63_41914318batch_normalization_63_41914320*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_4191393320
.batch_normalization_63/StatefulPartitionedCallĖ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0dense_51_41914345dense_51_41914347*
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
GPU 2J 8 *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_419143342"
 dense_51/StatefulPartitionedCallĒ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_64_41914376batch_normalization_64_41914378batch_normalization_64_41914380batch_normalization_64_41914382*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_4191407320
.batch_normalization_64/StatefulPartitionedCallĖ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0dense_52_41914407dense_52_41914409*
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
GPU 2J 8 *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_419143962"
 dense_52/StatefulPartitionedCallĒ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_65_41914438batch_normalization_65_41914440batch_normalization_65_41914442batch_normalization_65_41914444*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_4191421320
.batch_normalization_65/StatefulPartitionedCallĖ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_53_41914469dense_53_41914471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_419144582"
 dense_53/StatefulPartitionedCall
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
į

+__inference_dense_52_layer_call_fn_41915288

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_419143962
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
½
¬
9__inference_batch_normalization_65_layer_call_fn_41915370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_419142462
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
¦0
Ļ
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41914213

inputs
assignmovingavg_41914188
assignmovingavg_1_41914194)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41914188*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41914188*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41914188*
_output_shapes
:
2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41914188*
_output_shapes
:
2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41914188AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41914188*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41914194*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41914194*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41914194*
_output_shapes
:
2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41914194*
_output_shapes
:
2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41914194AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41914194*
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
¦0
Ļ
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41913933

inputs
assignmovingavg_41913908
assignmovingavg_1_41913914)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913908*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41913908*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913908*
_output_shapes
:2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41913908*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41913908AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41913908*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913914*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41913914*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913914*
_output_shapes
:2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41913914*
_output_shapes
:2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41913914AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41913914*
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
4
ļ	
!__inference__traced_save_41915473
file_prefix.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop;
7savev2_batch_normalization_63_gamma_read_readvariableop:
6savev2_batch_normalization_63_beta_read_readvariableopA
=savev2_batch_normalization_63_moving_mean_read_readvariableopE
Asavev2_batch_normalization_63_moving_variance_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop;
7savev2_batch_normalization_64_gamma_read_readvariableop:
6savev2_batch_normalization_64_beta_read_readvariableopA
=savev2_batch_normalization_64_moving_mean_read_readvariableopE
Asavev2_batch_normalization_64_moving_variance_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop;
7savev2_batch_normalization_65_gamma_read_readvariableop:
6savev2_batch_normalization_65_beta_read_readvariableopA
=savev2_batch_normalization_65_moving_mean_read_readvariableopE
Asavev2_batch_normalization_65_moving_variance_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop
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
ShardedFilename

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¬	
value¢	B	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices’	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop7savev2_batch_normalization_63_gamma_read_readvariableop6savev2_batch_normalization_63_beta_read_readvariableop=savev2_batch_normalization_63_moving_mean_read_readvariableopAsavev2_batch_normalization_63_moving_variance_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop7savev2_batch_normalization_64_gamma_read_readvariableop6savev2_batch_normalization_64_beta_read_readvariableop=savev2_batch_normalization_64_moving_mean_read_readvariableopAsavev2_batch_normalization_64_moving_variance_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop7savev2_batch_normalization_65_gamma_read_readvariableop6savev2_batch_normalization_65_beta_read_readvariableop=savev2_batch_normalization_65_moving_mean_read_readvariableopAsavev2_batch_normalization_65_moving_variance_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
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

identity_1Identity_1:output:0*”
_input_shapes
: :::::::
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
:: 2(
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
::

_output_shapes
: 
š	
ß
F__inference_dense_50_layer_call_and_return_conditional_losses_41914272

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
į

+__inference_dense_51_layer_call_fn_41915186

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_419143342
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
×

T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915242

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
»
¬
9__inference_batch_normalization_64_layer_call_fn_41915255

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_419140732
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
ØX
ą
$__inference__traced_restore_41915543
file_prefix$
 assignvariableop_dense_50_kernel$
 assignvariableop_1_dense_50_bias3
/assignvariableop_2_batch_normalization_63_gamma2
.assignvariableop_3_batch_normalization_63_beta9
5assignvariableop_4_batch_normalization_63_moving_mean=
9assignvariableop_5_batch_normalization_63_moving_variance&
"assignvariableop_6_dense_51_kernel$
 assignvariableop_7_dense_51_bias3
/assignvariableop_8_batch_normalization_64_gamma2
.assignvariableop_9_batch_normalization_64_beta:
6assignvariableop_10_batch_normalization_64_moving_mean>
:assignvariableop_11_batch_normalization_64_moving_variance'
#assignvariableop_12_dense_52_kernel%
!assignvariableop_13_dense_52_bias4
0assignvariableop_14_batch_normalization_65_gamma3
/assignvariableop_15_batch_normalization_65_beta:
6assignvariableop_16_batch_normalization_65_moving_mean>
:assignvariableop_17_batch_normalization_65_moving_variance'
#assignvariableop_18_dense_53_kernel%
!assignvariableop_19_dense_53_bias
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9 

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¬	
value¢	B	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesø
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_50_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1„
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_50_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2“
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_63_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_63_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ŗ
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_63_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¾
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_63_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_51_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7„
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_51_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8“
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_64_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_64_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¾
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_64_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ā
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_64_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_52_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_52_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ø
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_65_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_65_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¾
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_65_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ā
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_65_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_53_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_53_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
°-
ł
E__inference_Decoder_layer_call_and_return_conditional_losses_41914526
input_20
dense_50_41914478
dense_50_41914480#
batch_normalization_63_41914483#
batch_normalization_63_41914485#
batch_normalization_63_41914487#
batch_normalization_63_41914489
dense_51_41914492
dense_51_41914494#
batch_normalization_64_41914497#
batch_normalization_64_41914499#
batch_normalization_64_41914501#
batch_normalization_64_41914503
dense_52_41914506
dense_52_41914508#
batch_normalization_65_41914511#
batch_normalization_65_41914513#
batch_normalization_65_41914515#
batch_normalization_65_41914517
dense_53_41914520
dense_53_41914522
identity¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢.batch_normalization_65/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_50_41914478dense_50_41914480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_419142722"
 dense_50/StatefulPartitionedCallÉ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_63_41914483batch_normalization_63_41914485batch_normalization_63_41914487batch_normalization_63_41914489*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_4191396620
.batch_normalization_63/StatefulPartitionedCallĖ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0dense_51_41914492dense_51_41914494*
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
GPU 2J 8 *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_419143342"
 dense_51/StatefulPartitionedCallÉ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_64_41914497batch_normalization_64_41914499batch_normalization_64_41914501batch_normalization_64_41914503*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_4191410620
.batch_normalization_64/StatefulPartitionedCallĖ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0dense_52_41914506dense_52_41914508*
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
GPU 2J 8 *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_419143962"
 dense_52/StatefulPartitionedCallÉ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_65_41914511batch_normalization_65_41914513batch_normalization_65_41914515batch_normalization_65_41914517*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_4191424620
.batch_normalization_65/StatefulPartitionedCallĖ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_53_41914520dense_53_41914522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_419144582"
 dense_53/StatefulPartitionedCall
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
¬
ģ
#__inference__wrapped_model_41913837
input_203
/decoder_dense_50_matmul_readvariableop_resource4
0decoder_dense_50_biasadd_readvariableop_resourceD
@decoder_batch_normalization_63_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_63_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_63_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_63_batchnorm_readvariableop_2_resource3
/decoder_dense_51_matmul_readvariableop_resource4
0decoder_dense_51_biasadd_readvariableop_resourceD
@decoder_batch_normalization_64_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_64_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_64_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_64_batchnorm_readvariableop_2_resource3
/decoder_dense_52_matmul_readvariableop_resource4
0decoder_dense_52_biasadd_readvariableop_resourceD
@decoder_batch_normalization_65_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_65_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_65_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_65_batchnorm_readvariableop_2_resource3
/decoder_dense_53_matmul_readvariableop_resource4
0decoder_dense_53_biasadd_readvariableop_resource
identity¢7Decoder/batch_normalization_63/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOp¢7Decoder/batch_normalization_64/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOp¢7Decoder/batch_normalization_65/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOp¢'Decoder/dense_50/BiasAdd/ReadVariableOp¢&Decoder/dense_50/MatMul/ReadVariableOp¢'Decoder/dense_51/BiasAdd/ReadVariableOp¢&Decoder/dense_51/MatMul/ReadVariableOp¢'Decoder/dense_52/BiasAdd/ReadVariableOp¢&Decoder/dense_52/MatMul/ReadVariableOp¢'Decoder/dense_53/BiasAdd/ReadVariableOp¢&Decoder/dense_53/MatMul/ReadVariableOpĄ
&Decoder/dense_50/MatMul/ReadVariableOpReadVariableOp/decoder_dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Decoder/dense_50/MatMul/ReadVariableOpØ
Decoder/dense_50/MatMulMatMulinput_20.Decoder/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_50/MatMulæ
'Decoder/dense_50/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Decoder/dense_50/BiasAdd/ReadVariableOpÅ
Decoder/dense_50/BiasAddBiasAdd!Decoder/dense_50/MatMul:product:0/Decoder/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_50/BiasAdd
Decoder/dense_50/ReluRelu!Decoder/dense_50/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_50/Reluļ
7Decoder/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Decoder/batch_normalization_63/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_63/batchnorm/add/y
,Decoder/batch_normalization_63/batchnorm/addAddV2?Decoder/batch_normalization_63/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_63/batchnorm/addĄ
.Decoder/batch_normalization_63/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_63/batchnorm/Rsqrtū
;Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_63/batchnorm/mulMul2Decoder/batch_normalization_63/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_63/batchnorm/mulš
.Decoder/batch_normalization_63/batchnorm/mul_1Mul#Decoder/dense_50/Relu:activations:00Decoder/batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’20
.Decoder/batch_normalization_63/batchnorm/mul_1õ
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_63/batchnorm/mul_2MulADecoder/batch_normalization_63/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_63/batchnorm/mul_2õ
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_63/batchnorm/subSubADecoder/batch_normalization_63/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_63/batchnorm/sub
.Decoder/batch_normalization_63/batchnorm/add_1AddV22Decoder/batch_normalization_63/batchnorm/mul_1:z:00Decoder/batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’20
.Decoder/batch_normalization_63/batchnorm/add_1Ą
&Decoder/dense_51/MatMul/ReadVariableOpReadVariableOp/decoder_dense_51_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Decoder/dense_51/MatMul/ReadVariableOpŅ
Decoder/dense_51/MatMulMatMul2Decoder/batch_normalization_63/batchnorm/add_1:z:0.Decoder/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_51/MatMulæ
'Decoder/dense_51/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_51_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Decoder/dense_51/BiasAdd/ReadVariableOpÅ
Decoder/dense_51/BiasAddBiasAdd!Decoder/dense_51/MatMul:product:0/Decoder/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_51/BiasAdd
Decoder/dense_51/ReluRelu!Decoder/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_51/Reluļ
7Decoder/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype029
7Decoder/batch_normalization_64/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_64/batchnorm/add/y
,Decoder/batch_normalization_64/batchnorm/addAddV2?Decoder/batch_normalization_64/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_64/batchnorm/addĄ
.Decoder/batch_normalization_64/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_64/batchnorm/Rsqrtū
;Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02=
;Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_64/batchnorm/mulMul2Decoder/batch_normalization_64/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_64/batchnorm/mulš
.Decoder/batch_normalization_64/batchnorm/mul_1Mul#Decoder/dense_51/Relu:activations:00Decoder/batch_normalization_64/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_64/batchnorm/mul_1õ
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_64_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_64/batchnorm/mul_2MulADecoder/batch_normalization_64/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_64/batchnorm/mul_2õ
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_64_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_64/batchnorm/subSubADecoder/batch_normalization_64/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_64/batchnorm/sub
.Decoder/batch_normalization_64/batchnorm/add_1AddV22Decoder/batch_normalization_64/batchnorm/mul_1:z:00Decoder/batch_normalization_64/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_64/batchnorm/add_1Ą
&Decoder/dense_52/MatMul/ReadVariableOpReadVariableOp/decoder_dense_52_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Decoder/dense_52/MatMul/ReadVariableOpŅ
Decoder/dense_52/MatMulMatMul2Decoder/batch_normalization_64/batchnorm/add_1:z:0.Decoder/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_52/MatMulæ
'Decoder/dense_52/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_52_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Decoder/dense_52/BiasAdd/ReadVariableOpÅ
Decoder/dense_52/BiasAddBiasAdd!Decoder/dense_52/MatMul:product:0/Decoder/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_52/BiasAdd
Decoder/dense_52/ReluRelu!Decoder/dense_52/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_52/Reluļ
7Decoder/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype029
7Decoder/batch_normalization_65/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_65/batchnorm/add/y
,Decoder/batch_normalization_65/batchnorm/addAddV2?Decoder/batch_normalization_65/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_65/batchnorm/addĄ
.Decoder/batch_normalization_65/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_65/batchnorm/Rsqrtū
;Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02=
;Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_65/batchnorm/mulMul2Decoder/batch_normalization_65/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_65/batchnorm/mulš
.Decoder/batch_normalization_65/batchnorm/mul_1Mul#Decoder/dense_52/Relu:activations:00Decoder/batch_normalization_65/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_65/batchnorm/mul_1õ
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_65_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_65/batchnorm/mul_2MulADecoder/batch_normalization_65/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_65/batchnorm/mul_2õ
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_65_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_65/batchnorm/subSubADecoder/batch_normalization_65/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_65/batchnorm/sub
.Decoder/batch_normalization_65/batchnorm/add_1AddV22Decoder/batch_normalization_65/batchnorm/mul_1:z:00Decoder/batch_normalization_65/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_65/batchnorm/add_1Ą
&Decoder/dense_53/MatMul/ReadVariableOpReadVariableOp/decoder_dense_53_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Decoder/dense_53/MatMul/ReadVariableOpŅ
Decoder/dense_53/MatMulMatMul2Decoder/batch_normalization_65/batchnorm/add_1:z:0.Decoder/dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_53/MatMulæ
'Decoder/dense_53/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Decoder/dense_53/BiasAdd/ReadVariableOpÅ
Decoder/dense_53/BiasAddBiasAdd!Decoder/dense_53/MatMul:product:0/Decoder/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_53/BiasAdd
Decoder/dense_53/TanhTanh!Decoder/dense_53/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_53/Tanh	
IdentityIdentityDecoder/dense_53/Tanh:y:08^Decoder/batch_normalization_63/batchnorm/ReadVariableOp:^Decoder/batch_normalization_63/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_63/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_64/batchnorm/ReadVariableOp:^Decoder/batch_normalization_64/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_64/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_65/batchnorm/ReadVariableOp:^Decoder/batch_normalization_65/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_65/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOp(^Decoder/dense_50/BiasAdd/ReadVariableOp'^Decoder/dense_50/MatMul/ReadVariableOp(^Decoder/dense_51/BiasAdd/ReadVariableOp'^Decoder/dense_51/MatMul/ReadVariableOp(^Decoder/dense_52/BiasAdd/ReadVariableOp'^Decoder/dense_52/MatMul/ReadVariableOp(^Decoder/dense_53/BiasAdd/ReadVariableOp'^Decoder/dense_53/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2r
7Decoder/batch_normalization_63/batchnorm/ReadVariableOp7Decoder/batch_normalization_63/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_19Decoder/batch_normalization_63/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_63/batchnorm/ReadVariableOp_29Decoder/batch_normalization_63/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_63/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_64/batchnorm/ReadVariableOp7Decoder/batch_normalization_64/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_19Decoder/batch_normalization_64/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_64/batchnorm/ReadVariableOp_29Decoder/batch_normalization_64/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_64/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_65/batchnorm/ReadVariableOp7Decoder/batch_normalization_65/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_19Decoder/batch_normalization_65/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_65/batchnorm/ReadVariableOp_29Decoder/batch_normalization_65/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_65/batchnorm/mul/ReadVariableOp2R
'Decoder/dense_50/BiasAdd/ReadVariableOp'Decoder/dense_50/BiasAdd/ReadVariableOp2P
&Decoder/dense_50/MatMul/ReadVariableOp&Decoder/dense_50/MatMul/ReadVariableOp2R
'Decoder/dense_51/BiasAdd/ReadVariableOp'Decoder/dense_51/BiasAdd/ReadVariableOp2P
&Decoder/dense_51/MatMul/ReadVariableOp&Decoder/dense_51/MatMul/ReadVariableOp2R
'Decoder/dense_52/BiasAdd/ReadVariableOp'Decoder/dense_52/BiasAdd/ReadVariableOp2P
&Decoder/dense_52/MatMul/ReadVariableOp&Decoder/dense_52/MatMul/ReadVariableOp2R
'Decoder/dense_53/BiasAdd/ReadVariableOp'Decoder/dense_53/BiasAdd/ReadVariableOp2P
&Decoder/dense_53/MatMul/ReadVariableOp&Decoder/dense_53/MatMul/ReadVariableOp:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
š	
ß
F__inference_dense_52_layer_call_and_return_conditional_losses_41914396

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
į

+__inference_dense_50_layer_call_fn_41915084

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_419142722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦0
Ļ
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915120

inputs
assignmovingavg_41915095
assignmovingavg_1_41915101)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915095*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41915095*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915095*
_output_shapes
:2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915095*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41915095AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915095*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915101*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41915101*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915101*
_output_shapes
:2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915101*
_output_shapes
:2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41915101AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915101*
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
½
¬
9__inference_batch_normalization_63_layer_call_fn_41915166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_419139662
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
š	
ß
F__inference_dense_51_layer_call_and_return_conditional_losses_41914334

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
¦0
Ļ
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915222

inputs
assignmovingavg_41915197
assignmovingavg_1_41915203)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915197*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41915197*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915197*
_output_shapes
:
2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915197*
_output_shapes
:
2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41915197AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915197*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915203*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41915203*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915203*
_output_shapes
:
2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915203*
_output_shapes
:
2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41915203AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915203*
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
¶ö
¢
E__inference_Decoder_layer_call_and_return_conditional_losses_41914894

inputs+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource3
/batch_normalization_63_assignmovingavg_419147845
1batch_normalization_63_assignmovingavg_1_41914790@
<batch_normalization_63_batchnorm_mul_readvariableop_resource<
8batch_normalization_63_batchnorm_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource3
/batch_normalization_64_assignmovingavg_419148235
1batch_normalization_64_assignmovingavg_1_41914829@
<batch_normalization_64_batchnorm_mul_readvariableop_resource<
8batch_normalization_64_batchnorm_readvariableop_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource3
/batch_normalization_65_assignmovingavg_419148625
1batch_normalization_65_assignmovingavg_1_41914868@
<batch_normalization_65_batchnorm_mul_readvariableop_resource<
8batch_normalization_65_batchnorm_readvariableop_resource+
'dense_53_matmul_readvariableop_resource,
(dense_53_biasadd_readvariableop_resource
identity¢:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_63/AssignMovingAvg/ReadVariableOp¢<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_63/batchnorm/ReadVariableOp¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢:batch_normalization_64/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_64/AssignMovingAvg/ReadVariableOp¢<batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_65/AssignMovingAvg/ReadVariableOp¢<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_65/batchnorm/ReadVariableOp¢3batch_normalization_65/batchnorm/mul/ReadVariableOp¢dense_50/BiasAdd/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/BiasAdd/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/BiasAdd/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/BiasAdd/ReadVariableOp¢dense_53/MatMul/ReadVariableOpØ
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_50/MatMul/ReadVariableOp
dense_50/MatMulMatMulinputs&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/MatMul§
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_50/BiasAdd/ReadVariableOp„
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/BiasAdds
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/Reluø
5batch_normalization_63/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_63/moments/mean/reduction_indicesé
#batch_normalization_63/moments/meanMeandense_50/Relu:activations:0>batch_normalization_63/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_63/moments/meanĮ
+batch_normalization_63/moments/StopGradientStopGradient,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_63/moments/StopGradientž
0batch_normalization_63/moments/SquaredDifferenceSquaredDifferencedense_50/Relu:activations:04batch_normalization_63/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
0batch_normalization_63/moments/SquaredDifferenceĄ
9batch_normalization_63/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_63/moments/variance/reduction_indices
'batch_normalization_63/moments/varianceMean4batch_normalization_63/moments/SquaredDifference:z:0Bbatch_normalization_63/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_63/moments/varianceÅ
&batch_normalization_63/moments/SqueezeSqueeze,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_63/moments/SqueezeĶ
(batch_normalization_63/moments/Squeeze_1Squeeze0batch_normalization_63/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_63/moments/Squeeze_1
,batch_normalization_63/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_63/AssignMovingAvg/41914784*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_63/AssignMovingAvg/decayŚ
5batch_normalization_63/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_63_assignmovingavg_41914784*
_output_shapes
:*
dtype027
5batch_normalization_63/AssignMovingAvg/ReadVariableOpę
*batch_normalization_63/AssignMovingAvg/subSub=batch_normalization_63/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_63/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_63/AssignMovingAvg/41914784*
_output_shapes
:2,
*batch_normalization_63/AssignMovingAvg/subŻ
*batch_normalization_63/AssignMovingAvg/mulMul.batch_normalization_63/AssignMovingAvg/sub:z:05batch_normalization_63/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_63/AssignMovingAvg/41914784*
_output_shapes
:2,
*batch_normalization_63/AssignMovingAvg/mul½
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_63_assignmovingavg_41914784.batch_normalization_63/AssignMovingAvg/mul:z:06^batch_normalization_63/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_63/AssignMovingAvg/41914784*
_output_shapes
 *
dtype02<
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_63/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_63/AssignMovingAvg_1/41914790*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_63/AssignMovingAvg_1/decayą
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_63_assignmovingavg_1_41914790*
_output_shapes
:*
dtype029
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpš
,batch_normalization_63/AssignMovingAvg_1/subSub?batch_normalization_63/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_63/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_63/AssignMovingAvg_1/41914790*
_output_shapes
:2.
,batch_normalization_63/AssignMovingAvg_1/subē
,batch_normalization_63/AssignMovingAvg_1/mulMul0batch_normalization_63/AssignMovingAvg_1/sub:z:07batch_normalization_63/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_63/AssignMovingAvg_1/41914790*
_output_shapes
:2.
,batch_normalization_63/AssignMovingAvg_1/mulÉ
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_63_assignmovingavg_1_419147900batch_normalization_63/AssignMovingAvg_1/mul:z:08^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_63/AssignMovingAvg_1/41914790*
_output_shapes
 *
dtype02>
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_63/batchnorm/add/yŽ
$batch_normalization_63/batchnorm/addAddV21batch_normalization_63/moments/Squeeze_1:output:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/addØ
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/Rsqrtć
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpį
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/mulŠ
&batch_normalization_63/batchnorm/mul_1Muldense_50/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_63/batchnorm/mul_1×
&batch_normalization_63/batchnorm/mul_2Mul/batch_normalization_63/moments/Squeeze:output:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/mul_2×
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpŻ
$batch_normalization_63/batchnorm/subSub7batch_normalization_63/batchnorm/ReadVariableOp:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/subį
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_63/batchnorm/add_1Ø
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_51/MatMul/ReadVariableOp²
dense_51/MatMulMatMul*batch_normalization_63/batchnorm/add_1:z:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/MatMul§
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_51/BiasAdd/ReadVariableOp„
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/BiasAdds
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/Reluø
5batch_normalization_64/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_64/moments/mean/reduction_indicesé
#batch_normalization_64/moments/meanMeandense_51/Relu:activations:0>batch_normalization_64/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_64/moments/meanĮ
+batch_normalization_64/moments/StopGradientStopGradient,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_64/moments/StopGradientž
0batch_normalization_64/moments/SquaredDifferenceSquaredDifferencedense_51/Relu:activations:04batch_normalization_64/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
22
0batch_normalization_64/moments/SquaredDifferenceĄ
9batch_normalization_64/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_64/moments/variance/reduction_indices
'batch_normalization_64/moments/varianceMean4batch_normalization_64/moments/SquaredDifference:z:0Bbatch_normalization_64/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_64/moments/varianceÅ
&batch_normalization_64/moments/SqueezeSqueeze,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_64/moments/SqueezeĶ
(batch_normalization_64/moments/Squeeze_1Squeeze0batch_normalization_64/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_64/moments/Squeeze_1
,batch_normalization_64/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_64/AssignMovingAvg/41914823*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_64/AssignMovingAvg/decayŚ
5batch_normalization_64/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_64_assignmovingavg_41914823*
_output_shapes
:
*
dtype027
5batch_normalization_64/AssignMovingAvg/ReadVariableOpę
*batch_normalization_64/AssignMovingAvg/subSub=batch_normalization_64/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_64/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_64/AssignMovingAvg/41914823*
_output_shapes
:
2,
*batch_normalization_64/AssignMovingAvg/subŻ
*batch_normalization_64/AssignMovingAvg/mulMul.batch_normalization_64/AssignMovingAvg/sub:z:05batch_normalization_64/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_64/AssignMovingAvg/41914823*
_output_shapes
:
2,
*batch_normalization_64/AssignMovingAvg/mul½
:batch_normalization_64/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_64_assignmovingavg_41914823.batch_normalization_64/AssignMovingAvg/mul:z:06^batch_normalization_64/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_64/AssignMovingAvg/41914823*
_output_shapes
 *
dtype02<
:batch_normalization_64/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_64/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_64/AssignMovingAvg_1/41914829*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_64/AssignMovingAvg_1/decayą
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_64_assignmovingavg_1_41914829*
_output_shapes
:
*
dtype029
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpš
,batch_normalization_64/AssignMovingAvg_1/subSub?batch_normalization_64/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_64/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_64/AssignMovingAvg_1/41914829*
_output_shapes
:
2.
,batch_normalization_64/AssignMovingAvg_1/subē
,batch_normalization_64/AssignMovingAvg_1/mulMul0batch_normalization_64/AssignMovingAvg_1/sub:z:07batch_normalization_64/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_64/AssignMovingAvg_1/41914829*
_output_shapes
:
2.
,batch_normalization_64/AssignMovingAvg_1/mulÉ
<batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_64_assignmovingavg_1_419148290batch_normalization_64/AssignMovingAvg_1/mul:z:08^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_64/AssignMovingAvg_1/41914829*
_output_shapes
 *
dtype02>
<batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_64/batchnorm/add/yŽ
$batch_normalization_64/batchnorm/addAddV21batch_normalization_64/moments/Squeeze_1:output:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/addØ
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_64/batchnorm/Rsqrtć
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpį
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/mulŠ
&batch_normalization_64/batchnorm/mul_1Muldense_51/Relu:activations:0(batch_normalization_64/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_64/batchnorm/mul_1×
&batch_normalization_64/batchnorm/mul_2Mul/batch_normalization_64/moments/Squeeze:output:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_64/batchnorm/mul_2×
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOpŻ
$batch_normalization_64/batchnorm/subSub7batch_normalization_64/batchnorm/ReadVariableOp:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/subį
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_64/batchnorm/add_1Ø
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_52/MatMul/ReadVariableOp²
dense_52/MatMulMatMul*batch_normalization_64/batchnorm/add_1:z:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/MatMul§
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_52/BiasAdd/ReadVariableOp„
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/BiasAdds
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/Reluø
5batch_normalization_65/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_65/moments/mean/reduction_indicesé
#batch_normalization_65/moments/meanMeandense_52/Relu:activations:0>batch_normalization_65/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_65/moments/meanĮ
+batch_normalization_65/moments/StopGradientStopGradient,batch_normalization_65/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_65/moments/StopGradientž
0batch_normalization_65/moments/SquaredDifferenceSquaredDifferencedense_52/Relu:activations:04batch_normalization_65/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
22
0batch_normalization_65/moments/SquaredDifferenceĄ
9batch_normalization_65/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_65/moments/variance/reduction_indices
'batch_normalization_65/moments/varianceMean4batch_normalization_65/moments/SquaredDifference:z:0Bbatch_normalization_65/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_65/moments/varianceÅ
&batch_normalization_65/moments/SqueezeSqueeze,batch_normalization_65/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_65/moments/SqueezeĶ
(batch_normalization_65/moments/Squeeze_1Squeeze0batch_normalization_65/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_65/moments/Squeeze_1
,batch_normalization_65/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_65/AssignMovingAvg/41914862*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_65/AssignMovingAvg/decayŚ
5batch_normalization_65/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_65_assignmovingavg_41914862*
_output_shapes
:
*
dtype027
5batch_normalization_65/AssignMovingAvg/ReadVariableOpę
*batch_normalization_65/AssignMovingAvg/subSub=batch_normalization_65/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_65/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_65/AssignMovingAvg/41914862*
_output_shapes
:
2,
*batch_normalization_65/AssignMovingAvg/subŻ
*batch_normalization_65/AssignMovingAvg/mulMul.batch_normalization_65/AssignMovingAvg/sub:z:05batch_normalization_65/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_65/AssignMovingAvg/41914862*
_output_shapes
:
2,
*batch_normalization_65/AssignMovingAvg/mul½
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_65_assignmovingavg_41914862.batch_normalization_65/AssignMovingAvg/mul:z:06^batch_normalization_65/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_65/AssignMovingAvg/41914862*
_output_shapes
 *
dtype02<
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_65/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_65/AssignMovingAvg_1/41914868*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_65/AssignMovingAvg_1/decayą
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_65_assignmovingavg_1_41914868*
_output_shapes
:
*
dtype029
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOpš
,batch_normalization_65/AssignMovingAvg_1/subSub?batch_normalization_65/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_65/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_65/AssignMovingAvg_1/41914868*
_output_shapes
:
2.
,batch_normalization_65/AssignMovingAvg_1/subē
,batch_normalization_65/AssignMovingAvg_1/mulMul0batch_normalization_65/AssignMovingAvg_1/sub:z:07batch_normalization_65/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_65/AssignMovingAvg_1/41914868*
_output_shapes
:
2.
,batch_normalization_65/AssignMovingAvg_1/mulÉ
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_65_assignmovingavg_1_419148680batch_normalization_65/AssignMovingAvg_1/mul:z:08^batch_normalization_65/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_65/AssignMovingAvg_1/41914868*
_output_shapes
 *
dtype02>
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_65/batchnorm/add/yŽ
$batch_normalization_65/batchnorm/addAddV21batch_normalization_65/moments/Squeeze_1:output:0/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/addØ
&batch_normalization_65/batchnorm/RsqrtRsqrt(batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_65/batchnorm/Rsqrtć
3batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_65/batchnorm/mul/ReadVariableOpį
$batch_normalization_65/batchnorm/mulMul*batch_normalization_65/batchnorm/Rsqrt:y:0;batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/mulŠ
&batch_normalization_65/batchnorm/mul_1Muldense_52/Relu:activations:0(batch_normalization_65/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_65/batchnorm/mul_1×
&batch_normalization_65/batchnorm/mul_2Mul/batch_normalization_65/moments/Squeeze:output:0(batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_65/batchnorm/mul_2×
/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_65/batchnorm/ReadVariableOpŻ
$batch_normalization_65/batchnorm/subSub7batch_normalization_65/batchnorm/ReadVariableOp:value:0*batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/subį
&batch_normalization_65/batchnorm/add_1AddV2*batch_normalization_65/batchnorm/mul_1:z:0(batch_normalization_65/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_65/batchnorm/add_1Ø
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_53/MatMul/ReadVariableOp²
dense_53/MatMulMatMul*batch_normalization_65/batchnorm/add_1:z:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/MatMul§
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_53/BiasAdd/ReadVariableOp„
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/BiasAdds
dense_53/TanhTanhdense_53/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/Tanhó

IdentityIdentitydense_53/Tanh:y:0;^batch_normalization_63/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_63/AssignMovingAvg/ReadVariableOp=^batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp4^batch_normalization_63/batchnorm/mul/ReadVariableOp;^batch_normalization_64/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_64/AssignMovingAvg/ReadVariableOp=^batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp4^batch_normalization_64/batchnorm/mul/ReadVariableOp;^batch_normalization_65/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_65/AssignMovingAvg/ReadVariableOp=^batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_65/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_65/batchnorm/ReadVariableOp4^batch_normalization_65/batchnorm/mul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2x
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_63/AssignMovingAvg/ReadVariableOp5batch_normalization_63/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2x
:batch_normalization_64/AssignMovingAvg/AssignSubVariableOp:batch_normalization_64/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_64/AssignMovingAvg/ReadVariableOp5batch_normalization_64/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_64/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2x
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_65/AssignMovingAvg/ReadVariableOp5batch_normalization_65/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_65/batchnorm/ReadVariableOp/batch_normalization_65/batchnorm/ReadVariableOp2j
3batch_normalization_65/batchnorm/mul/ReadVariableOp3batch_normalization_65/batchnorm/mul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
»
¬
9__inference_batch_normalization_65_layer_call_fn_41915357

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_419142132
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


&__inference_signature_wrapper_41914766
input_20
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
identity¢StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:’’’’’’’’’*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_419138372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
×

T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915140

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
¦0
Ļ
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41914073

inputs
assignmovingavg_41914048
assignmovingavg_1_41914054)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41914048*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41914048*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41914048*
_output_shapes
:
2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41914048*
_output_shapes
:
2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41914048AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41914048*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41914054*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41914054*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41914054*
_output_shapes
:
2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41914054*
_output_shapes
:
2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41914054AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41914054*
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
»
¬
9__inference_batch_normalization_63_layer_call_fn_41915153

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_419139332
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
·

*__inference_Decoder_layer_call_fn_41914719
input_20
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
identity¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:’’’’’’’’’*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_419146762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
¤-
÷
E__inference_Decoder_layer_call_and_return_conditional_losses_41914580

inputs
dense_50_41914532
dense_50_41914534#
batch_normalization_63_41914537#
batch_normalization_63_41914539#
batch_normalization_63_41914541#
batch_normalization_63_41914543
dense_51_41914546
dense_51_41914548#
batch_normalization_64_41914551#
batch_normalization_64_41914553#
batch_normalization_64_41914555#
batch_normalization_64_41914557
dense_52_41914560
dense_52_41914562#
batch_normalization_65_41914565#
batch_normalization_65_41914567#
batch_normalization_65_41914569#
batch_normalization_65_41914571
dense_53_41914574
dense_53_41914576
identity¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢.batch_normalization_65/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_41914532dense_50_41914534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_419142722"
 dense_50/StatefulPartitionedCallĒ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_63_41914537batch_normalization_63_41914539batch_normalization_63_41914541batch_normalization_63_41914543*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_4191393320
.batch_normalization_63/StatefulPartitionedCallĖ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0dense_51_41914546dense_51_41914548*
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
GPU 2J 8 *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_419143342"
 dense_51/StatefulPartitionedCallĒ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_64_41914551batch_normalization_64_41914553batch_normalization_64_41914555batch_normalization_64_41914557*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_4191407320
.batch_normalization_64/StatefulPartitionedCallĖ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0dense_52_41914560dense_52_41914562*
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
GPU 2J 8 *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_419143962"
 dense_52/StatefulPartitionedCallĒ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_65_41914565batch_normalization_65_41914567batch_normalization_65_41914569batch_normalization_65_41914571*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_4191421320
.batch_normalization_65/StatefulPartitionedCallĖ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_53_41914574dense_53_41914576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_419144582"
 dense_53/StatefulPartitionedCall
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ŗ-
÷
E__inference_Decoder_layer_call_and_return_conditional_losses_41914676

inputs
dense_50_41914628
dense_50_41914630#
batch_normalization_63_41914633#
batch_normalization_63_41914635#
batch_normalization_63_41914637#
batch_normalization_63_41914639
dense_51_41914642
dense_51_41914644#
batch_normalization_64_41914647#
batch_normalization_64_41914649#
batch_normalization_64_41914651#
batch_normalization_64_41914653
dense_52_41914656
dense_52_41914658#
batch_normalization_65_41914661#
batch_normalization_65_41914663#
batch_normalization_65_41914665#
batch_normalization_65_41914667
dense_53_41914670
dense_53_41914672
identity¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢.batch_normalization_65/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_41914628dense_50_41914630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_419142722"
 dense_50/StatefulPartitionedCallÉ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_63_41914633batch_normalization_63_41914635batch_normalization_63_41914637batch_normalization_63_41914639*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_4191396620
.batch_normalization_63/StatefulPartitionedCallĖ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0dense_51_41914642dense_51_41914644*
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
GPU 2J 8 *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_419143342"
 dense_51/StatefulPartitionedCallÉ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_64_41914647batch_normalization_64_41914649batch_normalization_64_41914651batch_normalization_64_41914653*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_4191410620
.batch_normalization_64/StatefulPartitionedCallĖ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0dense_52_41914656dense_52_41914658*
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
GPU 2J 8 *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_419143962"
 dense_52/StatefulPartitionedCallÉ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_65_41914661batch_normalization_65_41914663batch_normalization_65_41914665batch_normalization_65_41914667*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_4191424620
.batch_normalization_65/StatefulPartitionedCallĖ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_53_41914670dense_53_41914672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_419144582"
 dense_53/StatefulPartitionedCall
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ę	
ß
F__inference_dense_53_layer_call_and_return_conditional_losses_41915381

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

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
±

*__inference_Decoder_layer_call_fn_41914623
input_20
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
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:’’’’’’’’’*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_419145802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_20
ę	
ß
F__inference_dense_53_layer_call_and_return_conditional_losses_41914458

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
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

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
½
¬
9__inference_batch_normalization_64_layer_call_fn_41915268

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_419141062
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
±

*__inference_Decoder_layer_call_fn_41915064

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
identity¢StatefulPartitionedCallč
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
:’’’’’’’’’*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_419146762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¬}
Ģ
E__inference_Decoder_layer_call_and_return_conditional_losses_41914974

inputs+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource<
8batch_normalization_63_batchnorm_readvariableop_resource@
<batch_normalization_63_batchnorm_mul_readvariableop_resource>
:batch_normalization_63_batchnorm_readvariableop_1_resource>
:batch_normalization_63_batchnorm_readvariableop_2_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource<
8batch_normalization_64_batchnorm_readvariableop_resource@
<batch_normalization_64_batchnorm_mul_readvariableop_resource>
:batch_normalization_64_batchnorm_readvariableop_1_resource>
:batch_normalization_64_batchnorm_readvariableop_2_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource<
8batch_normalization_65_batchnorm_readvariableop_resource@
<batch_normalization_65_batchnorm_mul_readvariableop_resource>
:batch_normalization_65_batchnorm_readvariableop_1_resource>
:batch_normalization_65_batchnorm_readvariableop_2_resource+
'dense_53_matmul_readvariableop_resource,
(dense_53_biasadd_readvariableop_resource
identity¢/batch_normalization_63/batchnorm/ReadVariableOp¢1batch_normalization_63/batchnorm/ReadVariableOp_1¢1batch_normalization_63/batchnorm/ReadVariableOp_2¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢1batch_normalization_64/batchnorm/ReadVariableOp_1¢1batch_normalization_64/batchnorm/ReadVariableOp_2¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢/batch_normalization_65/batchnorm/ReadVariableOp¢1batch_normalization_65/batchnorm/ReadVariableOp_1¢1batch_normalization_65/batchnorm/ReadVariableOp_2¢3batch_normalization_65/batchnorm/mul/ReadVariableOp¢dense_50/BiasAdd/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/BiasAdd/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/BiasAdd/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/BiasAdd/ReadVariableOp¢dense_53/MatMul/ReadVariableOpØ
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_50/MatMul/ReadVariableOp
dense_50/MatMulMatMulinputs&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/MatMul§
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_50/BiasAdd/ReadVariableOp„
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/BiasAdds
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_50/Relu×
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOp
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_63/batchnorm/add/yä
$batch_normalization_63/batchnorm/addAddV27batch_normalization_63/batchnorm/ReadVariableOp:value:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/addØ
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/Rsqrtć
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpį
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/mulŠ
&batch_normalization_63/batchnorm/mul_1Muldense_50/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_63/batchnorm/mul_1Ż
1batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_1į
&batch_normalization_63/batchnorm/mul_2Mul9batch_normalization_63/batchnorm/ReadVariableOp_1:value:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_63/batchnorm/mul_2Ż
1batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_2ß
$batch_normalization_63/batchnorm/subSub9batch_normalization_63/batchnorm/ReadVariableOp_2:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_63/batchnorm/subį
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_63/batchnorm/add_1Ø
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_51/MatMul/ReadVariableOp²
dense_51/MatMulMatMul*batch_normalization_63/batchnorm/add_1:z:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/MatMul§
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_51/BiasAdd/ReadVariableOp„
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/BiasAdds
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_51/Relu×
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOp
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_64/batchnorm/add/yä
$batch_normalization_64/batchnorm/addAddV27batch_normalization_64/batchnorm/ReadVariableOp:value:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/addØ
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_64/batchnorm/Rsqrtć
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpį
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/mulŠ
&batch_normalization_64/batchnorm/mul_1Muldense_51/Relu:activations:0(batch_normalization_64/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_64/batchnorm/mul_1Ż
1batch_normalization_64/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_1į
&batch_normalization_64/batchnorm/mul_2Mul9batch_normalization_64/batchnorm/ReadVariableOp_1:value:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_64/batchnorm/mul_2Ż
1batch_normalization_64/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_2ß
$batch_normalization_64/batchnorm/subSub9batch_normalization_64/batchnorm/ReadVariableOp_2:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_64/batchnorm/subį
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_64/batchnorm/add_1Ø
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_52/MatMul/ReadVariableOp²
dense_52/MatMulMatMul*batch_normalization_64/batchnorm/add_1:z:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/MatMul§
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_52/BiasAdd/ReadVariableOp„
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/BiasAdds
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_52/Relu×
/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_65/batchnorm/ReadVariableOp
&batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_65/batchnorm/add/yä
$batch_normalization_65/batchnorm/addAddV27batch_normalization_65/batchnorm/ReadVariableOp:value:0/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/addØ
&batch_normalization_65/batchnorm/RsqrtRsqrt(batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_65/batchnorm/Rsqrtć
3batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_65/batchnorm/mul/ReadVariableOpį
$batch_normalization_65/batchnorm/mulMul*batch_normalization_65/batchnorm/Rsqrt:y:0;batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/mulŠ
&batch_normalization_65/batchnorm/mul_1Muldense_52/Relu:activations:0(batch_normalization_65/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_65/batchnorm/mul_1Ż
1batch_normalization_65/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_65_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_65/batchnorm/ReadVariableOp_1į
&batch_normalization_65/batchnorm/mul_2Mul9batch_normalization_65/batchnorm/ReadVariableOp_1:value:0(batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_65/batchnorm/mul_2Ż
1batch_normalization_65/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_65_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_65/batchnorm/ReadVariableOp_2ß
$batch_normalization_65/batchnorm/subSub9batch_normalization_65/batchnorm/ReadVariableOp_2:value:0*batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_65/batchnorm/subį
&batch_normalization_65/batchnorm/add_1AddV2*batch_normalization_65/batchnorm/mul_1:z:0(batch_normalization_65/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_65/batchnorm/add_1Ø
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_53/MatMul/ReadVariableOp²
dense_53/MatMulMatMul*batch_normalization_65/batchnorm/add_1:z:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/MatMul§
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_53/BiasAdd/ReadVariableOp„
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/BiasAdds
dense_53/TanhTanhdense_53/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_53/Tanhį
IdentityIdentitydense_53/Tanh:y:00^batch_normalization_63/batchnorm/ReadVariableOp2^batch_normalization_63/batchnorm/ReadVariableOp_12^batch_normalization_63/batchnorm/ReadVariableOp_24^batch_normalization_63/batchnorm/mul/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp2^batch_normalization_64/batchnorm/ReadVariableOp_12^batch_normalization_64/batchnorm/ReadVariableOp_24^batch_normalization_64/batchnorm/mul/ReadVariableOp0^batch_normalization_65/batchnorm/ReadVariableOp2^batch_normalization_65/batchnorm/ReadVariableOp_12^batch_normalization_65/batchnorm/ReadVariableOp_24^batch_normalization_65/batchnorm/mul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2f
1batch_normalization_63/batchnorm/ReadVariableOp_11batch_normalization_63/batchnorm/ReadVariableOp_12f
1batch_normalization_63/batchnorm/ReadVariableOp_21batch_normalization_63/batchnorm/ReadVariableOp_22j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2f
1batch_normalization_64/batchnorm/ReadVariableOp_11batch_normalization_64/batchnorm/ReadVariableOp_12f
1batch_normalization_64/batchnorm/ReadVariableOp_21batch_normalization_64/batchnorm/ReadVariableOp_22j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2b
/batch_normalization_65/batchnorm/ReadVariableOp/batch_normalization_65/batchnorm/ReadVariableOp2f
1batch_normalization_65/batchnorm/ReadVariableOp_11batch_normalization_65/batchnorm/ReadVariableOp_12f
1batch_normalization_65/batchnorm/ReadVariableOp_21batch_normalization_65/batchnorm/ReadVariableOp_22j
3batch_normalization_65/batchnorm/mul/ReadVariableOp3batch_normalization_65/batchnorm/mul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
«

*__inference_Decoder_layer_call_fn_41915019

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
identity¢StatefulPartitionedCallā
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
:’’’’’’’’’*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Decoder_layer_call_and_return_conditional_losses_419145802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*v
_input_shapese
c:’’’’’’’’’::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦0
Ļ
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915324

inputs
assignmovingavg_41915299
assignmovingavg_1_41915305)
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
moments/Squeeze_1Ī
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915299*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41915299*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915299*
_output_shapes
:
2
AssignMovingAvg/subź
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915299*
_output_shapes
:
2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41915299AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915299*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpŌ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915305*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41915305*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpż
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915305*
_output_shapes
:
2
AssignMovingAvg_1/subō
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915305*
_output_shapes
:
2
AssignMovingAvg_1/mulæ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41915305AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915305*
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
×

T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915344

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

 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
=
input_201
serving_default_input_20:0’’’’’’’’’<
dense_530
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ē
K
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
*i&call_and_return_all_conditional_losses
j__call__
k_default_save_signature"°G
_tf_keras_networkG{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_20"}, "name": "input_20", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_50", "inbound_nodes": [[["input_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_63", "inbound_nodes": [[["dense_50", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_51", "inbound_nodes": [[["batch_normalization_63", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_64", "inbound_nodes": [[["dense_51", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_52", "inbound_nodes": [[["batch_normalization_64", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["dense_52", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_53", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}], "input_layers": [["input_20", 0, 0]], "output_layers": [["dense_53", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_20"}, "name": "input_20", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_50", "inbound_nodes": [[["input_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_63", "inbound_nodes": [[["dense_50", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_51", "inbound_nodes": [[["batch_normalization_63", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_64", "inbound_nodes": [[["dense_51", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_52", "inbound_nodes": [[["batch_normalization_64", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["dense_52", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_53", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}], "input_layers": [["input_20", 0, 0]], "output_layers": [["dense_53", 0, 0]]}}}
ė"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_20", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_20"}}
ņ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
“	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
ō

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Ļ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
“	
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(regularization_losses
)trainable_variables
*	variables
+	keras_api
*r&call_and_return_all_conditional_losses
s__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ō

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
*t&call_and_return_all_conditional_losses
u__call__"Ļ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
“	
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
*v&call_and_return_all_conditional_losses
w__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ó

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": "True", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
 "
trackable_list_wrapper

0
1
2
3
4
5
$6
%7
,8
-9
310
411
;12
<13"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
$8
%9
&10
'11
,12
-13
314
415
516
617
;18
<19"
trackable_list_wrapper
Ź
	regularization_losses
Alayer_metrics
Bmetrics
Cnon_trainable_variables

trainable_variables
	variables
Dlayer_regularization_losses

Elayers
j__call__
k_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
!:2dense_50/kernel
:2dense_50/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
Flayer_metrics
Gmetrics
Hnon_trainable_variables
trainable_variables
	variables
Ilayer_regularization_losses

Jlayers
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_63/gamma
):'2batch_normalization_63/beta
2:0 (2"batch_normalization_63/moving_mean
6:4 (2&batch_normalization_63/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
­
regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
trainable_variables
	variables
Nlayer_regularization_losses

Olayers
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_51/kernel
:
2dense_51/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
 trainable_variables
!	variables
Slayer_regularization_losses

Tlayers
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_64/gamma
):'
2batch_normalization_64/beta
2:0
 (2"batch_normalization_64/moving_mean
6:4
 (2&batch_normalization_64/moving_variance
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
­
(regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
)trainable_variables
*	variables
Xlayer_regularization_losses

Ylayers
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_52/kernel
:
2dense_52/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
­
.regularization_losses
Zlayer_metrics
[metrics
\non_trainable_variables
/trainable_variables
0	variables
]layer_regularization_losses

^layers
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_65/gamma
):'
2batch_normalization_65/beta
2:0
 (2"batch_normalization_65/moving_mean
6:4
 (2&batch_normalization_65/moving_variance
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
­
7regularization_losses
_layer_metrics
`metrics
anon_trainable_variables
8trainable_variables
9	variables
blayer_regularization_losses

clayers
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_53/kernel
:2dense_53/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
­
=regularization_losses
dlayer_metrics
emetrics
fnon_trainable_variables
>trainable_variables
?	variables
glayer_regularization_losses

hlayers
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
&2
'3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
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
0
1"
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
50
61"
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
ā2ß
E__inference_Decoder_layer_call_and_return_conditional_losses_41914974
E__inference_Decoder_layer_call_and_return_conditional_losses_41914894
E__inference_Decoder_layer_call_and_return_conditional_losses_41914526
E__inference_Decoder_layer_call_and_return_conditional_losses_41914475Ą
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
ö2ó
*__inference_Decoder_layer_call_fn_41914719
*__inference_Decoder_layer_call_fn_41914623
*__inference_Decoder_layer_call_fn_41915064
*__inference_Decoder_layer_call_fn_41915019Ą
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
ā2ß
#__inference__wrapped_model_41913837·
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
annotationsŖ *'¢$
"
input_20’’’’’’’’’
š2ķ
F__inference_dense_50_layer_call_and_return_conditional_losses_41915075¢
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
Õ2Ņ
+__inference_dense_50_layer_call_fn_41915084¢
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
ę2ć
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915140
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915120“
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
°2­
9__inference_batch_normalization_63_layer_call_fn_41915166
9__inference_batch_normalization_63_layer_call_fn_41915153“
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
š2ķ
F__inference_dense_51_layer_call_and_return_conditional_losses_41915177¢
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
Õ2Ņ
+__inference_dense_51_layer_call_fn_41915186¢
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
ę2ć
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915242
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915222“
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
°2­
9__inference_batch_normalization_64_layer_call_fn_41915268
9__inference_batch_normalization_64_layer_call_fn_41915255“
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
š2ķ
F__inference_dense_52_layer_call_and_return_conditional_losses_41915279¢
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
Õ2Ņ
+__inference_dense_52_layer_call_fn_41915288¢
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
ę2ć
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915324
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915344“
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
°2­
9__inference_batch_normalization_65_layer_call_fn_41915357
9__inference_batch_normalization_65_layer_call_fn_41915370“
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
š2ķ
F__inference_dense_53_layer_call_and_return_conditional_losses_41915381¢
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
Õ2Ņ
+__inference_dense_53_layer_call_fn_41915390¢
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
ĪBĖ
&__inference_signature_wrapper_41914766input_20"
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
 Į
E__inference_Decoder_layer_call_and_return_conditional_losses_41914475x&'$%,-5634;<9¢6
/¢,
"
input_20’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 Į
E__inference_Decoder_layer_call_and_return_conditional_losses_41914526x'$&%,-6354;<9¢6
/¢,
"
input_20’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 æ
E__inference_Decoder_layer_call_and_return_conditional_losses_41914894v&'$%,-5634;<7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 æ
E__inference_Decoder_layer_call_and_return_conditional_losses_41914974v'$&%,-6354;<7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
*__inference_Decoder_layer_call_fn_41914623k&'$%,-5634;<9¢6
/¢,
"
input_20’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
*__inference_Decoder_layer_call_fn_41914719k'$&%,-6354;<9¢6
/¢,
"
input_20’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
*__inference_Decoder_layer_call_fn_41915019i&'$%,-5634;<7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
*__inference_Decoder_layer_call_fn_41915064i'$&%,-6354;<7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’„
#__inference__wrapped_model_41913837~'$&%,-6354;<1¢.
'¢$
"
input_20’’’’’’’’’
Ŗ "3Ŗ0
.
dense_53"
dense_53’’’’’’’’’ŗ
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915120b3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ŗ
T__inference_batch_normalization_63_layer_call_and_return_conditional_losses_41915140b3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 
9__inference_batch_normalization_63_layer_call_fn_41915153U3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
9__inference_batch_normalization_63_layer_call_fn_41915166U3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’ŗ
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915222b&'$%3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ŗ
T__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41915242b'$&%3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
9__inference_batch_normalization_64_layer_call_fn_41915255U&'$%3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

9__inference_batch_normalization_64_layer_call_fn_41915268U'$&%3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
ŗ
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915324b56343¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ŗ
T__inference_batch_normalization_65_layer_call_and_return_conditional_losses_41915344b63543¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
9__inference_batch_normalization_65_layer_call_fn_41915357U56343¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

9__inference_batch_normalization_65_layer_call_fn_41915370U63543¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
¦
F__inference_dense_50_layer_call_and_return_conditional_losses_41915075\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_50_layer_call_fn_41915084O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¦
F__inference_dense_51_layer_call_and_return_conditional_losses_41915177\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’

 ~
+__inference_dense_51_layer_call_fn_41915186O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
¦
F__inference_dense_52_layer_call_and_return_conditional_losses_41915279\,-/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 ~
+__inference_dense_52_layer_call_fn_41915288O,-/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
¦
F__inference_dense_53_layer_call_and_return_conditional_losses_41915381\;</¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_53_layer_call_fn_41915390O;</¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’µ
&__inference_signature_wrapper_41914766'$&%,-6354;<=¢:
¢ 
3Ŗ0
.
input_20"
input_20’’’’’’’’’"3Ŗ0
.
dense_53"
dense_53’’’’’’’’’