Ò¯
Ç
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
;
Elu
features"T
activations"T"
Ttype:
2
,
Exp
x"T
y"T"
Ttype:

2
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ð

z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:(*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:(*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:(*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:(*
dtype0

NoOpNoOp
¦+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*á*
value×*BÔ* BÍ*

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api

)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
v
0
1
2
3
4
5
#6
$7
*8
+9
,10
-11
612
713
<14
=15
V
0
1
2
3
#4
$5
*6
+7
68
79
<10
=11
 
­
Fnon_trainable_variables
	variables

Glayers
trainable_variables
Hlayer_regularization_losses
regularization_losses
Ilayer_metrics
Jmetrics
 
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Knon_trainable_variables
regularization_losses
	variables

Llayers
trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
­
Pnon_trainable_variables
regularization_losses
	variables

Qlayers
trainable_variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
 
 
 
­
Unon_trainable_variables
regularization_losses
 	variables

Vlayers
!trainable_variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
­
Znon_trainable_variables
%regularization_losses
&	variables

[layers
'trainable_variables
\layer_regularization_losses
]metrics
^layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
,2
-3

*0
+1
­
_non_trainable_variables
.regularization_losses
/	variables

`layers
0trainable_variables
alayer_regularization_losses
bmetrics
clayer_metrics
 
 
 
­
dnon_trainable_variables
2regularization_losses
3	variables

elayers
4trainable_variables
flayer_regularization_losses
gmetrics
hlayer_metrics
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
­
inon_trainable_variables
8regularization_losses
9	variables

jlayers
:trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
­
nnon_trainable_variables
>regularization_losses
?	variables

olayers
@trainable_variables
player_regularization_losses
qmetrics
rlayer_metrics
 
 
 
­
snon_trainable_variables
Bregularization_losses
C	variables

tlayers
Dtrainable_variables
ulayer_regularization_losses
vmetrics
wlayer_metrics

0
1
,2
-3
F
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
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
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
 
 
 
 
 
 
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
serving_default_input_4Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_16/kerneldense_16/bias&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betadense_17/kerneldense_17/bias&batch_normalization_13/moving_variancebatch_normalization_13/gamma"batch_normalization_13/moving_meanbatch_normalization_13/betadense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_13667136
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_13667775
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancedense_17/kerneldense_17/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancedense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_13667833¹

×

T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13666480

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
s
F__inference_lambda_1_layer_call_and_return_conditional_losses_13666871

inputs
inputs_1
identityD
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
 *  ?2
random_normal/stddevÎ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2£û«2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
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
:ÿÿÿÿÿÿÿÿÿ(2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

E
)__inference_elu_13_layer_call_fn_13667620

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_13_layer_call_and_return_conditional_losses_136667772
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
¬
9__inference_batch_normalization_12_layer_call_fn_13667498

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
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_136664802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·G
¡	
$__inference__traced_restore_13667833
file_prefix$
 assignvariableop_dense_16_kernel$
 assignvariableop_1_dense_16_bias3
/assignvariableop_2_batch_normalization_12_gamma2
.assignvariableop_3_batch_normalization_12_beta9
5assignvariableop_4_batch_normalization_12_moving_mean=
9assignvariableop_5_batch_normalization_12_moving_variance&
"assignvariableop_6_dense_17_kernel$
 assignvariableop_7_dense_17_bias3
/assignvariableop_8_batch_normalization_13_gamma2
.assignvariableop_9_batch_normalization_13_beta:
6assignvariableop_10_batch_normalization_13_moving_mean>
:assignvariableop_11_batch_normalization_13_moving_variance'
#assignvariableop_12_dense_18_kernel%
!assignvariableop_13_dense_18_bias'
#assignvariableop_14_dense_19_kernel%
!assignvariableop_15_dense_19_bias
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9µ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Á
value·B´B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
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

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_12_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_12_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4º
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_12_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¾
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_12_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_13_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_13_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¾
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_13_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Â
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_13_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_18_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_18_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_19_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_19_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16±
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
ã
u
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667676
inputs_0
inputs_1
identityF
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
 *  ?2
random_normal/stddevÍ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2ÆË52$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
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
:ÿÿÿÿÿÿÿÿÿ(2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1
®
`
D__inference_elu_13_layer_call_and_return_conditional_losses_13667615

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13666447

inputs
assignmovingavg_13666422
assignmovingavg_1_13666428)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13666422*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13666422*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13666422*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13666422*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13666422AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13666422*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13666428*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13666428*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13666428*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13666428*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13666428AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13666428*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13666620

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667564

inputs
assignmovingavg_13667539
assignmovingavg_1_13667545)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13667539*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13667539*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13667539*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13667539*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13667539AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13667539*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13667545*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13667545*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13667545*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13667545*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13667545AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13667545*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
0
Ö
E__inference_model_1_layer_call_and_return_conditional_losses_13666932
input_4
dense_16_13666890
dense_16_13666892#
batch_normalization_12_13666895#
batch_normalization_12_13666897#
batch_normalization_12_13666899#
batch_normalization_12_13666901
dense_17_13666905
dense_17_13666907#
batch_normalization_13_13666910#
batch_normalization_13_13666912#
batch_normalization_13_13666914#
batch_normalization_13_13666916
dense_18_13666920
dense_18_13666922
dense_19_13666925
dense_19_13666927
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ lambda_1/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_16_13666890dense_16_13666892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_136666462"
 dense_16/StatefulPartitionedCallÉ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_13666895batch_normalization_12_13666897batch_normalization_12_13666899batch_normalization_12_13666901*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1366648020
.batch_normalization_12/StatefulPartitionedCall
elu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_12_layer_call_and_return_conditional_losses_136667022
elu_12/PartitionedCall³
 dense_17/StatefulPartitionedCallStatefulPartitionedCallelu_12/PartitionedCall:output:0dense_17_13666905dense_17_13666907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_136667212"
 dense_17/StatefulPartitionedCallÉ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_13666910batch_normalization_13_13666912batch_normalization_13_13666914batch_normalization_13_13666916*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1366662020
.batch_normalization_13/StatefulPartitionedCall
elu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_13_layer_call_and_return_conditional_losses_136667772
elu_13/PartitionedCall³
 dense_18/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_18_13666920dense_18_13666922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_136667962"
 dense_18/StatefulPartitionedCall³
 dense_19/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_19_13666925dense_19_13666927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_136668232"
 dense_19/StatefulPartitionedCall½
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668712"
 lambda_1/StatefulPartitionedCall
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Ú
s
F__inference_lambda_1_layer_call_and_return_conditional_losses_13666855

inputs
inputs_1
identityD
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
 *  ?2
random_normal/stddevÎ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2Ù´·2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
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
:ÿÿÿÿÿÿÿÿÿ(2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
×

T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667584

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
t
+__inference_lambda_1_layer_call_fn_13667704
inputs_0
inputs_1
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1
½
¬
9__inference_batch_normalization_13_layer_call_fn_13667610

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
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_136666202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ	
ß
F__inference_dense_16_layer_call_and_return_conditional_losses_13667407

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
á

+__inference_dense_16_layer_call_fn_13667416

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
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_136666462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

E
)__inference_elu_12_layer_call_fn_13667508

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_12_layer_call_and_return_conditional_losses_136667022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
`
D__inference_elu_12_layer_call_and_return_conditional_losses_13666702

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

×
*__inference_model_1_layer_call_fn_13667015
input_4
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
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ(*.
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_136669802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
¿
à
E__inference_model_1_layer_call_and_return_conditional_losses_13667245

inputs+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource3
/batch_normalization_12_assignmovingavg_136671545
1batch_normalization_12_assignmovingavg_1_13667160@
<batch_normalization_12_batchnorm_mul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource3
/batch_normalization_13_assignmovingavg_136671945
1batch_normalization_13_assignmovingavg_1_13667200@
<batch_normalization_13_batchnorm_mul_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity¢:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_12/AssignMovingAvg/ReadVariableOp¢<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_12/batchnorm/ReadVariableOp¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_13/AssignMovingAvg/ReadVariableOp¢<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAdds
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/Tanh¸
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_12/moments/mean/reduction_indicesß
#batch_normalization_12/moments/meanMeandense_16/Tanh:y:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_12/moments/meanÁ
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_12/moments/StopGradientô
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencedense_16/Tanh:y:04batch_normalization_12/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_12/moments/SquaredDifferenceÀ
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_12/moments/variance/reduction_indices
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_12/moments/varianceÅ
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_12/moments/SqueezeÍ
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1
,batch_normalization_12/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg/13667154*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_12/AssignMovingAvg/decayÚ
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_12_assignmovingavg_13667154*
_output_shapes
:*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg/13667154*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/subÝ
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg/13667154*
_output_shapes
:2,
*batch_normalization_12/AssignMovingAvg/mul½
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_12_assignmovingavg_13667154.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_12/AssignMovingAvg/13667154*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_12/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_12/AssignMovingAvg_1/13667160*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_12/AssignMovingAvg_1/decayà
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_12_assignmovingavg_1_13667160*
_output_shapes
:*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_12/AssignMovingAvg_1/13667160*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/subç
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_12/AssignMovingAvg_1/13667160*
_output_shapes
:2.
,batch_normalization_12/AssignMovingAvg_1/mulÉ
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_12_assignmovingavg_1_136671600batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_12/AssignMovingAvg_1/13667160*
_output_shapes
 *
dtype02>
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_12/batchnorm/add/yÞ
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add¨
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrtã
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpá
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mulÆ
&batch_normalization_12/batchnorm/mul_1Muldense_16/Tanh:y:0(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_12/batchnorm/mul_1×
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2×
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOpÝ
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/subá
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_12/batchnorm/add_1}

elu_12/EluElu*batch_normalization_12/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

elu_12/Elu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp 
dense_17/MatMulMatMulelu_12/Elu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/BiasAdds
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/Tanh¸
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_13/moments/mean/reduction_indicesß
#batch_normalization_13/moments/meanMeandense_17/Tanh:y:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_13/moments/meanÁ
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_13/moments/StopGradientô
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedense_17/Tanh:y:04batch_normalization_13/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_13/moments/SquaredDifferenceÀ
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_13/moments/variance/reduction_indices
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_13/moments/varianceÅ
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_13/moments/SqueezeÍ
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1
,batch_normalization_13/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg/13667194*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_13/AssignMovingAvg/decayÚ
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_13_assignmovingavg_13667194*
_output_shapes
:*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg/13667194*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/subÝ
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg/13667194*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/mul½
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_13_assignmovingavg_13667194.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_13/AssignMovingAvg/13667194*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_13/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_13/AssignMovingAvg_1/13667200*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_13/AssignMovingAvg_1/decayà
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_13_assignmovingavg_1_13667200*
_output_shapes
:*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_13/AssignMovingAvg_1/13667200*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/subç
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_13/AssignMovingAvg_1/13667200*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/mulÉ
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_13_assignmovingavg_1_136672000batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_13/AssignMovingAvg_1/13667200*
_output_shapes
 *
dtype02>
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_13/batchnorm/add/yÞ
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrtã
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpá
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mulÆ
&batch_normalization_13/batchnorm/mul_1Muldense_17/Tanh:y:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_13/batchnorm/mul_1×
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2×
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOpÝ
$batch_normalization_13/batchnorm/subSub7batch_normalization_13/batchnorm/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/subá
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_13/batchnorm/add_1}

elu_13/EluElu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

elu_13/Elu¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_18/MatMul/ReadVariableOp 
dense_18/MatMulMatMulelu_13/Elu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/BiasAdds
dense_18/TanhTanhdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/Tanh¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_19/MatMul/ReadVariableOp 
dense_19/MatMulMatMulelu_13/Elu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/BiasAdds
dense_19/TanhTanhdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/Tanha
lambda_1/ShapeShapedense_18/Tanh:y:0*
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
lambda_1/random_normal/mean
lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lambda_1/random_normal/stddevè
+lambda_1/random_normal/RandomStandardNormalRandomStandardNormallambda_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2±2-
+lambda_1/random_normal/RandomStandardNormalÏ
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/random_normal/mul¯
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/random_normalm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y
lambda_1/truedivRealDivdense_19/Tanh:y:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/Exp
lambda_1/mulMullambda_1/random_normal:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/mul|
lambda_1/addAddV2dense_18/Tanh:y:0lambda_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/add
IdentityIdentitylambda_1/add:z:0;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_12/AssignMovingAvg/ReadVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_13/AssignMovingAvg/ReadVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp4^batch_normalization_13/batchnorm/mul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2x
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2x
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ	
ß
F__inference_dense_19_layer_call_and_return_conditional_losses_13667651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

Ö
*__inference_model_1_layer_call_fn_13667396

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
identity¢StatefulPartitionedCall°
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
:ÿÿÿÿÿÿÿÿÿ(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_136670622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

+__inference_dense_18_layer_call_fn_13667640

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
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_136667962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
¬
9__inference_batch_normalization_12_layer_call_fn_13667485

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
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_136664472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
u
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667692
inputs_0
inputs_1
identityF
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
 *  ?2
random_normal/stddevÎ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2Çîß2$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
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
:ÿÿÿÿÿÿÿÿÿ(2	
truedivP
ExpExptruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Exp_
mulMulrandom_normal:z:0Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1
§
t
+__inference_lambda_1_layer_call_fn_13667698
inputs_0
inputs_1
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/1
0
Ö
E__inference_model_1_layer_call_and_return_conditional_losses_13666887
input_4
dense_16_13666657
dense_16_13666659#
batch_normalization_12_13666688#
batch_normalization_12_13666690#
batch_normalization_12_13666692#
batch_normalization_12_13666694
dense_17_13666732
dense_17_13666734#
batch_normalization_13_13666763#
batch_normalization_13_13666765#
batch_normalization_13_13666767#
batch_normalization_13_13666769
dense_18_13666807
dense_18_13666809
dense_19_13666834
dense_19_13666836
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ lambda_1/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_16_13666657dense_16_13666659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_136666462"
 dense_16/StatefulPartitionedCallÇ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_13666688batch_normalization_12_13666690batch_normalization_12_13666692batch_normalization_12_13666694*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1366644720
.batch_normalization_12/StatefulPartitionedCall
elu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_12_layer_call_and_return_conditional_losses_136667022
elu_12/PartitionedCall³
 dense_17/StatefulPartitionedCallStatefulPartitionedCallelu_12/PartitionedCall:output:0dense_17_13666732dense_17_13666734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_136667212"
 dense_17/StatefulPartitionedCallÇ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_13666763batch_normalization_13_13666765batch_normalization_13_13666767batch_normalization_13_13666769*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1366658720
.batch_normalization_13/StatefulPartitionedCall
elu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_13_layer_call_and_return_conditional_losses_136667772
elu_13/PartitionedCall³
 dense_18/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_18_13666807dense_18_13666809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_136667962"
 dense_18/StatefulPartitionedCall³
 dense_19/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_19_13666834dense_19_13666836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_136668232"
 dense_19/StatefulPartitionedCall½
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668552"
 lambda_1/StatefulPartitionedCall
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
æ	
ß
F__inference_dense_17_layer_call_and_return_conditional_losses_13667519

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13666587

inputs
assignmovingavg_13666562
assignmovingavg_1_13666568)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13666562*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13666562*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13666562*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13666562*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13666562AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13666562*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13666568*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13666568*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13666568*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13666568*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13666568AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13666568*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

×
*__inference_model_1_layer_call_fn_13667097
input_4
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
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_136670622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Ñn
ü
E__inference_model_1_layer_call_and_return_conditional_losses_13667322

inputs+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource@
<batch_normalization_12_batchnorm_mul_readvariableop_resource>
:batch_normalization_12_batchnorm_readvariableop_1_resource>
:batch_normalization_12_batchnorm_readvariableop_2_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource@
<batch_normalization_13_batchnorm_mul_readvariableop_resource>
:batch_normalization_13_batchnorm_readvariableop_1_resource>
:batch_normalization_13_batchnorm_readvariableop_2_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity¢/batch_normalization_12/batchnorm/ReadVariableOp¢1batch_normalization_12/batchnorm/ReadVariableOp_1¢1batch_normalization_12/batchnorm/ReadVariableOp_2¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢1batch_normalization_13/batchnorm/ReadVariableOp_1¢1batch_normalization_13/batchnorm/ReadVariableOp_2¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAdds
dense_16/TanhTanhdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/Tanh×
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_12/batchnorm/add/yä
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/add¨
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/Rsqrtã
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpá
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/mulÆ
&batch_normalization_12/batchnorm/mul_1Muldense_16/Tanh:y:0(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_12/batchnorm/mul_1Ý
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1á
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_12/batchnorm/mul_2Ý
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2ß
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_12/batchnorm/subá
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_12/batchnorm/add_1}

elu_12/EluElu*batch_normalization_12/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

elu_12/Elu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp 
dense_17/MatMulMatMulelu_12/Elu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/BiasAdds
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/Tanh×
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOp
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_13/batchnorm/add/yä
$batch_normalization_13/batchnorm/addAddV27batch_normalization_13/batchnorm/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrtã
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpá
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mulÆ
&batch_normalization_13/batchnorm/mul_1Muldense_17/Tanh:y:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_13/batchnorm/mul_1Ý
1batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_1á
&batch_normalization_13/batchnorm/mul_2Mul9batch_normalization_13/batchnorm/ReadVariableOp_1:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2Ý
1batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_2ß
$batch_normalization_13/batchnorm/subSub9batch_normalization_13/batchnorm/ReadVariableOp_2:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/subá
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_13/batchnorm/add_1}

elu_13/EluElu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

elu_13/Elu¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_18/MatMul/ReadVariableOp 
dense_18/MatMulMatMulelu_13/Elu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/MatMul§
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_18/BiasAdd/ReadVariableOp¥
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/BiasAdds
dense_18/TanhTanhdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_18/Tanh¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_19/MatMul/ReadVariableOp 
dense_19/MatMulMatMulelu_13/Elu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/BiasAdds
dense_19/TanhTanhdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
dense_19/Tanha
lambda_1/ShapeShapedense_18/Tanh:y:0*
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
lambda_1/random_normal/mean
lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lambda_1/random_normal/stddevè
+lambda_1/random_normal/RandomStandardNormalRandomStandardNormallambda_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2ø£2-
+lambda_1/random_normal/RandomStandardNormalÏ
lambda_1/random_normal/mulMul4lambda_1/random_normal/RandomStandardNormal:output:0&lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/random_normal/mul¯
lambda_1/random_normalAddlambda_1/random_normal/mul:z:0$lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/random_normalm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda_1/truediv/y
lambda_1/truedivRealDivdense_19/Tanh:y:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/truedivk
lambda_1/ExpExplambda_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/Exp
lambda_1/mulMullambda_1/random_normal:z:0lambda_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/mul|
lambda_1/addAddV2dense_18/Tanh:y:0lambda_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
lambda_1/add
IdentityIdentitylambda_1/add:z:00^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp2^batch_normalization_13/batchnorm/ReadVariableOp_12^batch_normalization_13/batchnorm/ReadVariableOp_24^batch_normalization_13/batchnorm/mul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2f
1batch_normalization_13/batchnorm/ReadVariableOp_11batch_normalization_13/batchnorm/ReadVariableOp_12f
1batch_normalization_13/batchnorm/ReadVariableOp_21batch_normalization_13/batchnorm/ReadVariableOp_22j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
Û
#__inference__wrapped_model_13666351
input_43
/model_1_dense_16_matmul_readvariableop_resource4
0model_1_dense_16_biasadd_readvariableop_resourceD
@model_1_batch_normalization_12_batchnorm_readvariableop_resourceH
Dmodel_1_batch_normalization_12_batchnorm_mul_readvariableop_resourceF
Bmodel_1_batch_normalization_12_batchnorm_readvariableop_1_resourceF
Bmodel_1_batch_normalization_12_batchnorm_readvariableop_2_resource3
/model_1_dense_17_matmul_readvariableop_resource4
0model_1_dense_17_biasadd_readvariableop_resourceD
@model_1_batch_normalization_13_batchnorm_readvariableop_resourceH
Dmodel_1_batch_normalization_13_batchnorm_mul_readvariableop_resourceF
Bmodel_1_batch_normalization_13_batchnorm_readvariableop_1_resourceF
Bmodel_1_batch_normalization_13_batchnorm_readvariableop_2_resource3
/model_1_dense_18_matmul_readvariableop_resource4
0model_1_dense_18_biasadd_readvariableop_resource3
/model_1_dense_19_matmul_readvariableop_resource4
0model_1_dense_19_biasadd_readvariableop_resource
identity¢7model_1/batch_normalization_12/batchnorm/ReadVariableOp¢9model_1/batch_normalization_12/batchnorm/ReadVariableOp_1¢9model_1/batch_normalization_12/batchnorm/ReadVariableOp_2¢;model_1/batch_normalization_12/batchnorm/mul/ReadVariableOp¢7model_1/batch_normalization_13/batchnorm/ReadVariableOp¢9model_1/batch_normalization_13/batchnorm/ReadVariableOp_1¢9model_1/batch_normalization_13/batchnorm/ReadVariableOp_2¢;model_1/batch_normalization_13/batchnorm/mul/ReadVariableOp¢'model_1/dense_16/BiasAdd/ReadVariableOp¢&model_1/dense_16/MatMul/ReadVariableOp¢'model_1/dense_17/BiasAdd/ReadVariableOp¢&model_1/dense_17/MatMul/ReadVariableOp¢'model_1/dense_18/BiasAdd/ReadVariableOp¢&model_1/dense_18/MatMul/ReadVariableOp¢'model_1/dense_19/BiasAdd/ReadVariableOp¢&model_1/dense_19/MatMul/ReadVariableOpÀ
&model_1/dense_16/MatMul/ReadVariableOpReadVariableOp/model_1_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_1/dense_16/MatMul/ReadVariableOp§
model_1/dense_16/MatMulMatMulinput_4.model_1/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_16/MatMul¿
'model_1/dense_16/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_16/BiasAdd/ReadVariableOpÅ
model_1/dense_16/BiasAddBiasAdd!model_1/dense_16/MatMul:product:0/model_1/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_16/BiasAdd
model_1/dense_16/TanhTanh!model_1/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_16/Tanhï
7model_1/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp@model_1_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7model_1/batch_normalization_12/batchnorm/ReadVariableOp¥
.model_1/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.model_1/batch_normalization_12/batchnorm/add/y
,model_1/batch_normalization_12/batchnorm/addAddV2?model_1/batch_normalization_12/batchnorm/ReadVariableOp:value:07model_1/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_12/batchnorm/addÀ
.model_1/batch_normalization_12/batchnorm/RsqrtRsqrt0model_1/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:20
.model_1/batch_normalization_12/batchnorm/Rsqrtû
;model_1/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_1_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;model_1/batch_normalization_12/batchnorm/mul/ReadVariableOp
,model_1/batch_normalization_12/batchnorm/mulMul2model_1/batch_normalization_12/batchnorm/Rsqrt:y:0Cmodel_1/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_12/batchnorm/mulæ
.model_1/batch_normalization_12/batchnorm/mul_1Mulmodel_1/dense_16/Tanh:y:00model_1/batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_1/batch_normalization_12/batchnorm/mul_1õ
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_1_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_1
.model_1/batch_normalization_12/batchnorm/mul_2MulAmodel_1/batch_normalization_12/batchnorm/ReadVariableOp_1:value:00model_1/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.model_1/batch_normalization_12/batchnorm/mul_2õ
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_1_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_2ÿ
,model_1/batch_normalization_12/batchnorm/subSubAmodel_1/batch_normalization_12/batchnorm/ReadVariableOp_2:value:02model_1/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_12/batchnorm/sub
.model_1/batch_normalization_12/batchnorm/add_1AddV22model_1/batch_normalization_12/batchnorm/mul_1:z:00model_1/batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_1/batch_normalization_12/batchnorm/add_1
model_1/elu_12/EluElu2model_1/batch_normalization_12/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/elu_12/EluÀ
&model_1/dense_17/MatMul/ReadVariableOpReadVariableOp/model_1_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_1/dense_17/MatMul/ReadVariableOpÀ
model_1/dense_17/MatMulMatMul model_1/elu_12/Elu:activations:0.model_1/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_17/MatMul¿
'model_1/dense_17/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_17/BiasAdd/ReadVariableOpÅ
model_1/dense_17/BiasAddBiasAdd!model_1/dense_17/MatMul:product:0/model_1/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_17/BiasAdd
model_1/dense_17/TanhTanh!model_1/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense_17/Tanhï
7model_1/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp@model_1_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7model_1/batch_normalization_13/batchnorm/ReadVariableOp¥
.model_1/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.model_1/batch_normalization_13/batchnorm/add/y
,model_1/batch_normalization_13/batchnorm/addAddV2?model_1/batch_normalization_13/batchnorm/ReadVariableOp:value:07model_1/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_13/batchnorm/addÀ
.model_1/batch_normalization_13/batchnorm/RsqrtRsqrt0model_1/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:20
.model_1/batch_normalization_13/batchnorm/Rsqrtû
;model_1/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_1_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;model_1/batch_normalization_13/batchnorm/mul/ReadVariableOp
,model_1/batch_normalization_13/batchnorm/mulMul2model_1/batch_normalization_13/batchnorm/Rsqrt:y:0Cmodel_1/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_13/batchnorm/mulæ
.model_1/batch_normalization_13/batchnorm/mul_1Mulmodel_1/dense_17/Tanh:y:00model_1/batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_1/batch_normalization_13/batchnorm/mul_1õ
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_1_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_1
.model_1/batch_normalization_13/batchnorm/mul_2MulAmodel_1/batch_normalization_13/batchnorm/ReadVariableOp_1:value:00model_1/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.model_1/batch_normalization_13/batchnorm/mul_2õ
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_1_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_2ÿ
,model_1/batch_normalization_13/batchnorm/subSubAmodel_1/batch_normalization_13/batchnorm/ReadVariableOp_2:value:02model_1/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,model_1/batch_normalization_13/batchnorm/sub
.model_1/batch_normalization_13/batchnorm/add_1AddV22model_1/batch_normalization_13/batchnorm/mul_1:z:00model_1/batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_1/batch_normalization_13/batchnorm/add_1
model_1/elu_13/EluElu2model_1/batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/elu_13/EluÀ
&model_1/dense_18/MatMul/ReadVariableOpReadVariableOp/model_1_dense_18_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02(
&model_1/dense_18/MatMul/ReadVariableOpÀ
model_1/dense_18/MatMulMatMul model_1/elu_13/Elu:activations:0.model_1/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_18/MatMul¿
'model_1/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_18_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02)
'model_1/dense_18/BiasAdd/ReadVariableOpÅ
model_1/dense_18/BiasAddBiasAdd!model_1/dense_18/MatMul:product:0/model_1/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_18/BiasAdd
model_1/dense_18/TanhTanh!model_1/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_18/TanhÀ
&model_1/dense_19/MatMul/ReadVariableOpReadVariableOp/model_1_dense_19_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02(
&model_1/dense_19/MatMul/ReadVariableOpÀ
model_1/dense_19/MatMulMatMul model_1/elu_13/Elu:activations:0.model_1/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_19/MatMul¿
'model_1/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_19_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02)
'model_1/dense_19/BiasAdd/ReadVariableOpÅ
model_1/dense_19/BiasAddBiasAdd!model_1/dense_19/MatMul:product:0/model_1/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_19/BiasAdd
model_1/dense_19/TanhTanh!model_1/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/dense_19/Tanhy
model_1/lambda_1/ShapeShapemodel_1/dense_18/Tanh:y:0*
T0*
_output_shapes
:2
model_1/lambda_1/Shape
#model_1/lambda_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model_1/lambda_1/random_normal/mean
%model_1/lambda_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%model_1/lambda_1/random_normal/stddev
3model_1/lambda_1/random_normal/RandomStandardNormalRandomStandardNormalmodel_1/lambda_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
seed±ÿå)*
seed2½Þ25
3model_1/lambda_1/random_normal/RandomStandardNormalï
"model_1/lambda_1/random_normal/mulMul<model_1/lambda_1/random_normal/RandomStandardNormal:output:0.model_1/lambda_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2$
"model_1/lambda_1/random_normal/mulÏ
model_1/lambda_1/random_normalAdd&model_1/lambda_1/random_normal/mul:z:0,model_1/lambda_1/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2 
model_1/lambda_1/random_normal}
model_1/lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/lambda_1/truediv/y±
model_1/lambda_1/truedivRealDivmodel_1/dense_19/Tanh:y:0#model_1/lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/lambda_1/truediv
model_1/lambda_1/ExpExpmodel_1/lambda_1/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/lambda_1/Exp£
model_1/lambda_1/mulMul"model_1/lambda_1/random_normal:z:0model_1/lambda_1/Exp:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/lambda_1/mul
model_1/lambda_1/addAddV2model_1/dense_18/Tanh:y:0model_1/lambda_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model_1/lambda_1/add
IdentityIdentitymodel_1/lambda_1/add:z:08^model_1/batch_normalization_12/batchnorm/ReadVariableOp:^model_1/batch_normalization_12/batchnorm/ReadVariableOp_1:^model_1/batch_normalization_12/batchnorm/ReadVariableOp_2<^model_1/batch_normalization_12/batchnorm/mul/ReadVariableOp8^model_1/batch_normalization_13/batchnorm/ReadVariableOp:^model_1/batch_normalization_13/batchnorm/ReadVariableOp_1:^model_1/batch_normalization_13/batchnorm/ReadVariableOp_2<^model_1/batch_normalization_13/batchnorm/mul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2r
7model_1/batch_normalization_12/batchnorm/ReadVariableOp7model_1/batch_normalization_12/batchnorm/ReadVariableOp2v
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_19model_1/batch_normalization_12/batchnorm/ReadVariableOp_12v
9model_1/batch_normalization_12/batchnorm/ReadVariableOp_29model_1/batch_normalization_12/batchnorm/ReadVariableOp_22z
;model_1/batch_normalization_12/batchnorm/mul/ReadVariableOp;model_1/batch_normalization_12/batchnorm/mul/ReadVariableOp2r
7model_1/batch_normalization_13/batchnorm/ReadVariableOp7model_1/batch_normalization_13/batchnorm/ReadVariableOp2v
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_19model_1/batch_normalization_13/batchnorm/ReadVariableOp_12v
9model_1/batch_normalization_13/batchnorm/ReadVariableOp_29model_1/batch_normalization_13/batchnorm/ReadVariableOp_22z
;model_1/batch_normalization_13/batchnorm/mul/ReadVariableOp;model_1/batch_normalization_13/batchnorm/mul/ReadVariableOp2R
'model_1/dense_16/BiasAdd/ReadVariableOp'model_1/dense_16/BiasAdd/ReadVariableOp2P
&model_1/dense_16/MatMul/ReadVariableOp&model_1/dense_16/MatMul/ReadVariableOp2R
'model_1/dense_17/BiasAdd/ReadVariableOp'model_1/dense_17/BiasAdd/ReadVariableOp2P
&model_1/dense_17/MatMul/ReadVariableOp&model_1/dense_17/MatMul/ReadVariableOp2R
'model_1/dense_18/BiasAdd/ReadVariableOp'model_1/dense_18/BiasAdd/ReadVariableOp2P
&model_1/dense_18/MatMul/ReadVariableOp&model_1/dense_18/MatMul/ReadVariableOp2R
'model_1/dense_19/BiasAdd/ReadVariableOp'model_1/dense_19/BiasAdd/ReadVariableOp2P
&model_1/dense_19/MatMul/ReadVariableOp&model_1/dense_19/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
æ	
ß
F__inference_dense_16_layer_call_and_return_conditional_losses_13666646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
æ	
ß
F__inference_dense_17_layer_call_and_return_conditional_losses_13666721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥

Ö
*__inference_model_1_layer_call_fn_13667359

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
identity¢StatefulPartitionedCall¬
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
:ÿÿÿÿÿÿÿÿÿ(*.
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_136669802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ/
Õ
E__inference_model_1_layer_call_and_return_conditional_losses_13666980

inputs
dense_16_13666938
dense_16_13666940#
batch_normalization_12_13666943#
batch_normalization_12_13666945#
batch_normalization_12_13666947#
batch_normalization_12_13666949
dense_17_13666953
dense_17_13666955#
batch_normalization_13_13666958#
batch_normalization_13_13666960#
batch_normalization_13_13666962#
batch_normalization_13_13666964
dense_18_13666968
dense_18_13666970
dense_19_13666973
dense_19_13666975
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ lambda_1/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_13666938dense_16_13666940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_136666462"
 dense_16/StatefulPartitionedCallÇ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_13666943batch_normalization_12_13666945batch_normalization_12_13666947batch_normalization_12_13666949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1366644720
.batch_normalization_12/StatefulPartitionedCall
elu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_12_layer_call_and_return_conditional_losses_136667022
elu_12/PartitionedCall³
 dense_17/StatefulPartitionedCallStatefulPartitionedCallelu_12/PartitionedCall:output:0dense_17_13666953dense_17_13666955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_136667212"
 dense_17/StatefulPartitionedCallÇ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_13666958batch_normalization_13_13666960batch_normalization_13_13666962batch_normalization_13_13666964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1366658720
.batch_normalization_13/StatefulPartitionedCall
elu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_13_layer_call_and_return_conditional_losses_136667772
elu_13/PartitionedCall³
 dense_18/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_18_13666968dense_18_13666970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_136667962"
 dense_18/StatefulPartitionedCall³
 dense_19/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_19_13666973dense_19_13666975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_136668232"
 dense_19/StatefulPartitionedCall½
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668552"
 lambda_1/StatefulPartitionedCall
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
`
D__inference_elu_13_layer_call_and_return_conditional_losses_13666777

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667472

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
`
D__inference_elu_12_layer_call_and_return_conditional_losses_13667503

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
¬
9__inference_batch_normalization_13_layer_call_fn_13667597

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
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_136665872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ	
ß
F__inference_dense_19_layer_call_and_return_conditional_losses_13666823

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ	
ß
F__inference_dense_18_layer_call_and_return_conditional_losses_13666796

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667452

inputs
assignmovingavg_13667427
assignmovingavg_1_13667433)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13667427*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_13667427*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13667427*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/13667427*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_13667427AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/13667427*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13667433*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_13667433*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13667433*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/13667433*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_13667433AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/13667433*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ó
&__inference_signature_wrapper_13667136
input_4
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_136663512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
æ	
ß
F__inference_dense_18_layer_call_and_return_conditional_losses_13667631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö,
ì
!__inference__traced_save_13667775
file_prefix.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop
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
ShardedFilename¯
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Á
value·B´B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*
_input_shapesv
t: :::::::::::::(:(:(:(: 2(
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
:: 

_output_shapes
::$ 

_output_shapes

:: 
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
:: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
:(:

_output_shapes
: 
0
Õ
E__inference_model_1_layer_call_and_return_conditional_losses_13667062

inputs
dense_16_13667020
dense_16_13667022#
batch_normalization_12_13667025#
batch_normalization_12_13667027#
batch_normalization_12_13667029#
batch_normalization_12_13667031
dense_17_13667035
dense_17_13667037#
batch_normalization_13_13667040#
batch_normalization_13_13667042#
batch_normalization_13_13667044#
batch_normalization_13_13667046
dense_18_13667050
dense_18_13667052
dense_19_13667055
dense_19_13667057
identity¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ lambda_1/StatefulPartitionedCall
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_13667020dense_16_13667022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_136666462"
 dense_16/StatefulPartitionedCallÉ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_12_13667025batch_normalization_12_13667027batch_normalization_12_13667029batch_normalization_12_13667031*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1366648020
.batch_normalization_12/StatefulPartitionedCall
elu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_12_layer_call_and_return_conditional_losses_136667022
elu_12/PartitionedCall³
 dense_17/StatefulPartitionedCallStatefulPartitionedCallelu_12/PartitionedCall:output:0dense_17_13667035dense_17_13667037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_136667212"
 dense_17/StatefulPartitionedCallÉ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_13_13667040batch_normalization_13_13667042batch_normalization_13_13667044batch_normalization_13_13667046*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1366662020
.batch_normalization_13/StatefulPartitionedCall
elu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_elu_13_layer_call_and_return_conditional_losses_136667772
elu_13/PartitionedCall³
 dense_18/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_18_13667050dense_18_13667052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_136667962"
 dense_18/StatefulPartitionedCall³
 dense_19/StatefulPartitionedCallStatefulPartitionedCallelu_13/PartitionedCall:output:0dense_19_13667055dense_19_13667057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_136668232"
 dense_19/StatefulPartitionedCall½
 lambda_1/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_136668712"
 lambda_1/StatefulPartitionedCall
IdentityIdentity)lambda_1/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^lambda_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 lambda_1/StatefulPartitionedCall lambda_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

+__inference_dense_19_layer_call_fn_13667660

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
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_136668232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

+__inference_dense_17_layer_call_fn_13667528

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
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_136667212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_40
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ<
lambda_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ(tensorflow/serving/predict:ª
àR
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
x__call__
*y&call_and_return_all_conditional_losses
z_default_save_signature"O
_tf_keras_networkêN{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "ELU", "config": {"name": "elu_12", "trainable": true, "dtype": "float32", "alpha": 1.0}, "name": "elu_12", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["elu_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "ELU", "config": {"name": "elu_13", "trainable": true, "dtype": "float32", "alpha": 1.0}, "name": "elu_13", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["elu_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["elu_13", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+lsvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA1MDUvbmV0\nd29yazEyLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network12", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_18", 0, 0, {}], ["dense_19", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "ELU", "config": {"name": "elu_12", "trainable": true, "dtype": "float32", "alpha": 1.0}, "name": "elu_12", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["elu_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "ELU", "config": {"name": "elu_13", "trainable": true, "dtype": "float32", "alpha": 1.0}, "name": "elu_13", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["elu_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["elu_13", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+lsvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA1MDUvbmV0\nd29yazEyLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network12", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["dense_18", 0, 0, {}], ["dense_19", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
á

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
´	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"à
_tf_keras_layerÆ{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
»
regularization_losses
 	variables
!trainable_variables
"	keras_api
__call__
+&call_and_return_all_conditional_losses"«
_tf_keras_layer{"class_name": "ELU", "name": "elu_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "elu_12", "trainable": true, "dtype": "float32", "alpha": 1.0}}
ó

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
__call__
+&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
´	
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/	variables
0trainable_variables
1	keras_api
__call__
+&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
¼
2regularization_losses
3	variables
4trainable_variables
5	keras_api
__call__
+&call_and_return_all_conditional_losses"«
_tf_keras_layer{"class_name": "ELU", "name": "elu_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "elu_13", "trainable": true, "dtype": "float32", "alpha": 1.0}}
ò

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ò

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 40, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ì
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
__call__
+&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAHAAAAUwAAAHMuAAAAfABkARkAdACgAXQAoAJ8AGQBGQChAaEBdACg\nA3wAZAIZAGQDGwChARQAFwBTACkETukAAAAA6QEAAADpAgAAACkE2gdiYWNrZW5k2g1yYW5kb21f\nbm9ybWFs2gVzaGFwZdoDZXhwKQHaAXCpAHIJAAAA+lsvVXNlcnMvbGlseWh1YS9PbmVEcml2ZSAt\nIEltcGVyaWFsIENvbGxlZ2UgTG9uZG9uL0lOSEFMRSBDb2RlL0xpbHkvQUFFL0FBRTA1MDUvbmV0\nd29yazEyLnB52gg8bGFtYmRhPicAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "network12", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}

0
1
2
3
4
5
#6
$7
*8
+9
,10
-11
612
713
<14
=15"
trackable_list_wrapper
v
0
1
2
3
#4
$5
*6
+7
68
79
<10
=11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Fnon_trainable_variables
	variables

Glayers
trainable_variables
Hlayer_regularization_losses
regularization_losses
Ilayer_metrics
Jmetrics
x__call__
z_default_save_signature
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
!:2dense_16/kernel
:2dense_16/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Knon_trainable_variables
regularization_losses
	variables

Llayers
trainable_variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Pnon_trainable_variables
regularization_losses
	variables

Qlayers
trainable_variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¯
Unon_trainable_variables
regularization_losses
 	variables

Vlayers
!trainable_variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:2dense_17/kernel
:2dense_17/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
°
Znon_trainable_variables
%regularization_losses
&	variables

[layers
'trainable_variables
\layer_regularization_losses
]metrics
^layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_13/gamma
):'2batch_normalization_13/beta
2:0 (2"batch_normalization_13/moving_mean
6:4 (2&batch_normalization_13/moving_variance
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
°
_non_trainable_variables
.regularization_losses
/	variables

`layers
0trainable_variables
alayer_regularization_losses
bmetrics
clayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
dnon_trainable_variables
2regularization_losses
3	variables

elayers
4trainable_variables
flayer_regularization_losses
gmetrics
hlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:(2dense_18/kernel
:(2dense_18/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
°
inon_trainable_variables
8regularization_losses
9	variables

jlayers
:trainable_variables
klayer_regularization_losses
lmetrics
mlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:(2dense_19/kernel
:(2dense_19/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
°
nnon_trainable_variables
>regularization_losses
?	variables

olayers
@trainable_variables
player_regularization_losses
qmetrics
rlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
snon_trainable_variables
Bregularization_losses
C	variables

tlayers
Dtrainable_variables
ulayer_regularization_losses
vmetrics
wlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
<
0
1
,2
-3"
trackable_list_wrapper
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
9"
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
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
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
.
,0
-1"
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
ö2ó
*__inference_model_1_layer_call_fn_13667097
*__inference_model_1_layer_call_fn_13667015
*__inference_model_1_layer_call_fn_13667359
*__inference_model_1_layer_call_fn_13667396À
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
â2ß
E__inference_model_1_layer_call_and_return_conditional_losses_13667245
E__inference_model_1_layer_call_and_return_conditional_losses_13666932
E__inference_model_1_layer_call_and_return_conditional_losses_13667322
E__inference_model_1_layer_call_and_return_conditional_losses_13666887À
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
á2Þ
#__inference__wrapped_model_13666351¶
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
annotationsª *&¢#
!
input_4ÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_dense_16_layer_call_fn_13667416¢
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
ð2í
F__inference_dense_16_layer_call_and_return_conditional_losses_13667407¢
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
°2­
9__inference_batch_normalization_12_layer_call_fn_13667485
9__inference_batch_normalization_12_layer_call_fn_13667498´
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
æ2ã
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667452
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667472´
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
Ó2Ð
)__inference_elu_12_layer_call_fn_13667508¢
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
î2ë
D__inference_elu_12_layer_call_and_return_conditional_losses_13667503¢
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
Õ2Ò
+__inference_dense_17_layer_call_fn_13667528¢
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
ð2í
F__inference_dense_17_layer_call_and_return_conditional_losses_13667519¢
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
°2­
9__inference_batch_normalization_13_layer_call_fn_13667610
9__inference_batch_normalization_13_layer_call_fn_13667597´
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
æ2ã
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667564
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667584´
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
Ó2Ð
)__inference_elu_13_layer_call_fn_13667620¢
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
î2ë
D__inference_elu_13_layer_call_and_return_conditional_losses_13667615¢
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
Õ2Ò
+__inference_dense_18_layer_call_fn_13667640¢
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
ð2í
F__inference_dense_18_layer_call_and_return_conditional_losses_13667631¢
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
Õ2Ò
+__inference_dense_19_layer_call_fn_13667660¢
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
ð2í
F__inference_dense_19_layer_call_and_return_conditional_losses_13667651¢
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
 2
+__inference_lambda_1_layer_call_fn_13667704
+__inference_lambda_1_layer_call_fn_13667698À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667676
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667692À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_13667136input_4"
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
  
#__inference__wrapped_model_13666351y#$-*,+67<=0¢-
&¢#
!
input_4ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
lambda_1"
lambda_1ÿÿÿÿÿÿÿÿÿ(º
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667452b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_13667472b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_12_layer_call_fn_13667485U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_12_layer_call_fn_13667498U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿº
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667564b,-*+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_13667584b-*,+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_13_layer_call_fn_13667597U,-*+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_13_layer_call_fn_13667610U-*,+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_16_layer_call_and_return_conditional_losses_13667407\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_16_layer_call_fn_13667416O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_17_layer_call_and_return_conditional_losses_13667519\#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_17_layer_call_fn_13667528O#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_18_layer_call_and_return_conditional_losses_13667631\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ~
+__inference_dense_18_layer_call_fn_13667640O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ(¦
F__inference_dense_19_layer_call_and_return_conditional_losses_13667651\<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ~
+__inference_dense_19_layer_call_fn_13667660O<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ( 
D__inference_elu_12_layer_call_and_return_conditional_losses_13667503X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
)__inference_elu_12_layer_call_fn_13667508K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
D__inference_elu_13_layer_call_and_return_conditional_losses_13667615X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
)__inference_elu_13_layer_call_fn_13667620K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÖ
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667676b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ(

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 Ö
F__inference_lambda_1_layer_call_and_return_conditional_losses_13667692b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ(

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ­
+__inference_lambda_1_layer_call_fn_13667698~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ(

 
p
ª "ÿÿÿÿÿÿÿÿÿ(­
+__inference_lambda_1_layer_call_fn_13667704~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ(

 
p 
ª "ÿÿÿÿÿÿÿÿÿ(¼
E__inference_model_1_layer_call_and_return_conditional_losses_13666887s#$,-*+67<=8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ¼
E__inference_model_1_layer_call_and_return_conditional_losses_13666932s#$-*,+67<=8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 »
E__inference_model_1_layer_call_and_return_conditional_losses_13667245r#$,-*+67<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 »
E__inference_model_1_layer_call_and_return_conditional_losses_13667322r#$-*,+67<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
*__inference_model_1_layer_call_fn_13667015f#$,-*+67<=8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ(
*__inference_model_1_layer_call_fn_13667097f#$-*,+67<=8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(
*__inference_model_1_layer_call_fn_13667359e#$,-*+67<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ(
*__inference_model_1_layer_call_fn_13667396e#$-*,+67<=7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ(¯
&__inference_signature_wrapper_13667136#$-*,+67<=;¢8
¢ 
1ª.
,
input_4!
input_4ÿÿÿÿÿÿÿÿÿ"3ª0
.
lambda_1"
lambda_1ÿÿÿÿÿÿÿÿÿ(