µū
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8į
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0

batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma

0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:*
dtype0

batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta

/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:*
dtype0

"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean

6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance

:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:
*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:
*
dtype0

batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_20/gamma

0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
:
*
dtype0

batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_20/beta

/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
:
*
dtype0

"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_20/moving_mean

6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
:
*
dtype0
¤
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_20/moving_variance

:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
:
*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:

*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:
*
dtype0

batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_21/gamma

0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:
*
dtype0

batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_21/beta

/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:
*
dtype0

"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_21/moving_mean

6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:
*
dtype0
¤
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_21/moving_variance

:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:
*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:
*
dtype0

NoOpNoOp
ļ)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ŗ)
value )B) B)
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
		variables

regularization_losses
trainable_variables
	keras_api

signatures
 
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api

"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(regularization_losses
)trainable_variables
*	keras_api
h

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api

1axis
	2gamma
3beta
4moving_mean
5moving_variance
6	variables
7regularization_losses
8trainable_variables
9	keras_api
^

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api

0
1
2
3
4
5
6
#7
$8
%9
&10
+11
,12
213
314
415
516
:17
 
V
0
1
2
3
4
#5
$6
+7
,8
29
310
:11
­
		variables
?layer_regularization_losses

@layers

regularization_losses
trainable_variables
Anon_trainable_variables
Blayer_metrics
Cmetrics
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
	variables
Dlayer_regularization_losses

Elayers
regularization_losses
trainable_variables
Fnon_trainable_variables
Glayer_metrics
Hmetrics
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
­
	variables
Ilayer_regularization_losses

Jlayers
regularization_losses
trainable_variables
Knon_trainable_variables
Llayer_metrics
Mmetrics
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
Nlayer_regularization_losses

Olayers
regularization_losses
 trainable_variables
Pnon_trainable_variables
Qlayer_metrics
Rmetrics
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
%2
&3
 

#0
$1
­
'	variables
Slayer_regularization_losses

Tlayers
(regularization_losses
)trainable_variables
Unon_trainable_variables
Vlayer_metrics
Wmetrics
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
­
-	variables
Xlayer_regularization_losses

Ylayers
.regularization_losses
/trainable_variables
Znon_trainable_variables
[layer_metrics
\metrics
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

20
31
42
53
 

20
31
­
6	variables
]layer_regularization_losses

^layers
7regularization_losses
8trainable_variables
_non_trainable_variables
`layer_metrics
ametrics
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

:0
 

:0
­
;	variables
blayer_regularization_losses

clayers
<regularization_losses
=trainable_variables
dnon_trainable_variables
elayer_metrics
fmetrics
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
*
0
1
%2
&3
44
55
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
0
1
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
%0
&1
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
40
51
 
 
 
 
 
 
 
z
serving_default_input_6Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Ī
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6dense_14/kernel&batch_normalization_19/moving_variancebatch_normalization_19/gamma"batch_normalization_19/moving_meanbatch_normalization_19/betadense_15/kerneldense_15/bias&batch_normalization_20/moving_variancebatch_normalization_20/gamma"batch_normalization_20/moving_meanbatch_normalization_20/betadense_16/kerneldense_16/bias&batch_normalization_21/moving_variancebatch_normalization_21/gamma"batch_normalization_21/moving_meanbatch_normalization_21/betadense_17/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1077793
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOpConst*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1078464
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kernelbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_15/kerneldense_15/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_variancedense_16/kerneldense_16/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_17/kernel*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1078528ų

Č

ņ
%__inference_signature_wrapper_1077793
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
identity¢StatefulPartitionedCallŖ
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
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_10769022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
»
«
8__inference_batch_normalization_19_layer_call_fn_1078168

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_10770312
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
ļ	
Ž
E__inference_dense_15_layer_call_and_return_conditional_losses_1077392

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
Ī
 
E__inference_dense_14_layer_call_and_return_conditional_losses_1078079

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¾
p
*__inference_dense_17_layer_call_fn_1078387

inputs
unknown
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_10775132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

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
č

ö
)__inference_Decoder_layer_call_fn_1077662
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
identity¢StatefulPartitionedCallĘ
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
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_10776232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
u
«
D__inference_Decoder_layer_call_and_return_conditional_losses_1077989

inputs+
'dense_14_matmul_readvariableop_resource<
8batch_normalization_19_batchnorm_readvariableop_resource@
<batch_normalization_19_batchnorm_mul_readvariableop_resource>
:batch_normalization_19_batchnorm_readvariableop_1_resource>
:batch_normalization_19_batchnorm_readvariableop_2_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource@
<batch_normalization_20_batchnorm_mul_readvariableop_resource>
:batch_normalization_20_batchnorm_readvariableop_1_resource>
:batch_normalization_20_batchnorm_readvariableop_2_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource@
<batch_normalization_21_batchnorm_mul_readvariableop_resource>
:batch_normalization_21_batchnorm_readvariableop_1_resource>
:batch_normalization_21_batchnorm_readvariableop_2_resource+
'dense_17_matmul_readvariableop_resource
identity¢/batch_normalization_19/batchnorm/ReadVariableOp¢1batch_normalization_19/batchnorm/ReadVariableOp_1¢1batch_normalization_19/batchnorm/ReadVariableOp_2¢3batch_normalization_19/batchnorm/mul/ReadVariableOp¢/batch_normalization_20/batchnorm/ReadVariableOp¢1batch_normalization_20/batchnorm/ReadVariableOp_1¢1batch_normalization_20/batchnorm/ReadVariableOp_2¢3batch_normalization_20/batchnorm/mul/ReadVariableOp¢/batch_normalization_21/batchnorm/ReadVariableOp¢1batch_normalization_21/batchnorm/ReadVariableOp_1¢1batch_normalization_21/batchnorm/ReadVariableOp_2¢3batch_normalization_21/batchnorm/mul/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOpØ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/MatMuls
dense_14/ReluReludense_14/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/Relu×
/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_19/batchnorm/ReadVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_19/batchnorm/add/yä
$batch_normalization_19/batchnorm/addAddV27batch_normalization_19/batchnorm/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/addØ
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/Rsqrtć
3batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_19/batchnorm/mul/ReadVariableOpį
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:0;batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulŠ
&batch_normalization_19/batchnorm/mul_1Muldense_14/Relu:activations:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_19/batchnorm/mul_1Ż
1batch_normalization_19/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_19_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_19/batchnorm/ReadVariableOp_1į
&batch_normalization_19/batchnorm/mul_2Mul9batch_normalization_19/batchnorm/ReadVariableOp_1:value:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2Ż
1batch_normalization_19/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_19_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_19/batchnorm/ReadVariableOp_2ß
$batch_normalization_19/batchnorm/subSub9batch_normalization_19/batchnorm/ReadVariableOp_2:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subį
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_19/batchnorm/add_1Ø
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_15/BiasAdd/ReadVariableOp„
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/Relu×
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_20/batchnorm/add/yä
$batch_normalization_20/batchnorm/addAddV27batch_normalization_20/batchnorm/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/addØ
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_20/batchnorm/Rsqrtć
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOpį
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/mulŠ
&batch_normalization_20/batchnorm/mul_1Muldense_15/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_20/batchnorm/mul_1Ż
1batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_1į
&batch_normalization_20/batchnorm/mul_2Mul9batch_normalization_20/batchnorm/ReadVariableOp_1:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_20/batchnorm/mul_2Ż
1batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_2ß
$batch_normalization_20/batchnorm/subSub9batch_normalization_20/batchnorm/ReadVariableOp_2:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/subį
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_20/batchnorm/add_1Ø
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_16/MatMul/ReadVariableOp²
dense_16/MatMulMatMul*batch_normalization_20/batchnorm/add_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_16/BiasAdd/ReadVariableOp„
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/Relu×
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_21/batchnorm/add/yä
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/addØ
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_21/batchnorm/Rsqrtć
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOpį
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/mulŠ
&batch_normalization_21/batchnorm/mul_1Muldense_16/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_21/batchnorm/mul_1Ż
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_1į
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_21/batchnorm/mul_2Ż
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_2ß
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/subį
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_21/batchnorm/add_1Ø
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_17/MatMul/ReadVariableOp²
dense_17/MatMulMatMul*batch_normalization_21/batchnorm/add_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_17/MatMuls
dense_17/TanhTanhdense_17/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_17/Tanh
IdentityIdentitydense_17/Tanh:y:00^batch_normalization_19/batchnorm/ReadVariableOp2^batch_normalization_19/batchnorm/ReadVariableOp_12^batch_normalization_19/batchnorm/ReadVariableOp_24^batch_normalization_19/batchnorm/mul/ReadVariableOp0^batch_normalization_20/batchnorm/ReadVariableOp2^batch_normalization_20/batchnorm/ReadVariableOp_12^batch_normalization_20/batchnorm/ReadVariableOp_24^batch_normalization_20/batchnorm/mul/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2b
/batch_normalization_19/batchnorm/ReadVariableOp/batch_normalization_19/batchnorm/ReadVariableOp2f
1batch_normalization_19/batchnorm/ReadVariableOp_11batch_normalization_19/batchnorm/ReadVariableOp_12f
1batch_normalization_19/batchnorm/ReadVariableOp_21batch_normalization_19/batchnorm/ReadVariableOp_22j
3batch_normalization_19/batchnorm/mul/ReadVariableOp3batch_normalization_19/batchnorm/mul/ReadVariableOp2b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2f
1batch_normalization_20/batchnorm/ReadVariableOp_11batch_normalization_20/batchnorm/ReadVariableOp_12f
1batch_normalization_20/batchnorm/ReadVariableOp_21batch_normalization_20/batchnorm/ReadVariableOp_22j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
,
¶
D__inference_Decoder_layer_call_and_return_conditional_losses_1077711

inputs
dense_14_1077667"
batch_normalization_19_1077670"
batch_normalization_19_1077672"
batch_normalization_19_1077674"
batch_normalization_19_1077676
dense_15_1077679
dense_15_1077681"
batch_normalization_20_1077684"
batch_normalization_20_1077686"
batch_normalization_20_1077688"
batch_normalization_20_1077690
dense_16_1077693
dense_16_1077695"
batch_normalization_21_1077698"
batch_normalization_21_1077700"
batch_normalization_21_1077702"
batch_normalization_21_1077704
dense_17_1077707
identity¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_1077667*
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
GPU 2J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_10773342"
 dense_14/StatefulPartitionedCallÄ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_19_1077670batch_normalization_19_1077672batch_normalization_19_1077674batch_normalization_19_1077676*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_107703120
.batch_normalization_19/StatefulPartitionedCallČ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_15_1077679dense_15_1077681*
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
GPU 2J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_10773922"
 dense_15/StatefulPartitionedCallÄ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_20_1077684batch_normalization_20_1077686batch_normalization_20_1077688batch_normalization_20_1077690*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_107717120
.batch_normalization_20/StatefulPartitionedCallČ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0dense_16_1077693dense_16_1077695*
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
GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_10774542"
 dense_16/StatefulPartitionedCallÄ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_21_1077698batch_normalization_21_1077700batch_normalization_21_1077702batch_normalization_21_1077704*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_107731120
.batch_normalization_21/StatefulPartitionedCall“
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_17_1077707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_10775132"
 dense_17/StatefulPartitionedCall
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
óķ
ū
D__inference_Decoder_layer_call_and_return_conditional_losses_1077915

inputs+
'dense_14_matmul_readvariableop_resource2
.batch_normalization_19_assignmovingavg_10778084
0batch_normalization_19_assignmovingavg_1_1077814@
<batch_normalization_19_batchnorm_mul_readvariableop_resource<
8batch_normalization_19_batchnorm_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource2
.batch_normalization_20_assignmovingavg_10778474
0batch_normalization_20_assignmovingavg_1_1077853@
<batch_normalization_20_batchnorm_mul_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource2
.batch_normalization_21_assignmovingavg_10778864
0batch_normalization_21_assignmovingavg_1_1077892@
<batch_normalization_21_batchnorm_mul_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource+
'dense_17_matmul_readvariableop_resource
identity¢:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_19/AssignMovingAvg/ReadVariableOp¢<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_19/batchnorm/ReadVariableOp¢3batch_normalization_19/batchnorm/mul/ReadVariableOp¢:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_20/AssignMovingAvg/ReadVariableOp¢<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_20/batchnorm/ReadVariableOp¢3batch_normalization_20/batchnorm/mul/ReadVariableOp¢:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_21/AssignMovingAvg/ReadVariableOp¢<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_21/batchnorm/ReadVariableOp¢3batch_normalization_21/batchnorm/mul/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOpØ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOp
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/MatMuls
dense_14/ReluReludense_14/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_14/Reluø
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_19/moments/mean/reduction_indicesé
#batch_normalization_19/moments/meanMeandense_14/Relu:activations:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_19/moments/meanĮ
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_19/moments/StopGradientž
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_14/Relu:activations:04batch_normalization_19/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’22
0batch_normalization_19/moments/SquaredDifferenceĄ
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_19/moments/variance/reduction_indices
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_19/moments/varianceÅ
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_19/moments/SqueezeĶ
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_19/moments/Squeeze_1
,batch_normalization_19/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_19/AssignMovingAvg/1077808*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_19/AssignMovingAvg/decayŁ
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_19_assignmovingavg_1077808*
_output_shapes
:*
dtype027
5batch_normalization_19/AssignMovingAvg/ReadVariableOpå
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_19/AssignMovingAvg/1077808*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/subÜ
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:05batch_normalization_19/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_19/AssignMovingAvg/1077808*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/mul»
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_19_assignmovingavg_1077808.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_19/AssignMovingAvg/1077808*
_output_shapes
 *
dtype02<
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_19/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_19/AssignMovingAvg_1/1077814*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_19/AssignMovingAvg_1/decayß
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_19_assignmovingavg_1_1077814*
_output_shapes
:*
dtype029
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpļ
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_19/AssignMovingAvg_1/1077814*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/subę
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:07batch_normalization_19/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_19/AssignMovingAvg_1/1077814*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/mulĒ
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_19_assignmovingavg_1_10778140batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_19/AssignMovingAvg_1/1077814*
_output_shapes
 *
dtype02>
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_19/batchnorm/add/yŽ
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/addØ
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/Rsqrtć
3batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_19/batchnorm/mul/ReadVariableOpį
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:0;batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulŠ
&batch_normalization_19/batchnorm/mul_1Muldense_14/Relu:activations:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_19/batchnorm/mul_1×
&batch_normalization_19/batchnorm/mul_2Mul/batch_normalization_19/moments/Squeeze:output:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2×
/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_19/batchnorm/ReadVariableOpŻ
$batch_normalization_19/batchnorm/subSub7batch_normalization_19/batchnorm/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subį
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’2(
&batch_normalization_19/batchnorm/add_1Ø
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_19/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/MatMul§
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_15/BiasAdd/ReadVariableOp„
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_15/Reluø
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_20/moments/mean/reduction_indicesé
#batch_normalization_20/moments/meanMeandense_15/Relu:activations:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_20/moments/meanĮ
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_20/moments/StopGradientž
0batch_normalization_20/moments/SquaredDifferenceSquaredDifferencedense_15/Relu:activations:04batch_normalization_20/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
22
0batch_normalization_20/moments/SquaredDifferenceĄ
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_20/moments/variance/reduction_indices
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_20/moments/varianceÅ
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_20/moments/SqueezeĶ
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1
,batch_normalization_20/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_20/AssignMovingAvg/1077847*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_20/AssignMovingAvg/decayŁ
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_20_assignmovingavg_1077847*
_output_shapes
:
*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOpå
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_20/AssignMovingAvg/1077847*
_output_shapes
:
2,
*batch_normalization_20/AssignMovingAvg/subÜ
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:05batch_normalization_20/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_20/AssignMovingAvg/1077847*
_output_shapes
:
2,
*batch_normalization_20/AssignMovingAvg/mul»
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_20_assignmovingavg_1077847.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_20/AssignMovingAvg/1077847*
_output_shapes
 *
dtype02<
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_20/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_20/AssignMovingAvg_1/1077853*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_20/AssignMovingAvg_1/decayß
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_20_assignmovingavg_1_1077853*
_output_shapes
:
*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpļ
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_20/AssignMovingAvg_1/1077853*
_output_shapes
:
2.
,batch_normalization_20/AssignMovingAvg_1/subę
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:07batch_normalization_20/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_20/AssignMovingAvg_1/1077853*
_output_shapes
:
2.
,batch_normalization_20/AssignMovingAvg_1/mulĒ
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_20_assignmovingavg_1_10778530batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_20/AssignMovingAvg_1/1077853*
_output_shapes
 *
dtype02>
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_20/batchnorm/add/yŽ
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/addØ
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_20/batchnorm/Rsqrtć
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOpį
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/mulŠ
&batch_normalization_20/batchnorm/mul_1Muldense_15/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_20/batchnorm/mul_1×
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_20/batchnorm/mul_2×
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOpŻ
$batch_normalization_20/batchnorm/subSub7batch_normalization_20/batchnorm/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_20/batchnorm/subį
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_20/batchnorm/add_1Ø
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_16/MatMul/ReadVariableOp²
dense_16/MatMulMatMul*batch_normalization_20/batchnorm/add_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_16/BiasAdd/ReadVariableOp„
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_16/Reluø
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_21/moments/mean/reduction_indicesé
#batch_normalization_21/moments/meanMeandense_16/Relu:activations:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_21/moments/meanĮ
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_21/moments/StopGradientž
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_16/Relu:activations:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’
22
0batch_normalization_21/moments/SquaredDifferenceĄ
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_21/moments/variance/reduction_indices
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_21/moments/varianceÅ
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_21/moments/SqueezeĶ
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_21/moments/Squeeze_1
,batch_normalization_21/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_21/AssignMovingAvg/1077886*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_21/AssignMovingAvg/decayŁ
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_21_assignmovingavg_1077886*
_output_shapes
:
*
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOpå
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_21/AssignMovingAvg/1077886*
_output_shapes
:
2,
*batch_normalization_21/AssignMovingAvg/subÜ
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_21/AssignMovingAvg/1077886*
_output_shapes
:
2,
*batch_normalization_21/AssignMovingAvg/mul»
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_21_assignmovingavg_1077886.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_21/AssignMovingAvg/1077886*
_output_shapes
 *
dtype02<
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_21/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_21/AssignMovingAvg_1/1077892*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_21/AssignMovingAvg_1/decayß
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_21_assignmovingavg_1_1077892*
_output_shapes
:
*
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpļ
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_21/AssignMovingAvg_1/1077892*
_output_shapes
:
2.
,batch_normalization_21/AssignMovingAvg_1/subę
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_21/AssignMovingAvg_1/1077892*
_output_shapes
:
2.
,batch_normalization_21/AssignMovingAvg_1/mulĒ
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_21_assignmovingavg_1_10778920batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_21/AssignMovingAvg_1/1077892*
_output_shapes
 *
dtype02>
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_21/batchnorm/add/yŽ
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/addØ
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_21/batchnorm/Rsqrtć
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOpį
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/mulŠ
&batch_normalization_21/batchnorm/mul_1Muldense_16/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_21/batchnorm/mul_1×
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_21/batchnorm/mul_2×
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOpŻ
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_21/batchnorm/subį
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
2(
&batch_normalization_21/batchnorm/add_1Ø
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_17/MatMul/ReadVariableOp²
dense_17/MatMulMatMul*batch_normalization_21/batchnorm/add_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_17/MatMuls
dense_17/TanhTanhdense_17/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_17/TanhÆ

IdentityIdentitydense_17/Tanh:y:0;^batch_normalization_19/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_19/AssignMovingAvg/ReadVariableOp=^batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_19/batchnorm/ReadVariableOp4^batch_normalization_19/batchnorm/mul/ReadVariableOp;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_20/AssignMovingAvg/ReadVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_20/batchnorm/ReadVariableOp4^batch_normalization_20/batchnorm/mul/ReadVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_21/AssignMovingAvg/ReadVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2x
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_19/batchnorm/ReadVariableOp/batch_normalization_19/batchnorm/ReadVariableOp2j
3batch_normalization_19/batchnorm/mul/ReadVariableOp3batch_normalization_19/batchnorm/mul/ReadVariableOp2x
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2x
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä
 
E__inference_dense_17_layer_call_and_return_conditional_losses_1078380

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
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
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

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
Ö

S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078142

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
Ö

S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078244

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
Ž

*__inference_dense_15_layer_call_fn_1078188

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
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
GPU 2J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_10773922
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
Ž

*__inference_dense_16_layer_call_fn_1078290

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
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
GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_10774542
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
0
Ģ
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1077278

inputs
assignmovingavg_1077253
assignmovingavg_1_1077259)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077253*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077253*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077253*
_output_shapes
:
2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077253*
_output_shapes
:
2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077253AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077253*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077259*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077259*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077259*
_output_shapes
:
2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077259*
_output_shapes
:
2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077259AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077259*
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
¹
«
8__inference_batch_normalization_21_layer_call_fn_1078359

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10772782
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
¾
p
*__inference_dense_14_layer_call_fn_1078086

inputs
unknown
identity¢StatefulPartitionedCallč
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
GPU 2J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_10773342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ė

õ
)__inference_Decoder_layer_call_fn_1078071

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
identity¢StatefulPartitionedCallĖ
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
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_10777112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ļ	
Ž
E__inference_dense_16_layer_call_and_return_conditional_losses_1078281

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
Ö

S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078346

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
0
Ģ
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078122

inputs
assignmovingavg_1078097
assignmovingavg_1_1078103)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078097*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1078097*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078097*
_output_shapes
:2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078097*
_output_shapes
:2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1078097AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078097*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078103*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1078103*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078103*
_output_shapes
:2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078103*
_output_shapes
:2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1078103AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078103*
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
ļ	
Ž
E__inference_dense_16_layer_call_and_return_conditional_losses_1077454

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
Ö

S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1077311

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
«
8__inference_batch_normalization_20_layer_call_fn_1078270

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10771712
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
0
Ģ
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1076998

inputs
assignmovingavg_1076973
assignmovingavg_1_1076979)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076973*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1076973*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076973*
_output_shapes
:2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076973*
_output_shapes
:2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1076973AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076973*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076979*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1076979*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076979*
_output_shapes
:2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076979*
_output_shapes
:2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1076979AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076979*
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
0
Ģ
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078224

inputs
assignmovingavg_1078199
assignmovingavg_1_1078205)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078199*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1078199*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078199*
_output_shapes
:
2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078199*
_output_shapes
:
2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1078199AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078199*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078205*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1078205*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078205*
_output_shapes
:
2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078205*
_output_shapes
:
2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1078205AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078205*
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
ļ	
Ž
E__inference_dense_15_layer_call_and_return_conditional_losses_1078179

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
,
·
D__inference_Decoder_layer_call_and_return_conditional_losses_1077526
input_6
dense_14_1077343"
batch_normalization_19_1077372"
batch_normalization_19_1077374"
batch_normalization_19_1077376"
batch_normalization_19_1077378
dense_15_1077403
dense_15_1077405"
batch_normalization_20_1077434"
batch_normalization_20_1077436"
batch_normalization_20_1077438"
batch_normalization_20_1077440
dense_16_1077465
dense_16_1077467"
batch_normalization_21_1077496"
batch_normalization_21_1077498"
batch_normalization_21_1077500"
batch_normalization_21_1077502
dense_17_1077522
identity¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_14_1077343*
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
GPU 2J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_10773342"
 dense_14/StatefulPartitionedCallĀ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_19_1077372batch_normalization_19_1077374batch_normalization_19_1077376batch_normalization_19_1077378*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_107699820
.batch_normalization_19/StatefulPartitionedCallČ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_15_1077403dense_15_1077405*
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
GPU 2J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_10773922"
 dense_15/StatefulPartitionedCallĀ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_20_1077434batch_normalization_20_1077436batch_normalization_20_1077438batch_normalization_20_1077440*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_107713820
.batch_normalization_20/StatefulPartitionedCallČ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0dense_16_1077465dense_16_1077467*
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
GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_10774542"
 dense_16/StatefulPartitionedCallĀ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_21_1077496batch_normalization_21_1077498batch_normalization_21_1077500batch_normalization_21_1077502*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_107727820
.batch_normalization_21/StatefulPartitionedCall“
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_17_1077522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_10775132"
 dense_17/StatefulPartitionedCall
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
1
	
 __inference__traced_save_1078464
file_prefix.
*savev2_dense_14_kernel_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop
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
ShardedFilename®	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ą
value¶B³B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices©	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop*savev2_dense_17_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
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

identity_1Identity_1:output:0*
_input_shapes
: ::::::
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
: 2(
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
:

_output_shapes
: 
ĆP
å

#__inference__traced_restore_1078528
file_prefix$
 assignvariableop_dense_14_kernel3
/assignvariableop_1_batch_normalization_19_gamma2
.assignvariableop_2_batch_normalization_19_beta9
5assignvariableop_3_batch_normalization_19_moving_mean=
9assignvariableop_4_batch_normalization_19_moving_variance&
"assignvariableop_5_dense_15_kernel$
 assignvariableop_6_dense_15_bias3
/assignvariableop_7_batch_normalization_20_gamma2
.assignvariableop_8_batch_normalization_20_beta9
5assignvariableop_9_batch_normalization_20_moving_mean>
:assignvariableop_10_batch_normalization_20_moving_variance'
#assignvariableop_11_dense_16_kernel%
!assignvariableop_12_dense_16_bias4
0assignvariableop_13_batch_normalization_21_gamma3
/assignvariableop_14_batch_normalization_21_beta:
6assignvariableop_15_batch_normalization_21_moving_mean>
:assignvariableop_16_batch_normalization_21_moving_variance'
#assignvariableop_17_dense_17_kernel
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9“	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ą
value¶B³B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names“
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
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

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1“
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_19_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_19_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ŗ
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_19_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¾
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_19_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_15_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6„
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_15_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7“
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_20_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_20_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ŗ
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_20_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ā
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_20_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_16_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_16_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ø
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_21_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_21_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¾
AssignVariableOp_15AssignVariableOp6assignvariableop_15_batch_normalization_21_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ā
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_normalization_21_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_17_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpź
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18Ż
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
å

õ
)__inference_Decoder_layer_call_fn_1078030

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
identity¢StatefulPartitionedCallÅ
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
:’’’’’’’’’*.
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_10776232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
0
Ģ
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078326

inputs
assignmovingavg_1078301
assignmovingavg_1_1078307)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078301*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1078301*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078301*
_output_shapes
:
2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1078301*
_output_shapes
:
2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1078301AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1078301*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078307*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1078307*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078307*
_output_shapes
:
2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1078307*
_output_shapes
:
2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1078307AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1078307*
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
¹
«
8__inference_batch_normalization_20_layer_call_fn_1078257

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10771382
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
¹
«
8__inference_batch_normalization_19_layer_call_fn_1078155

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_10769982
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
»
«
8__inference_batch_normalization_21_layer_call_fn_1078372

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10773112
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
0
Ģ
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1077138

inputs
assignmovingavg_1077113
assignmovingavg_1_1077119)
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
moments/Squeeze_1Ķ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077113*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077113*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpņ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077113*
_output_shapes
:
2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077113*
_output_shapes
:
2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077113AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077113*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077119*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077119*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077119*
_output_shapes
:
2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077119*
_output_shapes
:
2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077119AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077119*
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
Ö

S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1077171

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
ī

ö
)__inference_Decoder_layer_call_fn_1077750
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
identity¢StatefulPartitionedCallĢ
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
:’’’’’’’’’*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_10777112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
Š
Ŗ
"__inference__wrapped_model_1076902
input_63
/decoder_dense_14_matmul_readvariableop_resourceD
@decoder_batch_normalization_19_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_19_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_19_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_19_batchnorm_readvariableop_2_resource3
/decoder_dense_15_matmul_readvariableop_resource4
0decoder_dense_15_biasadd_readvariableop_resourceD
@decoder_batch_normalization_20_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_20_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_20_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_20_batchnorm_readvariableop_2_resource3
/decoder_dense_16_matmul_readvariableop_resource4
0decoder_dense_16_biasadd_readvariableop_resourceD
@decoder_batch_normalization_21_batchnorm_readvariableop_resourceH
Ddecoder_batch_normalization_21_batchnorm_mul_readvariableop_resourceF
Bdecoder_batch_normalization_21_batchnorm_readvariableop_1_resourceF
Bdecoder_batch_normalization_21_batchnorm_readvariableop_2_resource3
/decoder_dense_17_matmul_readvariableop_resource
identity¢7Decoder/batch_normalization_19/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOp¢7Decoder/batch_normalization_20/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOp¢7Decoder/batch_normalization_21/batchnorm/ReadVariableOp¢9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_1¢9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_2¢;Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOp¢&Decoder/dense_14/MatMul/ReadVariableOp¢'Decoder/dense_15/BiasAdd/ReadVariableOp¢&Decoder/dense_15/MatMul/ReadVariableOp¢'Decoder/dense_16/BiasAdd/ReadVariableOp¢&Decoder/dense_16/MatMul/ReadVariableOp¢&Decoder/dense_17/MatMul/ReadVariableOpĄ
&Decoder/dense_14/MatMul/ReadVariableOpReadVariableOp/decoder_dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Decoder/dense_14/MatMul/ReadVariableOp§
Decoder/dense_14/MatMulMatMulinput_6.Decoder/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_14/MatMul
Decoder/dense_14/ReluRelu!Decoder/dense_14/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_14/Reluļ
7Decoder/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7Decoder/batch_normalization_19/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_19/batchnorm/add/y
,Decoder/batch_normalization_19/batchnorm/addAddV2?Decoder/batch_normalization_19/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_19/batchnorm/addĄ
.Decoder/batch_normalization_19/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_19/batchnorm/Rsqrtū
;Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_19/batchnorm/mulMul2Decoder/batch_normalization_19/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_19/batchnorm/mulš
.Decoder/batch_normalization_19/batchnorm/mul_1Mul#Decoder/dense_14/Relu:activations:00Decoder/batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’20
.Decoder/batch_normalization_19/batchnorm/mul_1õ
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_19_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_19/batchnorm/mul_2MulADecoder/batch_normalization_19/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.Decoder/batch_normalization_19/batchnorm/mul_2õ
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_19_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_19/batchnorm/subSubADecoder/batch_normalization_19/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,Decoder/batch_normalization_19/batchnorm/sub
.Decoder/batch_normalization_19/batchnorm/add_1AddV22Decoder/batch_normalization_19/batchnorm/mul_1:z:00Decoder/batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’20
.Decoder/batch_normalization_19/batchnorm/add_1Ą
&Decoder/dense_15/MatMul/ReadVariableOpReadVariableOp/decoder_dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Decoder/dense_15/MatMul/ReadVariableOpŅ
Decoder/dense_15/MatMulMatMul2Decoder/batch_normalization_19/batchnorm/add_1:z:0.Decoder/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_15/MatMulæ
'Decoder/dense_15/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_15_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Decoder/dense_15/BiasAdd/ReadVariableOpÅ
Decoder/dense_15/BiasAddBiasAdd!Decoder/dense_15/MatMul:product:0/Decoder/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_15/BiasAdd
Decoder/dense_15/ReluRelu!Decoder/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_15/Reluļ
7Decoder/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype029
7Decoder/batch_normalization_20/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_20/batchnorm/add/y
,Decoder/batch_normalization_20/batchnorm/addAddV2?Decoder/batch_normalization_20/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_20/batchnorm/addĄ
.Decoder/batch_normalization_20/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_20/batchnorm/Rsqrtū
;Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02=
;Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_20/batchnorm/mulMul2Decoder/batch_normalization_20/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_20/batchnorm/mulš
.Decoder/batch_normalization_20/batchnorm/mul_1Mul#Decoder/dense_15/Relu:activations:00Decoder/batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_20/batchnorm/mul_1õ
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_20/batchnorm/mul_2MulADecoder/batch_normalization_20/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_20/batchnorm/mul_2õ
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_20/batchnorm/subSubADecoder/batch_normalization_20/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_20/batchnorm/sub
.Decoder/batch_normalization_20/batchnorm/add_1AddV22Decoder/batch_normalization_20/batchnorm/mul_1:z:00Decoder/batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_20/batchnorm/add_1Ą
&Decoder/dense_16/MatMul/ReadVariableOpReadVariableOp/decoder_dense_16_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Decoder/dense_16/MatMul/ReadVariableOpŅ
Decoder/dense_16/MatMulMatMul2Decoder/batch_normalization_20/batchnorm/add_1:z:0.Decoder/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_16/MatMulæ
'Decoder/dense_16/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_16_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Decoder/dense_16/BiasAdd/ReadVariableOpÅ
Decoder/dense_16/BiasAddBiasAdd!Decoder/dense_16/MatMul:product:0/Decoder/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_16/BiasAdd
Decoder/dense_16/ReluRelu!Decoder/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
Decoder/dense_16/Reluļ
7Decoder/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp@decoder_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype029
7Decoder/batch_normalization_21/batchnorm/ReadVariableOp„
.Decoder/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.Decoder/batch_normalization_21/batchnorm/add/y
,Decoder/batch_normalization_21/batchnorm/addAddV2?Decoder/batch_normalization_21/batchnorm/ReadVariableOp:value:07Decoder/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_21/batchnorm/addĄ
.Decoder/batch_normalization_21/batchnorm/RsqrtRsqrt0Decoder/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_21/batchnorm/Rsqrtū
;Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpDdecoder_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02=
;Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOp
,Decoder/batch_normalization_21/batchnorm/mulMul2Decoder/batch_normalization_21/batchnorm/Rsqrt:y:0CDecoder/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_21/batchnorm/mulš
.Decoder/batch_normalization_21/batchnorm/mul_1Mul#Decoder/dense_16/Relu:activations:00Decoder/batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_21/batchnorm/mul_1õ
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpBdecoder_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_1
.Decoder/batch_normalization_21/batchnorm/mul_2MulADecoder/batch_normalization_21/batchnorm/ReadVariableOp_1:value:00Decoder/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:
20
.Decoder/batch_normalization_21/batchnorm/mul_2õ
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpBdecoder_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02;
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_2’
,Decoder/batch_normalization_21/batchnorm/subSubADecoder/batch_normalization_21/batchnorm/ReadVariableOp_2:value:02Decoder/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2.
,Decoder/batch_normalization_21/batchnorm/sub
.Decoder/batch_normalization_21/batchnorm/add_1AddV22Decoder/batch_normalization_21/batchnorm/mul_1:z:00Decoder/batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’
20
.Decoder/batch_normalization_21/batchnorm/add_1Ą
&Decoder/dense_17/MatMul/ReadVariableOpReadVariableOp/decoder_dense_17_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Decoder/dense_17/MatMul/ReadVariableOpŅ
Decoder/dense_17/MatMulMatMul2Decoder/batch_normalization_21/batchnorm/add_1:z:0.Decoder/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_17/MatMul
Decoder/dense_17/TanhTanh!Decoder/dense_17/MatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Decoder/dense_17/Tanhµ
IdentityIdentityDecoder/dense_17/Tanh:y:08^Decoder/batch_normalization_19/batchnorm/ReadVariableOp:^Decoder/batch_normalization_19/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_19/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_20/batchnorm/ReadVariableOp:^Decoder/batch_normalization_20/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_20/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOp8^Decoder/batch_normalization_21/batchnorm/ReadVariableOp:^Decoder/batch_normalization_21/batchnorm/ReadVariableOp_1:^Decoder/batch_normalization_21/batchnorm/ReadVariableOp_2<^Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOp'^Decoder/dense_14/MatMul/ReadVariableOp(^Decoder/dense_15/BiasAdd/ReadVariableOp'^Decoder/dense_15/MatMul/ReadVariableOp(^Decoder/dense_16/BiasAdd/ReadVariableOp'^Decoder/dense_16/MatMul/ReadVariableOp'^Decoder/dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2r
7Decoder/batch_normalization_19/batchnorm/ReadVariableOp7Decoder/batch_normalization_19/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_19Decoder/batch_normalization_19/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_19/batchnorm/ReadVariableOp_29Decoder/batch_normalization_19/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_19/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_20/batchnorm/ReadVariableOp7Decoder/batch_normalization_20/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_19Decoder/batch_normalization_20/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_20/batchnorm/ReadVariableOp_29Decoder/batch_normalization_20/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_20/batchnorm/mul/ReadVariableOp2r
7Decoder/batch_normalization_21/batchnorm/ReadVariableOp7Decoder/batch_normalization_21/batchnorm/ReadVariableOp2v
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_19Decoder/batch_normalization_21/batchnorm/ReadVariableOp_12v
9Decoder/batch_normalization_21/batchnorm/ReadVariableOp_29Decoder/batch_normalization_21/batchnorm/ReadVariableOp_22z
;Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOp;Decoder/batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&Decoder/dense_14/MatMul/ReadVariableOp&Decoder/dense_14/MatMul/ReadVariableOp2R
'Decoder/dense_15/BiasAdd/ReadVariableOp'Decoder/dense_15/BiasAdd/ReadVariableOp2P
&Decoder/dense_15/MatMul/ReadVariableOp&Decoder/dense_15/MatMul/ReadVariableOp2R
'Decoder/dense_16/BiasAdd/ReadVariableOp'Decoder/dense_16/BiasAdd/ReadVariableOp2P
&Decoder/dense_16/MatMul/ReadVariableOp&Decoder/dense_16/MatMul/ReadVariableOp2P
&Decoder/dense_17/MatMul/ReadVariableOp&Decoder/dense_17/MatMul/ReadVariableOp:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
Ī
 
E__inference_dense_14_layer_call_and_return_conditional_losses_1077334

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö

S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1077031

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
Ä
 
E__inference_dense_17_layer_call_and_return_conditional_losses_1077513

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
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
MatMulX
TanhTanhMatMul:product:0*
T0*'
_output_shapes
:’’’’’’’’’2
Tanht
IdentityIdentityTanh:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

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
”,
·
D__inference_Decoder_layer_call_and_return_conditional_losses_1077573
input_6
dense_14_1077529"
batch_normalization_19_1077532"
batch_normalization_19_1077534"
batch_normalization_19_1077536"
batch_normalization_19_1077538
dense_15_1077541
dense_15_1077543"
batch_normalization_20_1077546"
batch_normalization_20_1077548"
batch_normalization_20_1077550"
batch_normalization_20_1077552
dense_16_1077555
dense_16_1077557"
batch_normalization_21_1077560"
batch_normalization_21_1077562"
batch_normalization_21_1077564"
batch_normalization_21_1077566
dense_17_1077569
identity¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_14_1077529*
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
GPU 2J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_10773342"
 dense_14/StatefulPartitionedCallÄ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_19_1077532batch_normalization_19_1077534batch_normalization_19_1077536batch_normalization_19_1077538*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_107703120
.batch_normalization_19/StatefulPartitionedCallČ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_15_1077541dense_15_1077543*
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
GPU 2J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_10773922"
 dense_15/StatefulPartitionedCallÄ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_20_1077546batch_normalization_20_1077548batch_normalization_20_1077550batch_normalization_20_1077552*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_107717120
.batch_normalization_20/StatefulPartitionedCallČ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0dense_16_1077555dense_16_1077557*
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
GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_10774542"
 dense_16/StatefulPartitionedCallÄ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_21_1077560batch_normalization_21_1077562batch_normalization_21_1077564batch_normalization_21_1077566*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_107731120
.batch_normalization_21/StatefulPartitionedCall“
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_17_1077569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_10775132"
 dense_17/StatefulPartitionedCall
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_6
,
¶
D__inference_Decoder_layer_call_and_return_conditional_losses_1077623

inputs
dense_14_1077579"
batch_normalization_19_1077582"
batch_normalization_19_1077584"
batch_normalization_19_1077586"
batch_normalization_19_1077588
dense_15_1077591
dense_15_1077593"
batch_normalization_20_1077596"
batch_normalization_20_1077598"
batch_normalization_20_1077600"
batch_normalization_20_1077602
dense_16_1077605
dense_16_1077607"
batch_normalization_21_1077610"
batch_normalization_21_1077612"
batch_normalization_21_1077614"
batch_normalization_21_1077616
dense_17_1077619
identity¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_1077579*
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
GPU 2J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_10773342"
 dense_14/StatefulPartitionedCallĀ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_19_1077582batch_normalization_19_1077584batch_normalization_19_1077586batch_normalization_19_1077588*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_107699820
.batch_normalization_19/StatefulPartitionedCallČ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0dense_15_1077591dense_15_1077593*
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
GPU 2J 8 *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_10773922"
 dense_15/StatefulPartitionedCallĀ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_20_1077596batch_normalization_20_1077598batch_normalization_20_1077600batch_normalization_20_1077602*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_107713820
.batch_normalization_20/StatefulPartitionedCallČ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0dense_16_1077605dense_16_1077607*
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
GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_10774542"
 dense_16/StatefulPartitionedCallĀ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_21_1077610batch_normalization_21_1077612batch_normalization_21_1077614batch_normalization_21_1077616*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_107727820
.batch_normalization_21/StatefulPartitionedCall“
 dense_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_17_1077619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_10775132"
 dense_17/StatefulPartitionedCall
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:’’’’’’’’’::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
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
input_60
serving_default_input_6:0’’’’’’’’’<
dense_170
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ć
öJ
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
		variables

regularization_losses
trainable_variables
	keras_api

signatures
*g&call_and_return_all_conditional_losses
h_default_save_signature
i__call__"G
_tf_keras_networkG{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["dense_17", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["dense_17", 0, 0]]}}}
é"ę
_tf_keras_input_layerĘ{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
ē

kernel
	variables
regularization_losses
trainable_variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"Ģ
_tf_keras_layer²{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
“	
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
ņ

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
*n&call_and_return_all_conditional_losses
o__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
“	
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(regularization_losses
)trainable_variables
*	keras_api
*p&call_and_return_all_conditional_losses
q__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ņ

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
*r&call_and_return_all_conditional_losses
s__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
“	
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6	variables
7regularization_losses
8trainable_variables
9	keras_api
*t&call_and_return_all_conditional_losses
u__call__"ą
_tf_keras_layerĘ{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
č

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
*v&call_and_return_all_conditional_losses
w__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
¦
0
1
2
3
4
5
6
#7
$8
%9
&10
+11
,12
213
314
415
516
:17"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
#5
$6
+7
,8
29
310
:11"
trackable_list_wrapper
Ź
		variables
?layer_regularization_losses

@layers

regularization_losses
trainable_variables
Anon_trainable_variables
Blayer_metrics
Cmetrics
i__call__
h_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
!:2dense_14/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
	variables
Dlayer_regularization_losses

Elayers
regularization_losses
trainable_variables
Fnon_trainable_variables
Glayer_metrics
Hmetrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_19/gamma
):'2batch_normalization_19/beta
2:0 (2"batch_normalization_19/moving_mean
6:4 (2&batch_normalization_19/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
Ilayer_regularization_losses

Jlayers
regularization_losses
trainable_variables
Knon_trainable_variables
Llayer_metrics
Mmetrics
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_15/kernel
:
2dense_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
Nlayer_regularization_losses

Olayers
regularization_losses
 trainable_variables
Pnon_trainable_variables
Qlayer_metrics
Rmetrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_20/gamma
):'
2batch_normalization_20/beta
2:0
 (2"batch_normalization_20/moving_mean
6:4
 (2&batch_normalization_20/moving_variance
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
­
'	variables
Slayer_regularization_losses

Tlayers
(regularization_losses
)trainable_variables
Unon_trainable_variables
Vlayer_metrics
Wmetrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_16/kernel
:
2dense_16/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
­
-	variables
Xlayer_regularization_losses

Ylayers
.regularization_losses
/trainable_variables
Znon_trainable_variables
[layer_metrics
\metrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_21/gamma
):'
2batch_normalization_21/beta
2:0
 (2"batch_normalization_21/moving_mean
6:4
 (2&batch_normalization_21/moving_variance
<
20
31
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
­
6	variables
]layer_regularization_losses

^layers
7regularization_losses
8trainable_variables
_non_trainable_variables
`layer_metrics
ametrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_17/kernel
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
­
;	variables
blayer_regularization_losses

clayers
<regularization_losses
=trainable_variables
dnon_trainable_variables
elayer_metrics
fmetrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
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
J
0
1
%2
&3
44
55"
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
0
1"
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
%0
&1"
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
40
51"
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
Ž2Ū
D__inference_Decoder_layer_call_and_return_conditional_losses_1077989
D__inference_Decoder_layer_call_and_return_conditional_losses_1077526
D__inference_Decoder_layer_call_and_return_conditional_losses_1077573
D__inference_Decoder_layer_call_and_return_conditional_losses_1077915Ą
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
ą2Ż
"__inference__wrapped_model_1076902¶
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
annotationsŖ *&¢#
!
input_6’’’’’’’’’
ņ2ļ
)__inference_Decoder_layer_call_fn_1077662
)__inference_Decoder_layer_call_fn_1078071
)__inference_Decoder_layer_call_fn_1078030
)__inference_Decoder_layer_call_fn_1077750Ą
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
ļ2ģ
E__inference_dense_14_layer_call_and_return_conditional_losses_1078079¢
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
Ō2Ń
*__inference_dense_14_layer_call_fn_1078086¢
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
ä2į
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078142
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078122“
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
®2«
8__inference_batch_normalization_19_layer_call_fn_1078155
8__inference_batch_normalization_19_layer_call_fn_1078168“
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
ļ2ģ
E__inference_dense_15_layer_call_and_return_conditional_losses_1078179¢
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
Ō2Ń
*__inference_dense_15_layer_call_fn_1078188¢
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
ä2į
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078244
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078224“
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
®2«
8__inference_batch_normalization_20_layer_call_fn_1078257
8__inference_batch_normalization_20_layer_call_fn_1078270“
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
ļ2ģ
E__inference_dense_16_layer_call_and_return_conditional_losses_1078281¢
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
Ō2Ń
*__inference_dense_16_layer_call_fn_1078290¢
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
ä2į
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078326
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078346“
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
®2«
8__inference_batch_normalization_21_layer_call_fn_1078372
8__inference_batch_normalization_21_layer_call_fn_1078359“
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
ļ2ģ
E__inference_dense_17_layer_call_and_return_conditional_losses_1078380¢
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
Ō2Ń
*__inference_dense_17_layer_call_fn_1078387¢
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
ĢBÉ
%__inference_signature_wrapper_1077793input_6"
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
 ½
D__inference_Decoder_layer_call_and_return_conditional_losses_1077526u%&#$+,4523:8¢5
.¢+
!
input_6’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ½
D__inference_Decoder_layer_call_and_return_conditional_losses_1077573u&#%$+,5243:8¢5
.¢+
!
input_6’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 ¼
D__inference_Decoder_layer_call_and_return_conditional_losses_1077915t%&#$+,4523:7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’
 ¼
D__inference_Decoder_layer_call_and_return_conditional_losses_1077989t&#%$+,5243:7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
)__inference_Decoder_layer_call_fn_1077662h%&#$+,4523:8¢5
.¢+
!
input_6’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
)__inference_Decoder_layer_call_fn_1077750h&#%$+,5243:8¢5
.¢+
!
input_6’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
)__inference_Decoder_layer_call_fn_1078030g%&#$+,4523:7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
)__inference_Decoder_layer_call_fn_1078071g&#%$+,5243:7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’”
"__inference__wrapped_model_1076902{&#%$+,5243:0¢-
&¢#
!
input_6’’’’’’’’’
Ŗ "3Ŗ0
.
dense_17"
dense_17’’’’’’’’’¹
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078122b3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ¹
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1078142b3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 
8__inference_batch_normalization_19_layer_call_fn_1078155U3¢0
)¢&
 
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’
8__inference_batch_normalization_19_layer_call_fn_1078168U3¢0
)¢&
 
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’¹
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078224b%&#$3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ¹
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1078244b&#%$3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
8__inference_batch_normalization_20_layer_call_fn_1078257U%&#$3¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

8__inference_batch_normalization_20_layer_call_fn_1078270U&#%$3¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
¹
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078326b45233¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "%¢"

0’’’’’’’’’

 ¹
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1078346b52433¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "%¢"

0’’’’’’’’’

 
8__inference_batch_normalization_21_layer_call_fn_1078359U45233¢0
)¢&
 
inputs’’’’’’’’’

p
Ŗ "’’’’’’’’’

8__inference_batch_normalization_21_layer_call_fn_1078372U52433¢0
)¢&
 
inputs’’’’’’’’’

p 
Ŗ "’’’’’’’’’
¤
E__inference_dense_14_layer_call_and_return_conditional_losses_1078079[/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 |
*__inference_dense_14_layer_call_fn_1078086N/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’„
E__inference_dense_15_layer_call_and_return_conditional_losses_1078179\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’

 }
*__inference_dense_15_layer_call_fn_1078188O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
„
E__inference_dense_16_layer_call_and_return_conditional_losses_1078281\+,/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’

 }
*__inference_dense_16_layer_call_fn_1078290O+,/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’
¤
E__inference_dense_17_layer_call_and_return_conditional_losses_1078380[:/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "%¢"

0’’’’’’’’’
 |
*__inference_dense_17_layer_call_fn_1078387N:/¢,
%¢"
 
inputs’’’’’’’’’

Ŗ "’’’’’’’’’°
%__inference_signature_wrapper_1077793&#%$+,5243:;¢8
¢ 
1Ŗ.
,
input_6!
input_6’’’’’’’’’"3Ŗ0
.
dense_17"
dense_17’’’’’’’’’