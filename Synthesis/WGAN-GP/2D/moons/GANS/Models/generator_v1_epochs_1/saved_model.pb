Ó§
Í¢
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ª
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
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
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ë+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*+
valueü*Bù* Bò*
ó
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
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api

"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api

1axis
	2gamma
3beta
4moving_mean
5moving_variance
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api

0
1
2
3
4
5
6
7
#8
$9
%10
&11
+12
,13
214
315
416
517
:18
;19
f
0
1
2
3
4
5
#6
$7
+8
,9
210
311
:12
;13
 
­
@metrics

Alayers
	variables
	trainable_variables
Blayer_metrics

regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
 
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Emetrics

Flayers
	variables
trainable_variables
Glayer_metrics
regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
­
Jmetrics

Klayers
	variables
trainable_variables
Llayer_metrics
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Ometrics

Players
	variables
trainable_variables
Qlayer_metrics
 regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
%2
&3

#0
$1
 
­
Tmetrics

Ulayers
'	variables
(trainable_variables
Vlayer_metrics
)regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
­
Ymetrics

Zlayers
-	variables
.trainable_variables
[layer_metrics
/regularization_losses
\layer_regularization_losses
]non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

20
31
42
53

20
31
 
­
^metrics

_layers
6	variables
7trainable_variables
`layer_metrics
8regularization_losses
alayer_regularization_losses
bnon_trainable_variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
­
cmetrics

dlayers
<	variables
=trainable_variables
elayer_metrics
>regularization_losses
flayer_regularization_losses
gnon_trainable_variables
 
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

serving_default_dense_8_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_8_inputdense_8/kerneldense_8/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_9/kerneldense_9/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_10/kerneldense_10/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betadense_11/kerneldense_11/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_8971
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Â	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpConst*!
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
GPU 2J 8 *&
f!R
__inference__traced_save_9678
­
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_9/kerneldense_9/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_10/kerneldense_10/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_11/kerneldense_11/bias* 
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_9748ä©
Ïñ
å
F__inference_sequential_2_layer_call_and_return_conditional_losses_9099

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource.
*batch_normalization_3_assignmovingavg_89890
,batch_normalization_3_assignmovingavg_1_8995?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*batch_normalization_4_assignmovingavg_90280
,batch_normalization_4_assignmovingavg_1_9034?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource.
*batch_normalization_5_assignmovingavg_90670
,batch_normalization_5_assignmovingavg_1_9073?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
*batch_normalization_3/moments/StopGradientú
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
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
&batch_normalization_3/moments/varianceÂ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeÊ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/8989*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_3/AssignMovingAvg/decayÓ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_3_assignmovingavg_8989*
_output_shapes
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpÝ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/8989*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/subÔ
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/8989*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/mul¯
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_3_assignmovingavg_8989-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_3/AssignMovingAvg/8989*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/8995*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_3/AssignMovingAvg_1/decayÙ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_1_8995*
_output_shapes
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpç
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/8995*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/subÞ
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/8995*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/mul»
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_1_8995/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg_1/8995*
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
%batch_normalization_3/batchnorm/add/yÚ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulÌ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Ó
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ô
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpÙ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subÝ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
*batch_normalization_4/moments/StopGradientú
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
&batch_normalization_4/moments/varianceÂ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeÊ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/9028*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayÓ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_4_assignmovingavg_9028*
_output_shapes
:
*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpÝ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/9028*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/subÔ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/9028*
_output_shapes
:
2+
)batch_normalization_4/AssignMovingAvg/mul¯
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_4_assignmovingavg_9028-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_4/AssignMovingAvg/9028*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/9034*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayÙ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_1_9034*
_output_shapes
:
*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpç
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/9034*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/subÞ
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/9034*
_output_shapes
:
2-
+batch_normalization_4/AssignMovingAvg_1/mul»
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_1_9034/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg_1/9034*
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
%batch_normalization_4/batchnorm/add/yÚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulÌ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ô
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpÙ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/add_1¨
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
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/Relu¶
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesæ
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
*batch_normalization_5/moments/StopGradientû
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
&batch_normalization_5/moments/varianceÂ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeÊ
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1
+batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/9067*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_5/AssignMovingAvg/decayÓ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_5_assignmovingavg_9067*
_output_shapes
:
*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpÝ
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/9067*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/subÔ
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/9067*
_output_shapes
:
2+
)batch_normalization_5/AssignMovingAvg/mul¯
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_5_assignmovingavg_9067-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_5/AssignMovingAvg/9067*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/9073*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_5/AssignMovingAvg_1/decayÙ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_1_9073*
_output_shapes
:
*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpç
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/9073*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/subÞ
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/9073*
_output_shapes
:
2-
+batch_normalization_5/AssignMovingAvg_1/mul»
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_1_9073/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg_1/9073*
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
%batch_normalization_5/batchnorm/add/yÚ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulÍ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/mul_1Ó
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ô
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpÙ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/add_1¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/TanhÝ

IdentityIdentitydense_11/Tanh:y:0:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2v
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
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8138

inputs
assignmovingavg_8113
assignmovingavg_1_8119)
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
:ÿÿÿÿÿÿÿÿÿ2
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8113*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8113*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8113*
_output_shapes
:2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8113*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8113AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8113*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8119*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8119*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8119*
_output_shapes
:2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8119*
_output_shapes
:2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8119AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8119*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8418

inputs
assignmovingavg_8393
assignmovingavg_1_8399)
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
:ÿÿÿÿÿÿÿÿÿ
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8393*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8393*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8393*
_output_shapes
:
2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8393*
_output_shapes
:
2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8393AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8393*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8399*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8399*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8399*
_output_shapes
:
2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8399*
_output_shapes
:
2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8399AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8399*
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9345

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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â	
Û
B__inference_dense_11_layer_call_and_return_conditional_losses_9586

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
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8451

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ë	
Ú
A__inference_dense_9_layer_call_and_return_conditional_losses_9382

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

+__inference_sequential_2_layer_call_fn_8924
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

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_88812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
Ö
{
&__inference_dense_9_layer_call_fn_9391

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_85392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
{
&__inference_dense_8_layer_call_fn_9289

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_84772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë	
Ú
A__inference_dense_8_layer_call_and_return_conditional_losses_8477

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
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á3
Û	
__inference__traced_save_9678
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop:
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
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop
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
SaveV2/shape_and_slicesï	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
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

identity_1Identity_1:output:0*¡
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
³
§
4__inference_batch_normalization_3_layer_call_fn_9371

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
X
Ì
 __inference__traced_restore_9748
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias2
.assignvariableop_2_batch_normalization_3_gamma1
-assignvariableop_3_batch_normalization_3_beta8
4assignvariableop_4_batch_normalization_3_moving_mean<
8assignvariableop_5_batch_normalization_3_moving_variance%
!assignvariableop_6_dense_9_kernel#
assignvariableop_7_dense_9_bias2
.assignvariableop_8_batch_normalization_4_gamma1
-assignvariableop_9_batch_normalization_4_beta9
5assignvariableop_10_batch_normalization_4_moving_mean=
9assignvariableop_11_batch_normalization_4_moving_variance'
#assignvariableop_12_dense_10_kernel%
!assignvariableop_13_dense_10_bias3
/assignvariableop_14_batch_normalization_5_gamma2
.assignvariableop_15_batch_normalization_5_beta9
5assignvariableop_16_batch_normalization_5_moving_mean=
9assignvariableop_17_batch_normalization_5_moving_variance'
#assignvariableop_18_dense_11_kernel%
!assignvariableop_19_dense_11_bias
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9 

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¬	
value¢	B	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¸
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

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16½
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Á
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_11_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_11_biasIdentity_19:output:0"/device:CPU:0*
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
±
§
4__inference_batch_normalization_4_layer_call_fn_9460

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_82782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
³
§
4__inference_batch_normalization_4_layer_call_fn_9473

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_83112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9529

inputs
assignmovingavg_9504
assignmovingavg_1_9510)
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
:ÿÿÿÿÿÿÿÿÿ
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9504*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
:
2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
:
2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9504AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9504*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9510*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes
:
2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9510*
_output_shapes
:
2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9510AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9510*
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9549

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
â	
Û
B__inference_dense_11_layer_call_and_return_conditional_losses_8663

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
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
±
§
4__inference_batch_normalization_5_layer_call_fn_9562

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì	
Û
B__inference_dense_10_layer_call_and_return_conditional_losses_8601

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8278

inputs
assignmovingavg_8253
assignmovingavg_1_8259)
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
:ÿÿÿÿÿÿÿÿÿ
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8253*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_8253*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8253*
_output_shapes
:
2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/8253*
_output_shapes
:
2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_8253AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/8253*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8259*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_8259*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8259*
_output_shapes
:
2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/8259*
_output_shapes
:
2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_8259AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/8259*
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ø
|
'__inference_dense_11_layer_call_fn_9595

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_86632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ü

__inference__wrapped_model_8042
dense_8_input7
3sequential_2_dense_8_matmul_readvariableop_resource8
4sequential_2_dense_8_biasadd_readvariableop_resourceH
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
4sequential_2_dense_11_matmul_readvariableop_resource9
5sequential_2_dense_11_biasadd_readvariableop_resource
identity¢;sequential_2/batch_normalization_3/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_4/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp¢;sequential_2/batch_normalization_5/batchnorm/ReadVariableOp¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1¢=sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2¢?sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp¢,sequential_2/dense_10/BiasAdd/ReadVariableOp¢+sequential_2/dense_10/MatMul/ReadVariableOp¢,sequential_2/dense_11/BiasAdd/ReadVariableOp¢+sequential_2/dense_11/MatMul/ReadVariableOp¢+sequential_2/dense_8/BiasAdd/ReadVariableOp¢*sequential_2/dense_8/MatMul/ReadVariableOp¢+sequential_2/dense_9/BiasAdd/ReadVariableOp¢*sequential_2/dense_9/MatMul/ReadVariableOpÌ
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_2/dense_8/MatMul/ReadVariableOp¹
sequential_2/dense_8/MatMulMatMuldense_8_input2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/MatMulË
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_8/BiasAdd/ReadVariableOpÕ
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/BiasAdd
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_8/Reluû
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
0sequential_2/batch_normalization_3/batchnorm/addÌ
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
:ÿÿÿÿÿÿÿÿÿ24
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
:ÿÿÿÿÿÿÿÿÿ24
2sequential_2/batch_normalization_3/batchnorm/add_1Ì
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*sequential_2/dense_9/MatMul/ReadVariableOpâ
sequential_2/dense_9/MatMulMatMul6sequential_2/batch_normalization_3/batchnorm/add_1:z:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/MatMulË
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_2/dense_9/BiasAdd/ReadVariableOpÕ
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/BiasAdd
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_9/Reluû
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
0sequential_2/batch_normalization_4/batchnorm/addÌ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_4/batchnorm/add_1Ï
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
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/MatMulÎ
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_2/dense_10/BiasAdd/ReadVariableOpÙ
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/BiasAdd
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_2/dense_10/Reluû
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
0sequential_2/batch_normalization_5/batchnorm/addÌ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
24
2sequential_2/batch_normalization_5/batchnorm/add_1Ï
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02-
+sequential_2/dense_11/MatMul/ReadVariableOpå
sequential_2/dense_11/MatMulMatMul6sequential_2/batch_normalization_5/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/MatMulÎ
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/dense_11/BiasAdd/ReadVariableOpÙ
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/BiasAdd
sequential_2/dense_11/TanhTanh&sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_11/Tanhâ	
IdentityIdentitysequential_2/dense_11/Tanh:y:0<^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_3/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_3/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_4/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_4/batchnorm/mul/ReadVariableOp<^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_1>^sequential_2/batch_normalization_5/batchnorm/ReadVariableOp_2@^sequential_2/batch_normalization_5/batchnorm/mul/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2z
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
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
¼+

F__inference_sequential_2_layer_call_and_return_conditional_losses_8731
dense_8_input
dense_8_8683
dense_8_8685
batch_normalization_3_8688
batch_normalization_3_8690
batch_normalization_3_8692
batch_normalization_3_8694
dense_9_8697
dense_9_8699
batch_normalization_4_8702
batch_normalization_4_8704
batch_normalization_4_8706
batch_normalization_4_8708
dense_10_8711
dense_10_8713
batch_normalization_5_8716
batch_normalization_5_8718
batch_normalization_5_8720
batch_normalization_5_8722
dense_11_8725
dense_11_8727
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_8683dense_8_8685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_84772!
dense_8/StatefulPartitionedCall­
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_8688batch_normalization_3_8690batch_normalization_3_8692batch_normalization_3_8694*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81712/
-batch_normalization_3/StatefulPartitionedCall¹
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_8697dense_9_8699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_85392!
dense_9/StatefulPartitionedCall­
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_8702batch_normalization_4_8704batch_normalization_4_8706batch_normalization_4_8708*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_83112/
-batch_normalization_4/StatefulPartitionedCall¾
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_8711dense_10_8713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_86012"
 dense_10/StatefulPartitionedCall®
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_8716batch_normalization_5_8718batch_normalization_5_8720batch_normalization_5_8722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84512/
-batch_normalization_5/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_8725dense_11_8727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_86632"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
³

+__inference_sequential_2_layer_call_fn_9269

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
identity¢StatefulPartitionedCallé
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
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_88812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
|
'__inference_dense_10_layer_call_fn_9493

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_86012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8311

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
§+

F__inference_sequential_2_layer_call_and_return_conditional_losses_8881

inputs
dense_8_8833
dense_8_8835
batch_normalization_3_8838
batch_normalization_3_8840
batch_normalization_3_8842
batch_normalization_3_8844
dense_9_8847
dense_9_8849
batch_normalization_4_8852
batch_normalization_4_8854
batch_normalization_4_8856
batch_normalization_4_8858
dense_10_8861
dense_10_8863
batch_normalization_5_8866
batch_normalization_5_8868
batch_normalization_5_8870
batch_normalization_5_8872
dense_11_8875
dense_11_8877
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_8833dense_8_8835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_84772!
dense_8/StatefulPartitionedCall­
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_8838batch_normalization_3_8840batch_normalization_3_8842batch_normalization_3_8844*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81712/
-batch_normalization_3/StatefulPartitionedCall¹
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_8847dense_9_8849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_85392!
dense_9/StatefulPartitionedCall­
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_8852batch_normalization_4_8854batch_normalization_4_8856batch_normalization_4_8858*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_83112/
-batch_normalization_4/StatefulPartitionedCall¾
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_8861dense_10_8863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_86012"
 dense_10/StatefulPartitionedCall®
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_8866batch_normalization_5_8868batch_normalization_5_8870batch_normalization_5_8872*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84512/
-batch_normalization_5/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_8875dense_11_8877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_86632"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


"__inference_signature_wrapper_8971
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

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_80422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
ë	
Ú
A__inference_dense_8_layer_call_and_return_conditional_losses_9280

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
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
§
4__inference_batch_normalization_5_layer_call_fn_9575

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9325

inputs
assignmovingavg_9300
assignmovingavg_1_9306)
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
:ÿÿÿÿÿÿÿÿÿ2
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9300*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9300*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9300*
_output_shapes
:2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9300*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9300AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9300*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9306*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9306*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9306*
_output_shapes
:2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9306*
_output_shapes
:2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9306AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9306*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é/
Â
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9427

inputs
assignmovingavg_9402
assignmovingavg_1_9408)
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
:ÿÿÿÿÿÿÿÿÿ
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
moments/Squeeze_1Ê
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9402*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9402*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOpï
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9402*
_output_shapes
:
2
AssignMovingAvg/subæ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/9402*
_output_shapes
:
2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9402AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/9402*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÐ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9408*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9408*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOpù
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9408*
_output_shapes
:
2
AssignMovingAvg_1/subð
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/9408*
_output_shapes
:
2
AssignMovingAvg_1/mul·
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9408AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/9408*
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
À{
­
F__inference_sequential_2_layer_call_and_return_conditional_losses_9179

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource;
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
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity¢.batch_normalization_3/batchnorm/ReadVariableOp¢0batch_normalization_3/batchnorm/ReadVariableOp_1¢0batch_normalization_3/batchnorm/ReadVariableOp_2¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/ReluÔ
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
%batch_normalization_3/batchnorm/add/yà
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/add¥
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtà
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mulÌ
%batch_normalization_3/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Ú
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ý
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2Ú
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2Û
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subÝ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp®
dense_9/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9/ReluÔ
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
%batch_normalization_4/batchnorm/add/yà
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/mulÌ
%batch_normalization_4/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/mul_1Ú
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ý
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_4/batchnorm/mul_2Ú
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Û
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_4/batchnorm/add_1¨
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
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/MatMul§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOp¥
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/BiasAdds
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_10/ReluÔ
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
%batch_normalization_5/batchnorm/add/yà
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/mulÍ
%batch_normalization_5/batchnorm/mul_1Muldense_10/Relu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/mul_1Ú
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ý
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_5/batchnorm/mul_2Ú
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Û
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%batch_normalization_5/batchnorm/add_1¨
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_11/MatMul/ReadVariableOp±
dense_11/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¥
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/TanhÑ
IdentityIdentitydense_11/Tanh:y:0/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2`
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
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡+

F__inference_sequential_2_layer_call_and_return_conditional_losses_8785

inputs
dense_8_8737
dense_8_8739
batch_normalization_3_8742
batch_normalization_3_8744
batch_normalization_3_8746
batch_normalization_3_8748
dense_9_8751
dense_9_8753
batch_normalization_4_8756
batch_normalization_4_8758
batch_normalization_4_8760
batch_normalization_4_8762
dense_10_8765
dense_10_8767
batch_normalization_5_8770
batch_normalization_5_8772
batch_normalization_5_8774
batch_normalization_5_8776
dense_11_8779
dense_11_8781
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_8737dense_8_8739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_84772!
dense_8/StatefulPartitionedCall«
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_8742batch_normalization_3_8744batch_normalization_3_8746batch_normalization_3_8748*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81382/
-batch_normalization_3/StatefulPartitionedCall¹
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_8751dense_9_8753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_85392!
dense_9/StatefulPartitionedCall«
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_8756batch_normalization_4_8758batch_normalization_4_8760batch_normalization_4_8762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_82782/
-batch_normalization_4/StatefulPartitionedCall¾
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_8765dense_10_8767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_86012"
 dense_10/StatefulPartitionedCall¬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_8770batch_normalization_5_8772batch_normalization_5_8774batch_normalization_5_8776*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84182/
-batch_normalization_5/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_8779dense_11_8781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_86632"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë	
Ú
A__inference_dense_9_layer_call_and_return_conditional_losses_8539

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

+__inference_sequential_2_layer_call_fn_8828
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

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_87852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
Ò

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8171

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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶+

F__inference_sequential_2_layer_call_and_return_conditional_losses_8680
dense_8_input
dense_8_8488
dense_8_8490
batch_normalization_3_8519
batch_normalization_3_8521
batch_normalization_3_8523
batch_normalization_3_8525
dense_9_8550
dense_9_8552
batch_normalization_4_8581
batch_normalization_4_8583
batch_normalization_4_8585
batch_normalization_4_8587
dense_10_8612
dense_10_8614
batch_normalization_5_8643
batch_normalization_5_8645
batch_normalization_5_8647
batch_normalization_5_8649
dense_11_8674
dense_11_8676
identity¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_8488dense_8_8490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_84772!
dense_8/StatefulPartitionedCall«
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_3_8519batch_normalization_3_8521batch_normalization_3_8523batch_normalization_3_8525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81382/
-batch_normalization_3/StatefulPartitionedCall¹
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_9_8550dense_9_8552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_85392!
dense_9/StatefulPartitionedCall«
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_4_8581batch_normalization_4_8583batch_normalization_4_8585batch_normalization_4_8587*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_82782/
-batch_normalization_4/StatefulPartitionedCall¾
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_10_8612dense_10_8614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_86012"
 dense_10/StatefulPartitionedCall¬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_5_8643batch_normalization_5_8645batch_normalization_5_8647batch_normalization_5_8649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_84182/
-batch_normalization_5/StatefulPartitionedCall¾
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0dense_11_8674dense_11_8676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_86632"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_8_input
Ò

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9447

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì	
Û
B__inference_dense_10_layer_call_and_return_conditional_losses_9484

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
­

+__inference_sequential_2_layer_call_fn_9224

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
identity¢StatefulPartitionedCallã
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_87852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*v
_input_shapese
c:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
§
4__inference_batch_normalization_3_layer_call_fn_9358

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_81382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
G
dense_8_input6
serving_default_dense_8_input:0ÿÿÿÿÿÿÿÿÿ<
dense_110
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:èü
ÃA
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
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
*h&call_and_return_all_conditional_losses
i_default_save_signature
j__call__"ö=
_tf_keras_sequential×={"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ß

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*k&call_and_return_all_conditional_losses
l__call__"º
_tf_keras_layer {"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
²	
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
*m&call_and_return_all_conditional_losses
n__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
ð

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
*o&call_and_return_all_conditional_losses
p__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
²	
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'	variables
(trainable_variables
)regularization_losses
*	keras_api
*q&call_and_return_all_conditional_losses
r__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ò

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
*s&call_and_return_all_conditional_losses
t__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
²	
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6	variables
7trainable_variables
8regularization_losses
9	keras_api
*u&call_and_return_all_conditional_losses
v__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ñ

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
¶
0
1
2
3
4
5
6
7
#8
$9
%10
&11
+12
,13
214
315
416
517
:18
;19"
trackable_list_wrapper

0
1
2
3
4
5
#6
$7
+8
,9
210
311
:12
;13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
@metrics

Alayers
	variables
	trainable_variables
Blayer_metrics

regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
j__call__
i_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
,
yserving_default"
signature_map
 :2dense_8/kernel
:2dense_8/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Emetrics

Flayers
	variables
trainable_variables
Glayer_metrics
regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jmetrics

Klayers
	variables
trainable_variables
Llayer_metrics
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_9/kernel
:
2dense_9/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Ometrics

Players
	variables
trainable_variables
Qlayer_metrics
 regularization_losses
Rlayer_regularization_losses
Snon_trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
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
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tmetrics

Ulayers
'	variables
(trainable_variables
Vlayer_metrics
)regularization_losses
Wlayer_regularization_losses
Xnon_trainable_variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_10/kernel
:
2dense_10/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Ymetrics

Zlayers
-	variables
.trainable_variables
[layer_metrics
/regularization_losses
\layer_regularization_losses
]non_trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
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
<
20
31
42
53"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
­
^metrics

_layers
6	variables
7trainable_variables
`layer_metrics
8regularization_losses
alayer_regularization_losses
bnon_trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_11/kernel
:2dense_11/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cmetrics

dlayers
<	variables
=trainable_variables
elayer_metrics
>regularization_losses
flayer_regularization_losses
gnon_trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
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
trackable_dict_wrapper
 "
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
0
1"
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
%0
&1"
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
40
51"
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
æ2ã
F__inference_sequential_2_layer_call_and_return_conditional_losses_9179
F__inference_sequential_2_layer_call_and_return_conditional_losses_9099
F__inference_sequential_2_layer_call_and_return_conditional_losses_8731
F__inference_sequential_2_layer_call_and_return_conditional_losses_8680À
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
ã2à
__inference__wrapped_model_8042¼
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
annotationsª *,¢)
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
ú2÷
+__inference_sequential_2_layer_call_fn_8828
+__inference_sequential_2_layer_call_fn_8924
+__inference_sequential_2_layer_call_fn_9224
+__inference_sequential_2_layer_call_fn_9269À
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
ë2è
A__inference_dense_8_layer_call_and_return_conditional_losses_9280¢
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
Ð2Í
&__inference_dense_8_layer_call_fn_9289¢
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
Ü2Ù
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9325
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9345´
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
¦2£
4__inference_batch_normalization_3_layer_call_fn_9358
4__inference_batch_normalization_3_layer_call_fn_9371´
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
ë2è
A__inference_dense_9_layer_call_and_return_conditional_losses_9382¢
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
Ð2Í
&__inference_dense_9_layer_call_fn_9391¢
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
Ü2Ù
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9447
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9427´
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
¦2£
4__inference_batch_normalization_4_layer_call_fn_9460
4__inference_batch_normalization_4_layer_call_fn_9473´
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
ì2é
B__inference_dense_10_layer_call_and_return_conditional_losses_9484¢
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
Ñ2Î
'__inference_dense_10_layer_call_fn_9493¢
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
Ü2Ù
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9549
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9529´
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
¦2£
4__inference_batch_normalization_5_layer_call_fn_9575
4__inference_batch_normalization_5_layer_call_fn_9562´
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
ì2é
B__inference_dense_11_layer_call_and_return_conditional_losses_9586¢
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
Ñ2Î
'__inference_dense_11_layer_call_fn_9595¢
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
ÏBÌ
"__inference_signature_wrapper_8971dense_8_input"
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
 §
__inference__wrapped_model_8042&#%$+,5243:;6¢3
,¢)
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_11"
dense_11ÿÿÿÿÿÿÿÿÿµ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9325b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9345b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_batch_normalization_3_layer_call_fn_9358U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_batch_normalization_3_layer_call_fn_9371U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿµ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9427b%&#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9447b&#%$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
4__inference_batch_normalization_4_layer_call_fn_9460U%&#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ

4__inference_batch_normalization_4_layer_call_fn_9473U&#%$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
µ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9529b45233¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9549b52433¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
4__inference_batch_normalization_5_layer_call_fn_9562U45233¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ

4__inference_batch_normalization_5_layer_call_fn_9575U52433¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¢
B__inference_dense_10_layer_call_and_return_conditional_losses_9484\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_dense_10_layer_call_fn_9493O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
¢
B__inference_dense_11_layer_call_and_return_conditional_losses_9586\:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_11_layer_call_fn_9595O:;/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_8_layer_call_and_return_conditional_losses_9280\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_8_layer_call_fn_9289O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_9_layer_call_and_return_conditional_losses_9382\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 y
&__inference_dense_9_layer_call_fn_9391O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
Ç
F__inference_sequential_2_layer_call_and_return_conditional_losses_8680}%&#$+,4523:;>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
F__inference_sequential_2_layer_call_and_return_conditional_losses_8731}&#%$+,5243:;>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
F__inference_sequential_2_layer_call_and_return_conditional_losses_9099v%&#$+,4523:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
F__inference_sequential_2_layer_call_and_return_conditional_losses_9179v&#%$+,5243:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_2_layer_call_fn_8828p%&#$+,4523:;>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_2_layer_call_fn_8924p&#%$+,5243:;>¢;
4¢1
'$
dense_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_2_layer_call_fn_9224i%&#$+,4523:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_2_layer_call_fn_9269i&#%$+,5243:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ»
"__inference_signature_wrapper_8971&#%$+,5243:;G¢D
¢ 
=ª:
8
dense_8_input'$
dense_8_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_11"
dense_11ÿÿÿÿÿÿÿÿÿ