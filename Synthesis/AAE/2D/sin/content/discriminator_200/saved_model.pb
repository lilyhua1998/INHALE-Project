»
¿
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
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
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
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8î
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	2*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:2*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:2*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:2*
dtype0
¥
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:2*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P * 
shared_nameconv1d_2/kernel
x
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*#
_output_shapes
:P *
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
: *
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
: *
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
: *
dtype0
¥
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
: *
dtype0

conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: P* 
shared_nameconv1d_3/kernel
x
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*#
_output_shapes
: P*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:P*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:P*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:P*
dtype0
¤
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:P*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	2*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Nadam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*'
shared_nameNadam/dense_9/kernel/m

*Nadam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/m*
_output_shapes
:	2*
dtype0
¡
$Nadam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Nadam/batch_normalization_11/gamma/m

8Nadam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_11/gamma/m*
_output_shapes	
:2*
dtype0

#Nadam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Nadam/batch_normalization_11/beta/m

7Nadam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_11/beta/m*
_output_shapes	
:2*
dtype0

Nadam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P *(
shared_nameNadam/conv1d_2/kernel/m

+Nadam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_2/kernel/m*#
_output_shapes
:P *
dtype0
¡
$Nadam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Nadam/batch_normalization_12/gamma/m

8Nadam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_12/gamma/m*
_output_shapes	
: *
dtype0

#Nadam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Nadam/batch_normalization_12/beta/m

7Nadam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_12/beta/m*
_output_shapes	
: *
dtype0

Nadam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: P*(
shared_nameNadam/conv1d_3/kernel/m

+Nadam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_3/kernel/m*#
_output_shapes
: P*
dtype0
 
$Nadam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Nadam/batch_normalization_13/gamma/m

8Nadam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_13/gamma/m*
_output_shapes
:P*
dtype0

#Nadam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Nadam/batch_normalization_13/beta/m

7Nadam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_13/beta/m*
_output_shapes
:P*
dtype0

Nadam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*(
shared_nameNadam/dense_10/kernel/m

+Nadam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_10/kernel/m*
_output_shapes
:	2*
dtype0

Nadam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*'
shared_nameNadam/dense_9/kernel/v

*Nadam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9/kernel/v*
_output_shapes
:	2*
dtype0
¡
$Nadam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*5
shared_name&$Nadam/batch_normalization_11/gamma/v

8Nadam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_11/gamma/v*
_output_shapes	
:2*
dtype0

#Nadam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#Nadam/batch_normalization_11/beta/v

7Nadam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_11/beta/v*
_output_shapes	
:2*
dtype0

Nadam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P *(
shared_nameNadam/conv1d_2/kernel/v

+Nadam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_2/kernel/v*#
_output_shapes
:P *
dtype0
¡
$Nadam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Nadam/batch_normalization_12/gamma/v

8Nadam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_12/gamma/v*
_output_shapes	
: *
dtype0

#Nadam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Nadam/batch_normalization_12/beta/v

7Nadam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_12/beta/v*
_output_shapes	
: *
dtype0

Nadam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: P*(
shared_nameNadam/conv1d_3/kernel/v

+Nadam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_3/kernel/v*#
_output_shapes
: P*
dtype0
 
$Nadam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$Nadam/batch_normalization_13/gamma/v

8Nadam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp$Nadam/batch_normalization_13/gamma/v*
_output_shapes
:P*
dtype0

#Nadam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#Nadam/batch_normalization_13/beta/v

7Nadam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_13/beta/v*
_output_shapes
:P*
dtype0

Nadam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*(
shared_nameNadam/dense_10/kernel/v

+Nadam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_10/kernel/v*
_output_shapes
:	2*
dtype0

NoOpNoOp
­Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*èP
valueÞPBÛP BÔP
Ó
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
layer-11
layer_with_weights-6
layer-12
	optimizer
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

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
^

*kernel
+regularization_losses
,trainable_variables
-	variables
.	keras_api

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
^

<kernel
=regularization_losses
>trainable_variables
?	variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
^

Rkernel
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api

Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\momentum_cachem£m¤m¥*m¦0m§1m¨<m©BmªCm«Rm¬v­v®v¯*v°0v±1v²<v³Bv´CvµRv¶
 
F
0
1
2
*3
04
15
<6
B7
C8
R9
v
0
1
2
3
4
*5
06
17
28
39
<10
B11
C12
D13
E14
R15
­
regularization_losses
]metrics
trainable_variables
^non_trainable_variables
	variables

_layers
`layer_regularization_losses
alayer_metrics
 
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses
bmetrics
trainable_variables
cnon_trainable_variables
	variables

dlayers
elayer_regularization_losses
flayer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
­
regularization_losses
gmetrics
trainable_variables
hnon_trainable_variables
 	variables

ilayers
jlayer_regularization_losses
klayer_metrics
 
 
 
­
"regularization_losses
lmetrics
#trainable_variables
mnon_trainable_variables
$	variables

nlayers
olayer_regularization_losses
player_metrics
 
 
 
­
&regularization_losses
qmetrics
'trainable_variables
rnon_trainable_variables
(	variables

slayers
tlayer_regularization_losses
ulayer_metrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0
­
+regularization_losses
vmetrics
,trainable_variables
wnon_trainable_variables
-	variables

xlayers
ylayer_regularization_losses
zlayer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
22
33
­
4regularization_losses
{metrics
5trainable_variables
|non_trainable_variables
6	variables

}layers
~layer_regularization_losses
layer_metrics
 
 
 
²
8regularization_losses
metrics
9trainable_variables
non_trainable_variables
:	variables
layers
 layer_regularization_losses
layer_metrics
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

<0

<0
²
=regularization_losses
metrics
>trainable_variables
non_trainable_variables
?	variables
layers
 layer_regularization_losses
layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
D2
E3
²
Fregularization_losses
metrics
Gtrainable_variables
non_trainable_variables
H	variables
layers
 layer_regularization_losses
layer_metrics
 
 
 
²
Jregularization_losses
metrics
Ktrainable_variables
non_trainable_variables
L	variables
layers
 layer_regularization_losses
layer_metrics
 
 
 
²
Nregularization_losses
metrics
Otrainable_variables
non_trainable_variables
P	variables
layers
 layer_regularization_losses
layer_metrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

R0

R0
²
Sregularization_losses
metrics
Ttrainable_variables
non_trainable_variables
U	variables
layers
 layer_regularization_losses
layer_metrics
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE

0
*
0
1
22
33
D4
E5
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

0
1
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
20
31
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
8

total

 count
¡	variables
¢	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
 1

¡	variables
~|
VARIABLE_VALUENadam/dense_9/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_11/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_11/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_12/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_12/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_13/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_13/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_10/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_9/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_11/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_11/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_12/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_12/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv1d_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Nadam/batch_normalization_13/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_13/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_10/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_4Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
«
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_9/kernel&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betaconv1d_2/kernel&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betaconv1d_3/kernel&batch_normalization_13/moving_variancebatch_normalization_13/gamma"batch_normalization_13/moving_meanbatch_normalization_13/betadense_10/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1076780
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Nadam/dense_9/kernel/m/Read/ReadVariableOp8Nadam/batch_normalization_11/gamma/m/Read/ReadVariableOp7Nadam/batch_normalization_11/beta/m/Read/ReadVariableOp+Nadam/conv1d_2/kernel/m/Read/ReadVariableOp8Nadam/batch_normalization_12/gamma/m/Read/ReadVariableOp7Nadam/batch_normalization_12/beta/m/Read/ReadVariableOp+Nadam/conv1d_3/kernel/m/Read/ReadVariableOp8Nadam/batch_normalization_13/gamma/m/Read/ReadVariableOp7Nadam/batch_normalization_13/beta/m/Read/ReadVariableOp+Nadam/dense_10/kernel/m/Read/ReadVariableOp*Nadam/dense_9/kernel/v/Read/ReadVariableOp8Nadam/batch_normalization_11/gamma/v/Read/ReadVariableOp7Nadam/batch_normalization_11/beta/v/Read/ReadVariableOp+Nadam/conv1d_2/kernel/v/Read/ReadVariableOp8Nadam/batch_normalization_12/gamma/v/Read/ReadVariableOp7Nadam/batch_normalization_12/beta/v/Read/ReadVariableOp+Nadam/conv1d_3/kernel/v/Read/ReadVariableOp8Nadam/batch_normalization_13/gamma/v/Read/ReadVariableOp7Nadam/batch_normalization_13/beta/v/Read/ReadVariableOp+Nadam/dense_10/kernel/v/Read/ReadVariableOpConst*9
Tin2
02.	*
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
 __inference__traced_save_1077771

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kernelbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv1d_2/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv1d_3/kernelbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancedense_10/kernel
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense_9/kernel/m$Nadam/batch_normalization_11/gamma/m#Nadam/batch_normalization_11/beta/mNadam/conv1d_2/kernel/m$Nadam/batch_normalization_12/gamma/m#Nadam/batch_normalization_12/beta/mNadam/conv1d_3/kernel/m$Nadam/batch_normalization_13/gamma/m#Nadam/batch_normalization_13/beta/mNadam/dense_10/kernel/mNadam/dense_9/kernel/v$Nadam/batch_normalization_11/gamma/v#Nadam/batch_normalization_11/beta/vNadam/conv1d_2/kernel/v$Nadam/batch_normalization_12/gamma/v#Nadam/batch_normalization_12/beta/vNadam/conv1d_3/kernel/v$Nadam/batch_normalization_13/gamma/v#Nadam/batch_normalization_13/beta/vNadam/dense_10/kernel/v*8
Tin1
/2-*
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
#__inference__traced_restore_1077913±¹
ñ
«
8__inference_batch_normalization_12_layer_call_fn_1077292

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_10759472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

E
)__inference_re_lu_6_layer_call_fn_1077186

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_10761942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ó
 
E__inference_dense_10_layer_call_and_return_conditional_losses_1076501

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidw
IdentityIdentitySigmoid:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077472

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
1
Ì
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077452

inputs
assignmovingavg_1077427
assignmovingavg_1_1077433)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:P2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077427*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077427*
_output_shapes
:P*
dtype02 
AssignMovingAvg/ReadVariableOpò
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077427*
_output_shapes
:P2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077427*
_output_shapes
:P2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077427AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077427*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077433*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077433*
_output_shapes
:P*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077433*
_output_shapes
:P2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077433*
_output_shapes
:P2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077433AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077433*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
¶
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_1077181

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Æ
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_1077392

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
ø

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1076302

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
:

J__inference_Discriminator_layer_call_and_return_conditional_losses_1076613

inputs
dense_9_1076568"
batch_normalization_11_1076571"
batch_normalization_11_1076573"
batch_normalization_11_1076575"
batch_normalization_11_1076577
conv1d_2_1076582"
batch_normalization_12_1076585"
batch_normalization_12_1076587"
batch_normalization_12_1076589"
batch_normalization_12_1076591
conv1d_3_1076595"
batch_normalization_13_1076598"
batch_normalization_13_1076600"
batch_normalization_13_1076602"
batch_normalization_13_1076604
dense_10_1076609
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_1076568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_10761422!
dense_9/StatefulPartitionedCallÂ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_11_1076571batch_normalization_11_1076573batch_normalization_11_1076575batch_normalization_11_1076577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_107580720
.batch_normalization_11/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_10761942
re_lu_6/PartitionedCallö
reshape_2/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_10762152
reshape_2/PartitionedCall¤
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_1076582*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_10762352"
 conv1d_2/StatefulPartitionedCallÇ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_1076585batch_normalization_12_1076587batch_normalization_12_1076589batch_normalization_12_1076591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_107628220
.batch_normalization_12/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_10763432
re_lu_7/PartitionedCall¡
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_3_1076595*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_10763632"
 conv1d_3/StatefulPartitionedCallÆ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_13_1076598batch_normalization_13_1076600batch_normalization_13_1076602batch_normalization_13_1076604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_107641020
.batch_normalization_13/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_10764712
re_lu_8/PartitionedCalló
flatten_2/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_10764852
flatten_2/PartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_10_1076609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_10765012"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ì
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1076087

inputs
assignmovingavg_1076062
assignmovingavg_1_1076068)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:P2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076062*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1076062*
_output_shapes
:P*
dtype02 
AssignMovingAvg/ReadVariableOpò
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076062*
_output_shapes
:P2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076062*
_output_shapes
:P2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1076062AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076062*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076068*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1076068*
_output_shapes
:P*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076068*
_output_shapes
:P2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076068*
_output_shapes
:P2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1076068AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076068*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
Ð
p
*__inference_conv1d_3_layer_call_fn_1077416

inputs
unknown
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_10763632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
¢
E
)__inference_re_lu_8_layer_call_fn_1077590

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_10764712
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
Æ
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_1076343

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
­

Û
/__inference_Discriminator_layer_call_fn_1077043

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
identity¢StatefulPartitionedCall¯
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Discriminator_layer_call_and_return_conditional_losses_10766132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_1076194

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ø

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077361

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/add_1à
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
Í
«
8__inference_batch_normalization_12_layer_call_fn_1077374

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_10762822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
:

J__inference_Discriminator_layer_call_and_return_conditional_losses_1076562
input_4
dense_9_1076517"
batch_normalization_11_1076520"
batch_normalization_11_1076522"
batch_normalization_11_1076524"
batch_normalization_11_1076526
conv1d_2_1076531"
batch_normalization_12_1076534"
batch_normalization_12_1076536"
batch_normalization_12_1076538"
batch_normalization_12_1076540
conv1d_3_1076544"
batch_normalization_13_1076547"
batch_normalization_13_1076549"
batch_normalization_13_1076551"
batch_normalization_13_1076553
dense_10_1076558
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_9_1076517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_10761422!
dense_9/StatefulPartitionedCallÄ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_11_1076520batch_normalization_11_1076522batch_normalization_11_1076524batch_normalization_11_1076526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_107584020
.batch_normalization_11/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_10761942
re_lu_6/PartitionedCallö
reshape_2/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_10762152
reshape_2/PartitionedCall¤
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_1076531*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_10762352"
 conv1d_2/StatefulPartitionedCallÉ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_1076534batch_normalization_12_1076536batch_normalization_12_1076538batch_normalization_12_1076540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_107630220
.batch_normalization_12/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_10763432
re_lu_7/PartitionedCall¡
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_3_1076544*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_10763632"
 conv1d_3/StatefulPartitionedCallÈ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_13_1076547batch_normalization_13_1076549batch_normalization_13_1076551batch_normalization_13_1076553*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_107643020
.batch_normalization_13/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_10764712
re_lu_8/PartitionedCalló
flatten_2/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_10764852
flatten_2/PartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_10_1076558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_10765012"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
:

J__inference_Discriminator_layer_call_and_return_conditional_losses_1076698

inputs
dense_9_1076653"
batch_normalization_11_1076656"
batch_normalization_11_1076658"
batch_normalization_11_1076660"
batch_normalization_11_1076662
conv1d_2_1076667"
batch_normalization_12_1076670"
batch_normalization_12_1076672"
batch_normalization_12_1076674"
batch_normalization_12_1076676
conv1d_3_1076680"
batch_normalization_13_1076683"
batch_normalization_13_1076685"
batch_normalization_13_1076687"
batch_normalization_13_1076689
dense_10_1076694
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_1076653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_10761422!
dense_9/StatefulPartitionedCallÄ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_11_1076656batch_normalization_11_1076658batch_normalization_11_1076660batch_normalization_11_1076662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_107584020
.batch_normalization_11/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_10761942
re_lu_6/PartitionedCallö
reshape_2/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_10762152
reshape_2/PartitionedCall¤
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_1076667*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_10762352"
 conv1d_2/StatefulPartitionedCallÉ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_1076670batch_normalization_12_1076672batch_normalization_12_1076674batch_normalization_12_1076676*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_107630220
.batch_normalization_12/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_10763432
re_lu_7/PartitionedCall¡
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_3_1076680*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_10763632"
 conv1d_3/StatefulPartitionedCallÈ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_13_1076683batch_normalization_13_1076685batch_normalization_13_1076687batch_normalization_13_1076689*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_107643020
.batch_normalization_13/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_10764712
re_lu_8/PartitionedCalló
flatten_2/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_10764852
flatten_2/PartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_10_1076694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_10765012"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

Û
/__inference_Discriminator_layer_call_fn_1077080

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
identity¢StatefulPartitionedCallµ
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
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Discriminator_layer_call_and_return_conditional_losses_10766982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
 
E__inference_dense_10_layer_call_and_return_conditional_losses_1077609

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMula
SigmoidSigmoidMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidw
IdentityIdentitySigmoid:y:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ï
«
8__inference_batch_normalization_13_layer_call_fn_1077498

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_10761202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
ó
«
8__inference_batch_normalization_12_layer_call_fn_1077305

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_10759802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

º
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1076363

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: P*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: P2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
â

J__inference_Discriminator_layer_call_and_return_conditional_losses_1076917

inputs*
&dense_9_matmul_readvariableop_resource2
.batch_normalization_11_assignmovingavg_10767944
0batch_normalization_11_assignmovingavg_1_1076800@
<batch_normalization_11_batchnorm_mul_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource2
.batch_normalization_12_assignmovingavg_10768444
0batch_normalization_12_assignmovingavg_1_1076850@
<batch_normalization_12_batchnorm_mul_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource2
.batch_normalization_13_assignmovingavg_10768854
0batch_normalization_13_assignmovingavg_1_1076891@
<batch_normalization_13_batchnorm_mul_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource+
'dense_10_matmul_readvariableop_resource
identity¢:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_11/AssignMovingAvg/ReadVariableOp¢<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_11/batchnorm/ReadVariableOp¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_12/AssignMovingAvg/ReadVariableOp¢<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_12/batchnorm/ReadVariableOp¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_13/AssignMovingAvg/ReadVariableOp¢<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¦
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_9/MatMul¸
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_11/moments/mean/reduction_indicesç
#batch_normalization_11/moments/meanMeandense_9/MatMul:product:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2%
#batch_normalization_11/moments/meanÂ
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:	22-
+batch_normalization_11/moments/StopGradientü
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_9/MatMul:product:04batch_normalization_11/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ222
0batch_normalization_11/moments/SquaredDifferenceÀ
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_11/moments/variance/reduction_indices
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2)
'batch_normalization_11/moments/varianceÆ
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2(
&batch_normalization_11/moments/SqueezeÎ
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1
,batch_normalization_11/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/1076794*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_11/AssignMovingAvg/decayÚ
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_11_assignmovingavg_1076794*
_output_shapes	
:2*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/1076794*
_output_shapes	
:22,
*batch_normalization_11/AssignMovingAvg/subÝ
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/1076794*
_output_shapes	
:22,
*batch_normalization_11/AssignMovingAvg/mul»
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_11_assignmovingavg_1076794.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_11/AssignMovingAvg/1076794*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_11/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/1076800*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_11/AssignMovingAvg_1/decayà
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_11_assignmovingavg_1_1076800*
_output_shapes	
:2*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/1076800*
_output_shapes	
:22.
,batch_normalization_11/AssignMovingAvg_1/subç
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/1076800*
_output_shapes	
:22.
,batch_normalization_11/AssignMovingAvg_1/mulÇ
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_11_assignmovingavg_1_10768000batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_11/AssignMovingAvg_1/1076800*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_11/batchnorm/add/yß
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/add©
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:22(
&batch_normalization_11/batchnorm/Rsqrtä
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOpâ
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/mulÎ
&batch_normalization_11/batchnorm/mul_1Muldense_9/MatMul:product:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22(
&batch_normalization_11/batchnorm/mul_1Ø
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:22(
&batch_normalization_11/batchnorm/mul_2Ø
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype021
/batch_normalization_11/batchnorm/ReadVariableOpÞ
$batch_normalization_11/batchnorm/subSub7batch_normalization_11/batchnorm/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/subâ
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22(
&batch_normalization_11/batchnorm/add_1
re_lu_6/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
re_lu_6/Relul
reshape_2/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2
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
value	B :P2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :P2
reshape_2/Reshape/shape/2Ò
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape¥
reshape_2/ReshapeReshapere_lu_6/Relu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
reshape_2/Reshape
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÅ
conv1d_2/conv1d/ExpandDims
ExpandDimsreshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
conv1d_2/conv1d/ExpandDimsÔ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:P *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÜ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P 2
conv1d_2/conv1d/ExpandDims_1Û
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
paddingSAME*
strides
2
conv1d_2/conv1d®
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/Squeeze¿
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_12/moments/mean/reduction_indicesó
#batch_normalization_12/moments/meanMean conv1d_2/conv1d/Squeeze:output:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2%
#batch_normalization_12/moments/meanÆ
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*#
_output_shapes
: 2-
+batch_normalization_12/moments/StopGradient
0batch_normalization_12/moments/SquaredDifferenceSquaredDifference conv1d_2/conv1d/Squeeze:output:04batch_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 22
0batch_normalization_12/moments/SquaredDifferenceÇ
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_12/moments/variance/reduction_indices
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2)
'batch_normalization_12/moments/varianceÇ
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2(
&batch_normalization_12/moments/SqueezeÏ
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1
,batch_normalization_12/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/1076844*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_12/AssignMovingAvg/decayÚ
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_12_assignmovingavg_1076844*
_output_shapes	
: *
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/1076844*
_output_shapes	
: 2,
*batch_normalization_12/AssignMovingAvg/subÝ
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/1076844*
_output_shapes	
: 2,
*batch_normalization_12/AssignMovingAvg/mul»
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_12_assignmovingavg_1076844.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_12/AssignMovingAvg/1076844*
_output_shapes
 *
dtype02<
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_12/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/1076850*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_12/AssignMovingAvg_1/decayà
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_12_assignmovingavg_1_1076850*
_output_shapes	
: *
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/1076850*
_output_shapes	
: 2.
,batch_normalization_12/AssignMovingAvg_1/subç
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/1076850*
_output_shapes	
: 2.
,batch_normalization_12/AssignMovingAvg_1/mulÇ
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_12_assignmovingavg_1_10768500batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_12/AssignMovingAvg_1/1076850*
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
&batch_normalization_12/batchnorm/add/yß
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/add©
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
: 2(
&batch_normalization_12/batchnorm/Rsqrtä
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpâ
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/mulÚ
&batch_normalization_12/batchnorm/mul_1Mul conv1d_2/conv1d/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2(
&batch_normalization_12/batchnorm/mul_1Ø
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
: 2(
&batch_normalization_12/batchnorm/mul_2Ø
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype021
/batch_normalization_12/batchnorm/ReadVariableOpÞ
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/subæ
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2(
&batch_normalization_12/batchnorm/add_1
re_lu_7/ReluRelu*batch_normalization_12/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
re_lu_7/Relu
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_3/conv1d/ExpandDims/dimÆ
conv1d_3/conv1d/ExpandDims
ExpandDimsre_lu_7/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
conv1d_3/conv1d/ExpandDimsÔ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: P*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÜ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: P2
conv1d_3/conv1d/ExpandDims_1Ú
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
paddingSAME*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_3/conv1d/Squeeze¿
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_13/moments/mean/reduction_indicesò
#batch_normalization_13/moments/meanMean conv1d_3/conv1d/Squeeze:output:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2%
#batch_normalization_13/moments/meanÅ
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*"
_output_shapes
:P2-
+batch_normalization_13/moments/StopGradient
0batch_normalization_13/moments/SquaredDifferenceSquaredDifference conv1d_3/conv1d/Squeeze:output:04batch_normalization_13/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP22
0batch_normalization_13/moments/SquaredDifferenceÇ
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_13/moments/variance/reduction_indices
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2)
'batch_normalization_13/moments/varianceÆ
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2(
&batch_normalization_13/moments/SqueezeÎ
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1
,batch_normalization_13/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg/1076885*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_13/AssignMovingAvg/decayÙ
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_13_assignmovingavg_1076885*
_output_shapes
:P*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOpå
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg/1076885*
_output_shapes
:P2,
*batch_normalization_13/AssignMovingAvg/subÜ
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:05batch_normalization_13/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg/1076885*
_output_shapes
:P2,
*batch_normalization_13/AssignMovingAvg/mul»
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_13_assignmovingavg_1076885.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_13/AssignMovingAvg/1076885*
_output_shapes
 *
dtype02<
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_13/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_13/AssignMovingAvg_1/1076891*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_13/AssignMovingAvg_1/decayß
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_13_assignmovingavg_1_1076891*
_output_shapes
:P*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpï
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_13/AssignMovingAvg_1/1076891*
_output_shapes
:P2.
,batch_normalization_13/AssignMovingAvg_1/subæ
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:07batch_normalization_13/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_13/AssignMovingAvg_1/1076891*
_output_shapes
:P2.
,batch_normalization_13/AssignMovingAvg_1/mulÇ
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_13_assignmovingavg_1_10768910batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_13/AssignMovingAvg_1/1076891*
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
:P2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:P2(
&batch_normalization_13/batchnorm/Rsqrtã
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpá
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2&
$batch_normalization_13/batchnorm/mulÙ
&batch_normalization_13/batchnorm/mul_1Mul conv1d_3/conv1d/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2(
&batch_normalization_13/batchnorm/mul_1×
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:P2(
&batch_normalization_13/batchnorm/mul_2×
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype021
/batch_normalization_13/batchnorm/ReadVariableOpÝ
$batch_normalization_13/batchnorm/subSub7batch_normalization_13/batchnorm/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2&
$batch_normalization_13/batchnorm/subå
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2(
&batch_normalization_13/batchnorm/add_1
re_lu_8/ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
re_lu_8/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const
flatten_2/ReshapeReshapere_lu_8/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
flatten_2/Reshape©
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02 
dense_10/MatMul/ReadVariableOp¢
dense_10/MatMulMatMulflatten_2/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/MatMul|
dense_10/SigmoidSigmoiddense_10/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Sigmoid

IdentityIdentitydense_10/Sigmoid:y:0;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_11/batchnorm/ReadVariableOp4^batch_normalization_11/batchnorm/mul/ReadVariableOp;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_12/AssignMovingAvg/ReadVariableOp=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_13/AssignMovingAvg/ReadVariableOp=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp4^batch_normalization_13/batchnorm/mul/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2x
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
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2x
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp:batch_normalization_13/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯0
Ì
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1075807

inputs
assignmovingavg_1075782
assignmovingavg_1_1075788)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	22
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1075782*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1075782*
_output_shapes	
:2*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1075782*
_output_shapes	
:22
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1075782*
_output_shapes	
:22
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1075782AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1075782*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1075788*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1075788*
_output_shapes	
:2*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1075788*
_output_shapes	
:22
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1075788*
_output_shapes	
:22
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1075788AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1075788*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:22
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:22
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:22
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ê

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077554

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
a

 __inference__traced_save_1077771
file_prefix-
)savev2_dense_9_kernel_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_nadam_dense_9_kernel_m_read_readvariableopC
?savev2_nadam_batch_normalization_11_gamma_m_read_readvariableopB
>savev2_nadam_batch_normalization_11_beta_m_read_readvariableop6
2savev2_nadam_conv1d_2_kernel_m_read_readvariableopC
?savev2_nadam_batch_normalization_12_gamma_m_read_readvariableopB
>savev2_nadam_batch_normalization_12_beta_m_read_readvariableop6
2savev2_nadam_conv1d_3_kernel_m_read_readvariableopC
?savev2_nadam_batch_normalization_13_gamma_m_read_readvariableopB
>savev2_nadam_batch_normalization_13_beta_m_read_readvariableop6
2savev2_nadam_dense_10_kernel_m_read_readvariableop5
1savev2_nadam_dense_9_kernel_v_read_readvariableopC
?savev2_nadam_batch_normalization_11_gamma_v_read_readvariableopB
>savev2_nadam_batch_normalization_11_beta_v_read_readvariableop6
2savev2_nadam_conv1d_2_kernel_v_read_readvariableopC
?savev2_nadam_batch_normalization_12_gamma_v_read_readvariableopB
>savev2_nadam_batch_normalization_12_beta_v_read_readvariableop6
2savev2_nadam_conv1d_3_kernel_v_read_readvariableopC
?savev2_nadam_batch_normalization_13_gamma_v_read_readvariableopB
>savev2_nadam_batch_normalization_13_beta_v_read_readvariableop6
2savev2_nadam_dense_10_kernel_v_read_readvariableop
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
ShardedFilenameÆ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*Ø
valueÎBË-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesâ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesØ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_nadam_dense_9_kernel_m_read_readvariableop?savev2_nadam_batch_normalization_11_gamma_m_read_readvariableop>savev2_nadam_batch_normalization_11_beta_m_read_readvariableop2savev2_nadam_conv1d_2_kernel_m_read_readvariableop?savev2_nadam_batch_normalization_12_gamma_m_read_readvariableop>savev2_nadam_batch_normalization_12_beta_m_read_readvariableop2savev2_nadam_conv1d_3_kernel_m_read_readvariableop?savev2_nadam_batch_normalization_13_gamma_m_read_readvariableop>savev2_nadam_batch_normalization_13_beta_m_read_readvariableop2savev2_nadam_dense_10_kernel_m_read_readvariableop1savev2_nadam_dense_9_kernel_v_read_readvariableop?savev2_nadam_batch_normalization_11_gamma_v_read_readvariableop>savev2_nadam_batch_normalization_11_beta_v_read_readvariableop2savev2_nadam_conv1d_2_kernel_v_read_readvariableop?savev2_nadam_batch_normalization_12_gamma_v_read_readvariableop>savev2_nadam_batch_normalization_12_beta_v_read_readvariableop2savev2_nadam_conv1d_3_kernel_v_read_readvariableop?savev2_nadam_batch_normalization_13_gamma_v_read_readvariableop>savev2_nadam_batch_normalization_13_beta_v_read_readvariableop2savev2_nadam_dense_10_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
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

identity_1Identity_1:output:0*å
_input_shapesÓ
Ð: :	2:2:2:2:2:P : : : : : P:P:P:P:P:	2: : : : : : : : :	2:2:2:P : : : P:P:P:	2:	2:2:2:P : : : P:P:P:	2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	2:!

_output_shapes	
:2:!

_output_shapes	
:2:!

_output_shapes	
:2:!

_output_shapes	
:2:)%
#
_output_shapes
:P :!

_output_shapes	
: :!

_output_shapes	
: :!	

_output_shapes	
: :!


_output_shapes	
: :)%
#
_output_shapes
: P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:%!

_output_shapes
:	2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2:!

_output_shapes	
:2:!

_output_shapes	
:2:)%
#
_output_shapes
:P :!

_output_shapes	
: :!

_output_shapes	
: :)%
#
_output_shapes
: P:  

_output_shapes
:P: !

_output_shapes
:P:%"!

_output_shapes
:	2:%#!

_output_shapes
:	2:!$

_output_shapes	
:2:!%

_output_shapes	
:2:)&%
#
_output_shapes
:P :!'

_output_shapes	
: :!(

_output_shapes	
: :))%
#
_output_shapes
: P: *

_output_shapes
:P: +

_output_shapes
:P:%,!

_output_shapes
:	2:-

_output_shapes
: 
¾
o
)__inference_dense_9_layer_call_fn_1077094

inputs
unknown
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_10761422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
Ã
J__inference_Discriminator_layer_call_and_return_conditional_losses_1077006

inputs*
&dense_9_matmul_readvariableop_resource<
8batch_normalization_11_batchnorm_readvariableop_resource@
<batch_normalization_11_batchnorm_mul_readvariableop_resource>
:batch_normalization_11_batchnorm_readvariableop_1_resource>
:batch_normalization_11_batchnorm_readvariableop_2_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource<
8batch_normalization_12_batchnorm_readvariableop_resource@
<batch_normalization_12_batchnorm_mul_readvariableop_resource>
:batch_normalization_12_batchnorm_readvariableop_1_resource>
:batch_normalization_12_batchnorm_readvariableop_2_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource<
8batch_normalization_13_batchnorm_readvariableop_resource@
<batch_normalization_13_batchnorm_mul_readvariableop_resource>
:batch_normalization_13_batchnorm_readvariableop_1_resource>
:batch_normalization_13_batchnorm_readvariableop_2_resource+
'dense_10_matmul_readvariableop_resource
identity¢/batch_normalization_11/batchnorm/ReadVariableOp¢1batch_normalization_11/batchnorm/ReadVariableOp_1¢1batch_normalization_11/batchnorm/ReadVariableOp_2¢3batch_normalization_11/batchnorm/mul/ReadVariableOp¢/batch_normalization_12/batchnorm/ReadVariableOp¢1batch_normalization_12/batchnorm/ReadVariableOp_1¢1batch_normalization_12/batchnorm/ReadVariableOp_2¢3batch_normalization_12/batchnorm/mul/ReadVariableOp¢/batch_normalization_13/batchnorm/ReadVariableOp¢1batch_normalization_13/batchnorm/ReadVariableOp_1¢1batch_normalization_13/batchnorm/ReadVariableOp_2¢3batch_normalization_13/batchnorm/mul/ReadVariableOp¢+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¦
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_9/MatMulØ
/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype021
/batch_normalization_11/batchnorm/ReadVariableOp
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_11/batchnorm/add/yå
$batch_normalization_11/batchnorm/addAddV27batch_normalization_11/batchnorm/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/add©
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:22(
&batch_normalization_11/batchnorm/Rsqrtä
3batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype025
3batch_normalization_11/batchnorm/mul/ReadVariableOpâ
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:0;batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/mulÎ
&batch_normalization_11/batchnorm/mul_1Muldense_9/MatMul:product:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22(
&batch_normalization_11/batchnorm/mul_1Þ
1batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes	
:2*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_1â
&batch_normalization_11/batchnorm/mul_2Mul9batch_normalization_11/batchnorm/ReadVariableOp_1:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:22(
&batch_normalization_11/batchnorm/mul_2Þ
1batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes	
:2*
dtype023
1batch_normalization_11/batchnorm/ReadVariableOp_2à
$batch_normalization_11/batchnorm/subSub9batch_normalization_11/batchnorm/ReadVariableOp_2:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22&
$batch_normalization_11/batchnorm/subâ
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22(
&batch_normalization_11/batchnorm/add_1
re_lu_6/ReluRelu*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
re_lu_6/Relul
reshape_2/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shape
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2
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
value	B :P2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :P2
reshape_2/Reshape/shape/2Ò
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape¥
reshape_2/ReshapeReshapere_lu_6/Relu:activations:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
reshape_2/Reshape
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_2/conv1d/ExpandDims/dimÅ
conv1d_2/conv1d/ExpandDims
ExpandDimsreshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
conv1d_2/conv1d/ExpandDimsÔ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:P *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÜ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P 2
conv1d_2/conv1d/ExpandDims_1Û
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
paddingSAME*
strides
2
conv1d_2/conv1d®
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_2/conv1d/SqueezeØ
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_12/batchnorm/add/yå
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/add©
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
: 2(
&batch_normalization_12/batchnorm/Rsqrtä
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOpâ
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/mulÚ
&batch_normalization_12/batchnorm/mul_1Mul conv1d_2/conv1d/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2(
&batch_normalization_12/batchnorm/mul_1Þ
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1â
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
: 2(
&batch_normalization_12/batchnorm/mul_2Þ
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2à
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2&
$batch_normalization_12/batchnorm/subæ
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2(
&batch_normalization_12/batchnorm/add_1
re_lu_7/ReluRelu*batch_normalization_12/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
re_lu_7/Relu
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_3/conv1d/ExpandDims/dimÆ
conv1d_3/conv1d/ExpandDims
ExpandDimsre_lu_7/Relu:activations:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
conv1d_3/conv1d/ExpandDimsÔ
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: P*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimÜ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: P2
conv1d_3/conv1d/ExpandDims_1Ú
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
paddingSAME*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_3/conv1d/Squeeze×
/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:P2&
$batch_normalization_13/batchnorm/add¨
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:P2(
&batch_normalization_13/batchnorm/Rsqrtã
3batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype025
3batch_normalization_13/batchnorm/mul/ReadVariableOpá
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:0;batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2&
$batch_normalization_13/batchnorm/mulÙ
&batch_normalization_13/batchnorm/mul_1Mul conv1d_3/conv1d/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2(
&batch_normalization_13/batchnorm/mul_1Ý
1batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_1á
&batch_normalization_13/batchnorm/mul_2Mul9batch_normalization_13/batchnorm/ReadVariableOp_1:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:P2(
&batch_normalization_13/batchnorm/mul_2Ý
1batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype023
1batch_normalization_13/batchnorm/ReadVariableOp_2ß
$batch_normalization_13/batchnorm/subSub9batch_normalization_13/batchnorm/ReadVariableOp_2:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2&
$batch_normalization_13/batchnorm/subå
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2(
&batch_normalization_13/batchnorm/add_1
re_lu_8/ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
re_lu_8/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const
flatten_2/ReshapeReshapere_lu_8/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
flatten_2/Reshape©
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02 
dense_10/MatMul/ReadVariableOp¢
dense_10/MatMulMatMulflatten_2/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/MatMul|
dense_10/SigmoidSigmoiddense_10/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Sigmoidõ
IdentityIdentitydense_10/Sigmoid:y:00^batch_normalization_11/batchnorm/ReadVariableOp2^batch_normalization_11/batchnorm/ReadVariableOp_12^batch_normalization_11/batchnorm/ReadVariableOp_24^batch_normalization_11/batchnorm/mul/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp0^batch_normalization_13/batchnorm/ReadVariableOp2^batch_normalization_13/batchnorm/ReadVariableOp_12^batch_normalization_13/batchnorm/ReadVariableOp_24^batch_normalization_13/batchnorm/mul/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2b
/batch_normalization_11/batchnorm/ReadVariableOp/batch_normalization_11/batchnorm/ReadVariableOp2f
1batch_normalization_11/batchnorm/ReadVariableOp_11batch_normalization_11/batchnorm/ReadVariableOp_12f
1batch_normalization_11/batchnorm/ReadVariableOp_21batch_normalization_11/batchnorm/ReadVariableOp_22j
3batch_normalization_11/batchnorm/mul/ReadVariableOp3batch_normalization_11/batchnorm/mul/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2b
/batch_normalization_13/batchnorm/ReadVariableOp/batch_normalization_13/batchnorm/ReadVariableOp2f
1batch_normalization_13/batchnorm/ReadVariableOp_11batch_normalization_13/batchnorm/ReadVariableOp_12f
1batch_normalization_13/batchnorm/ReadVariableOp_21batch_normalization_13/batchnorm/ReadVariableOp_22j
3batch_normalization_13/batchnorm/mul/ReadVariableOp3batch_normalization_13/batchnorm/mul/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
G
+__inference_reshape_2_layer_call_fn_1077204

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_10762152
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

º
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1077216

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:P *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P 2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs


S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1076120

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
¶

Ü
/__inference_Discriminator_layer_call_fn_1076733
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
identity¢StatefulPartitionedCall¶
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
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Discriminator_layer_call_and_return_conditional_losses_10766982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
¶
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1076485

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
Ë
«
8__inference_batch_normalization_13_layer_call_fn_1077580

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_10764302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs

º
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1076235

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:P *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P 2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
¶
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1077596

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs


Ò
%__inference_signature_wrapper_1076780
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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_10757112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
¯0
Ì
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077130

inputs
assignmovingavg_1077105
assignmovingavg_1_1077111)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	22
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	2*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:2*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077105*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077105*
_output_shapes	
:2*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077105*
_output_shapes	
:22
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077105*
_output_shapes	
:22
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077105AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077105*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077111*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077111*
_output_shapes	
:2*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077111*
_output_shapes	
:22
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077111*
_output_shapes	
:22
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077111AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077111*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:22
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:22
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:22
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
1
Ì
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077259

inputs
assignmovingavg_1077234
assignmovingavg_1_1077240)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
: 2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077234*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077234*
_output_shapes	
: *
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077234*
_output_shapes	
: 2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077234*
_output_shapes	
: 2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077234AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077234*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077240*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077240*
_output_shapes	
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077240*
_output_shapes	
: 2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077240*
_output_shapes	
: 2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077240AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077240*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Á
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
:

J__inference_Discriminator_layer_call_and_return_conditional_losses_1076514
input_4
dense_9_1076151"
batch_normalization_11_1076180"
batch_normalization_11_1076182"
batch_normalization_11_1076184"
batch_normalization_11_1076186
conv1d_2_1076244"
batch_normalization_12_1076329"
batch_normalization_12_1076331"
batch_normalization_12_1076333"
batch_normalization_12_1076335
conv1d_3_1076372"
batch_normalization_13_1076457"
batch_normalization_13_1076459"
batch_normalization_13_1076461"
batch_normalization_13_1076463
dense_10_1076510
identity¢.batch_normalization_11/StatefulPartitionedCall¢.batch_normalization_12/StatefulPartitionedCall¢.batch_normalization_13/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCall¢ conv1d_3/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_9_1076151*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_10761422!
dense_9/StatefulPartitionedCallÂ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_11_1076180batch_normalization_11_1076182batch_normalization_11_1076184batch_normalization_11_1076186*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_107580720
.batch_normalization_11/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_10761942
re_lu_6/PartitionedCallö
reshape_2/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_10762152
reshape_2/PartitionedCall¤
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1d_2_1076244*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_10762352"
 conv1d_2/StatefulPartitionedCallÇ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_12_1076329batch_normalization_12_1076331batch_normalization_12_1076333batch_normalization_12_1076335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_107628220
.batch_normalization_12/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_10763432
re_lu_7/PartitionedCall¡
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv1d_3_1076372*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_10763632"
 conv1d_3/StatefulPartitionedCallÆ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0batch_normalization_13_1076457batch_normalization_13_1076459batch_normalization_13_1076461batch_normalization_13_1076463*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_107641020
.batch_normalization_13/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_10764712
re_lu_8/PartitionedCalló
flatten_2/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_10764852
flatten_2/PartitionedCall
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_10_1076510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_10765012"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
ô

D__inference_dense_9_layer_call_and_return_conditional_losses_1076142

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ì
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1075947

inputs
assignmovingavg_1075922
assignmovingavg_1_1075928)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
: 2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1075922*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1075922*
_output_shapes	
: *
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1075922*
_output_shapes	
: 2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1075922*
_output_shapes	
: 2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1075922AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1075922*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1075928*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1075928*
_output_shapes	
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1075928*
_output_shapes	
: 2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1075928*
_output_shapes	
: 2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1075928AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1075928*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1Á
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
á
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1077199

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
strided_slice/stack_2â
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
value	B :P2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Â
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_1077585

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
¦
E
)__inference_re_lu_7_layer_call_fn_1077397

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_10763432
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
¿
è
#__inference__traced_restore_1077913
file_prefix#
assignvariableop_dense_9_kernel3
/assignvariableop_1_batch_normalization_11_gamma2
.assignvariableop_2_batch_normalization_11_beta9
5assignvariableop_3_batch_normalization_11_moving_mean=
9assignvariableop_4_batch_normalization_11_moving_variance&
"assignvariableop_5_conv1d_2_kernel3
/assignvariableop_6_batch_normalization_12_gamma2
.assignvariableop_7_batch_normalization_12_beta9
5assignvariableop_8_batch_normalization_12_moving_mean=
9assignvariableop_9_batch_normalization_12_moving_variance'
#assignvariableop_10_conv1d_3_kernel4
0assignvariableop_11_batch_normalization_13_gamma3
/assignvariableop_12_batch_normalization_13_beta:
6assignvariableop_13_batch_normalization_13_moving_mean>
:assignvariableop_14_batch_normalization_13_moving_variance'
#assignvariableop_15_dense_10_kernel"
assignvariableop_16_nadam_iter$
 assignvariableop_17_nadam_beta_1$
 assignvariableop_18_nadam_beta_2#
assignvariableop_19_nadam_decay+
'assignvariableop_20_nadam_learning_rate,
(assignvariableop_21_nadam_momentum_cache
assignvariableop_22_total
assignvariableop_23_count.
*assignvariableop_24_nadam_dense_9_kernel_m<
8assignvariableop_25_nadam_batch_normalization_11_gamma_m;
7assignvariableop_26_nadam_batch_normalization_11_beta_m/
+assignvariableop_27_nadam_conv1d_2_kernel_m<
8assignvariableop_28_nadam_batch_normalization_12_gamma_m;
7assignvariableop_29_nadam_batch_normalization_12_beta_m/
+assignvariableop_30_nadam_conv1d_3_kernel_m<
8assignvariableop_31_nadam_batch_normalization_13_gamma_m;
7assignvariableop_32_nadam_batch_normalization_13_beta_m/
+assignvariableop_33_nadam_dense_10_kernel_m.
*assignvariableop_34_nadam_dense_9_kernel_v<
8assignvariableop_35_nadam_batch_normalization_11_gamma_v;
7assignvariableop_36_nadam_batch_normalization_11_beta_v/
+assignvariableop_37_nadam_conv1d_2_kernel_v<
8assignvariableop_38_nadam_batch_normalization_12_gamma_v;
7assignvariableop_39_nadam_batch_normalization_12_beta_v/
+assignvariableop_40_nadam_conv1d_3_kernel_v<
8assignvariableop_41_nadam_batch_normalization_13_gamma_v;
7assignvariableop_42_nadam_batch_normalization_13_beta_v/
+assignvariableop_43_nadam_dense_10_kernel_v
identity_45¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ì
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*Ø
valueÎBË-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesè
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ê
_output_shapes·
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1´
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_11_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_11_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3º
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_11_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¾
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_11_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv1d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_12_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_12_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8º
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_12_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_12_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¸
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_13_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_13_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¾
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_13_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Â
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_13_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_10_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¨
AssignVariableOp_18AssignVariableOp assignvariableop_18_nadam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_nadam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21°
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_momentum_cacheIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_nadam_dense_9_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_nadam_batch_normalization_11_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¿
AssignVariableOp_26AssignVariableOp7assignvariableop_26_nadam_batch_normalization_11_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_nadam_conv1d_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28À
AssignVariableOp_28AssignVariableOp8assignvariableop_28_nadam_batch_normalization_12_gamma_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¿
AssignVariableOp_29AssignVariableOp7assignvariableop_29_nadam_batch_normalization_12_beta_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30³
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_conv1d_3_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31À
AssignVariableOp_31AssignVariableOp8assignvariableop_31_nadam_batch_normalization_13_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¿
AssignVariableOp_32AssignVariableOp7assignvariableop_32_nadam_batch_normalization_13_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_nadam_dense_10_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34²
AssignVariableOp_34AssignVariableOp*assignvariableop_34_nadam_dense_9_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35À
AssignVariableOp_35AssignVariableOp8assignvariableop_35_nadam_batch_normalization_11_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¿
AssignVariableOp_36AssignVariableOp7assignvariableop_36_nadam_batch_normalization_11_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_nadam_conv1d_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38À
AssignVariableOp_38AssignVariableOp8assignvariableop_38_nadam_batch_normalization_12_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¿
AssignVariableOp_39AssignVariableOp7assignvariableop_39_nadam_batch_normalization_12_beta_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40³
AssignVariableOp_40AssignVariableOp+assignvariableop_40_nadam_conv1d_3_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41À
AssignVariableOp_41AssignVariableOp8assignvariableop_41_nadam_batch_normalization_13_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¿
AssignVariableOp_42AssignVariableOp7assignvariableop_42_nadam_batch_normalization_13_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_nadam_dense_10_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¦
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44
Identity_45IdentityIdentity_44:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_45"#
identity_45Identity_45:output:0*Ç
_input_shapesµ
²: ::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
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
ê

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1076430

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
Ó¬
Ü
"__inference__wrapped_model_1075711
input_48
4discriminator_dense_9_matmul_readvariableop_resourceJ
Fdiscriminator_batch_normalization_11_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_11_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_11_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_11_batchnorm_readvariableop_2_resourceF
Bdiscriminator_conv1d_2_conv1d_expanddims_1_readvariableop_resourceJ
Fdiscriminator_batch_normalization_12_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_12_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_12_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_12_batchnorm_readvariableop_2_resourceF
Bdiscriminator_conv1d_3_conv1d_expanddims_1_readvariableop_resourceJ
Fdiscriminator_batch_normalization_13_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_13_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_13_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_13_batchnorm_readvariableop_2_resource9
5discriminator_dense_10_matmul_readvariableop_resource
identity¢=Discriminator/batch_normalization_11/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOp¢=Discriminator/batch_normalization_12/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOp¢=Discriminator/batch_normalization_13/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOp¢9Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢9Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢,Discriminator/dense_10/MatMul/ReadVariableOp¢+Discriminator/dense_9/MatMul/ReadVariableOpÐ
+Discriminator/dense_9/MatMul/ReadVariableOpReadVariableOp4discriminator_dense_9_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02-
+Discriminator/dense_9/MatMul/ReadVariableOp·
Discriminator/dense_9/MatMulMatMulinput_43Discriminator/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Discriminator/dense_9/MatMul
=Discriminator/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype02?
=Discriminator/batch_normalization_11/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_11/batchnorm/add/y
2Discriminator/batch_normalization_11/batchnorm/addAddV2EDiscriminator/batch_normalization_11/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:224
2Discriminator/batch_normalization_11/batchnorm/addÓ
4Discriminator/batch_normalization_11/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:226
4Discriminator/batch_normalization_11/batchnorm/Rsqrt
ADiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype02C
ADiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_11/batchnorm/mulMul8Discriminator/batch_normalization_11/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:224
2Discriminator/batch_normalization_11/batchnorm/mul
4Discriminator/batch_normalization_11/batchnorm/mul_1Mul&Discriminator/dense_9/MatMul:product:06Discriminator/batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ226
4Discriminator/batch_normalization_11/batchnorm/mul_1
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes	
:2*
dtype02A
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_11/batchnorm/mul_2MulGDiscriminator/batch_normalization_11/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:226
4Discriminator/batch_normalization_11/batchnorm/mul_2
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes	
:2*
dtype02A
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_11/batchnorm/subSubGDiscriminator/batch_normalization_11/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:224
2Discriminator/batch_normalization_11/batchnorm/sub
4Discriminator/batch_normalization_11/batchnorm/add_1AddV28Discriminator/batch_normalization_11/batchnorm/mul_1:z:06Discriminator/batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ226
4Discriminator/batch_normalization_11/batchnorm/add_1­
Discriminator/re_lu_6/ReluRelu8Discriminator/batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Discriminator/re_lu_6/Relu
Discriminator/reshape_2/ShapeShape(Discriminator/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
Discriminator/reshape_2/Shape¤
+Discriminator/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+Discriminator/reshape_2/strided_slice/stack¨
-Discriminator/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-Discriminator/reshape_2/strided_slice/stack_1¨
-Discriminator/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-Discriminator/reshape_2/strided_slice/stack_2ò
%Discriminator/reshape_2/strided_sliceStridedSlice&Discriminator/reshape_2/Shape:output:04Discriminator/reshape_2/strided_slice/stack:output:06Discriminator/reshape_2/strided_slice/stack_1:output:06Discriminator/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%Discriminator/reshape_2/strided_slice
'Discriminator/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :P2)
'Discriminator/reshape_2/Reshape/shape/1
'Discriminator/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :P2)
'Discriminator/reshape_2/Reshape/shape/2
%Discriminator/reshape_2/Reshape/shapePack.Discriminator/reshape_2/strided_slice:output:00Discriminator/reshape_2/Reshape/shape/1:output:00Discriminator/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%Discriminator/reshape_2/Reshape/shapeÝ
Discriminator/reshape_2/ReshapeReshape(Discriminator/re_lu_6/Relu:activations:0.Discriminator/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2!
Discriminator/reshape_2/Reshape§
,Discriminator/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2.
,Discriminator/conv1d_2/conv1d/ExpandDims/dimý
(Discriminator/conv1d_2/conv1d/ExpandDims
ExpandDims(Discriminator/reshape_2/Reshape:output:05Discriminator/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2*
(Discriminator/conv1d_2/conv1d/ExpandDimsþ
9Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBdiscriminator_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:P *
dtype02;
9Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp¢
.Discriminator/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.Discriminator/conv1d_2/conv1d/ExpandDims_1/dim
*Discriminator/conv1d_2/conv1d/ExpandDims_1
ExpandDimsADiscriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:07Discriminator/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P 2,
*Discriminator/conv1d_2/conv1d/ExpandDims_1
Discriminator/conv1d_2/conv1dConv2D1Discriminator/conv1d_2/conv1d/ExpandDims:output:03Discriminator/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
paddingSAME*
strides
2
Discriminator/conv1d_2/conv1dØ
%Discriminator/conv1d_2/conv1d/SqueezeSqueeze&Discriminator/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2'
%Discriminator/conv1d_2/conv1d/Squeeze
=Discriminator/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02?
=Discriminator/batch_normalization_12/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_12/batchnorm/add/y
2Discriminator/batch_normalization_12/batchnorm/addAddV2EDiscriminator/batch_normalization_12/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
: 24
2Discriminator/batch_normalization_12/batchnorm/addÓ
4Discriminator/batch_normalization_12/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
: 26
4Discriminator/batch_normalization_12/batchnorm/Rsqrt
ADiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02C
ADiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_12/batchnorm/mulMul8Discriminator/batch_normalization_12/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 24
2Discriminator/batch_normalization_12/batchnorm/mul
4Discriminator/batch_normalization_12/batchnorm/mul_1Mul.Discriminator/conv1d_2/conv1d/Squeeze:output:06Discriminator/batch_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 26
4Discriminator/batch_normalization_12/batchnorm/mul_1
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype02A
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_12/batchnorm/mul_2MulGDiscriminator/batch_normalization_12/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
: 26
4Discriminator/batch_normalization_12/batchnorm/mul_2
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype02A
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_12/batchnorm/subSubGDiscriminator/batch_normalization_12/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 24
2Discriminator/batch_normalization_12/batchnorm/sub
4Discriminator/batch_normalization_12/batchnorm/add_1AddV28Discriminator/batch_normalization_12/batchnorm/mul_1:z:06Discriminator/batch_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 26
4Discriminator/batch_normalization_12/batchnorm/add_1±
Discriminator/re_lu_7/ReluRelu8Discriminator/batch_normalization_12/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
Discriminator/re_lu_7/Relu§
,Discriminator/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2.
,Discriminator/conv1d_3/conv1d/ExpandDims/dimþ
(Discriminator/conv1d_3/conv1d/ExpandDims
ExpandDims(Discriminator/re_lu_7/Relu:activations:05Discriminator/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2*
(Discriminator/conv1d_3/conv1d/ExpandDimsþ
9Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpBdiscriminator_conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: P*
dtype02;
9Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp¢
.Discriminator/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.Discriminator/conv1d_3/conv1d/ExpandDims_1/dim
*Discriminator/conv1d_3/conv1d/ExpandDims_1
ExpandDimsADiscriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:07Discriminator/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: P2,
*Discriminator/conv1d_3/conv1d/ExpandDims_1
Discriminator/conv1d_3/conv1dConv2D1Discriminator/conv1d_3/conv1d/ExpandDims:output:03Discriminator/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
paddingSAME*
strides
2
Discriminator/conv1d_3/conv1d×
%Discriminator/conv1d_3/conv1d/SqueezeSqueeze&Discriminator/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2'
%Discriminator/conv1d_3/conv1d/Squeeze
=Discriminator/batch_normalization_13/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype02?
=Discriminator/batch_normalization_13/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_13/batchnorm/add/y
2Discriminator/batch_normalization_13/batchnorm/addAddV2EDiscriminator/batch_normalization_13/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:P24
2Discriminator/batch_normalization_13/batchnorm/addÒ
4Discriminator/batch_normalization_13/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:P26
4Discriminator/batch_normalization_13/batchnorm/Rsqrt
ADiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02C
ADiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_13/batchnorm/mulMul8Discriminator/batch_normalization_13/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P24
2Discriminator/batch_normalization_13/batchnorm/mul
4Discriminator/batch_normalization_13/batchnorm/mul_1Mul.Discriminator/conv1d_3/conv1d/Squeeze:output:06Discriminator/batch_normalization_13/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP26
4Discriminator/batch_normalization_13/batchnorm/mul_1
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_13_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype02A
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_13/batchnorm/mul_2MulGDiscriminator/batch_normalization_13/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:P26
4Discriminator/batch_normalization_13/batchnorm/mul_2
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_13_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype02A
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_13/batchnorm/subSubGDiscriminator/batch_normalization_13/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P24
2Discriminator/batch_normalization_13/batchnorm/sub
4Discriminator/batch_normalization_13/batchnorm/add_1AddV28Discriminator/batch_normalization_13/batchnorm/mul_1:z:06Discriminator/batch_normalization_13/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP26
4Discriminator/batch_normalization_13/batchnorm/add_1°
Discriminator/re_lu_8/ReluRelu8Discriminator/batch_normalization_13/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
Discriminator/re_lu_8/Relu
Discriminator/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Discriminator/flatten_2/ConstÒ
Discriminator/flatten_2/ReshapeReshape(Discriminator/re_lu_8/Relu:activations:0&Discriminator/flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22!
Discriminator/flatten_2/ReshapeÓ
,Discriminator/dense_10/MatMul/ReadVariableOpReadVariableOp5discriminator_dense_10_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02.
,Discriminator/dense_10/MatMul/ReadVariableOpÚ
Discriminator/dense_10/MatMulMatMul(Discriminator/flatten_2/Reshape:output:04Discriminator/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Discriminator/dense_10/MatMul¦
Discriminator/dense_10/SigmoidSigmoid'Discriminator/dense_10/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
Discriminator/dense_10/Sigmoidã
IdentityIdentity"Discriminator/dense_10/Sigmoid:y:0>^Discriminator/batch_normalization_11/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_11/batchnorm/mul/ReadVariableOp>^Discriminator/batch_normalization_12/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_12/batchnorm/mul/ReadVariableOp>^Discriminator/batch_normalization_13/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_13/batchnorm/mul/ReadVariableOp:^Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:^Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp-^Discriminator/dense_10/MatMul/ReadVariableOp,^Discriminator/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2~
=Discriminator/batch_normalization_11/batchnorm/ReadVariableOp=Discriminator/batch_normalization_11/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_11/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_11/batchnorm/mul/ReadVariableOp2~
=Discriminator/batch_normalization_12/batchnorm/ReadVariableOp=Discriminator/batch_normalization_12/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_12/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_12/batchnorm/mul/ReadVariableOp2~
=Discriminator/batch_normalization_13/batchnorm/ReadVariableOp=Discriminator/batch_normalization_13/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_13/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_13/batchnorm/mul/ReadVariableOp2v
9Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp9Discriminator/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2v
9Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp9Discriminator/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2\
,Discriminator/dense_10/MatMul/ReadVariableOp,Discriminator/dense_10/MatMul/ReadVariableOp2Z
+Discriminator/dense_9/MatMul/ReadVariableOp+Discriminator/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
ô

D__inference_dense_9_layer_call_and_return_conditional_losses_1077087

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077150

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:22
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:22
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:22
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ð
p
*__inference_conv1d_2_layer_call_fn_1077223

inputs
unknown
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_10762352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
½
«
8__inference_batch_normalization_11_layer_call_fn_1077163

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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_10758072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

º
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1077409

inputs/
+conv1d_expanddims_1_readvariableop_resource
identity¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
conv1d/ExpandDims¹
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: P*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¸
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: P2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿP :2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
ä

S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1075840

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:22
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:22
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:2*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:22
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:22
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:2*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:22
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¿
«
8__inference_batch_normalization_11_layer_call_fn_1077176

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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_10758402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ2::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ä0
Ì
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077341

inputs
assignmovingavg_1077316
assignmovingavg_1_1077322)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
: 2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077316*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077316*
_output_shapes	
: *
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077316*
_output_shapes	
: 2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077316*
_output_shapes	
: 2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077316AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077316*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077322*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077322*
_output_shapes	
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077322*
_output_shapes	
: 2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077322*
_output_shapes	
: 2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077322AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077322*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
Â
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_1076471

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
Ï
«
8__inference_batch_normalization_12_layer_call_fn_1077387

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_10763022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
í
«
8__inference_batch_normalization_13_layer_call_fn_1077485

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_10760872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 
_user_specified_nameinputs
À
p
*__inference_dense_10_layer_call_fn_1077616

inputs
unknown
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_10765012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
É
«
8__inference_batch_normalization_13_layer_call_fn_1077567

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_10764102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
á
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1076215

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
strided_slice/stack_2â
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
value	B :P2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
°

Ü
/__inference_Discriminator_layer_call_fn_1076648
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
identity¢StatefulPartitionedCall°
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Discriminator_layer_call_and_return_conditional_losses_10766132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
¦

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1075980

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¦

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077279

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì0
Ì
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077534

inputs
assignmovingavg_1077509
assignmovingavg_1_1077515)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:P2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077509*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1077509*
_output_shapes
:P*
dtype02 
AssignMovingAvg/ReadVariableOpò
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077509*
_output_shapes
:P2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1077509*
_output_shapes
:P2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1077509AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1077509*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077515*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1077515*
_output_shapes
:P*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077515*
_output_shapes
:P2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1077515*
_output_shapes
:P2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1077515AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1077515*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
ä0
Ì
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1076282

inputs
assignmovingavg_1076257
assignmovingavg_1_1076263)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
: 2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
: *
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076257*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1076257*
_output_shapes	
: *
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076257*
_output_shapes	
: 2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076257*
_output_shapes	
: 2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1076257AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076257*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076263*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1076263*
_output_shapes	
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076263*
_output_shapes	
: 2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076263*
_output_shapes	
: 2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1076263AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076263*
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
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
: 2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2
batchnorm/add_1¸
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿP ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP 
 
_user_specified_nameinputs
Ì0
Ì
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1076410

inputs
assignmovingavg_1076385
assignmovingavg_1_1076391)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:P2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:P*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 2
moments/Squeeze_1Í
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076385*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1076385*
_output_shapes
:P*
dtype02 
AssignMovingAvg/ReadVariableOpò
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076385*
_output_shapes
:P2
AssignMovingAvg/subé
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg/1076385*
_output_shapes
:P2
AssignMovingAvg/mul±
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1076385AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg/1076385*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÓ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076391*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1076391*
_output_shapes
:P*
dtype02"
 AssignMovingAvg_1/ReadVariableOpü
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076391*
_output_shapes
:P2
AssignMovingAvg_1/subó
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1076391*
_output_shapes
:P2
AssignMovingAvg_1/mul½
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1076391AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg_1/1076391*
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
:P2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:P2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:P2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿPP::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
 
G
+__inference_flatten_2_layer_call_fn_1077601

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_10764852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿPP:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
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
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ<
dense_100
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ù
àf
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
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
·_default_save_signature
+¸&call_and_return_all_conditional_losses
¹__call__"°b
_tf_keras_networkb{"class_name": "Functional", "name": "Discriminator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6400, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [80, 80]}}, "name": "reshape_2", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 160, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6400, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [80, 80]}}, "name": "reshape_2", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 160, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_13", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_10", 0, 0]]}}, "training_config": {"loss": "wasserstein_loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.5, "beta_2": 0.8999999761581421, "epsilon": 1e-07}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ë

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+º&call_and_return_all_conditional_losses
»__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 6400, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
º	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 6400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6400]}}
í
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ø
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"ç
_tf_keras_layerÍ{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [80, 80]}}}
á	

*kernel
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"Ä
_tf_keras_layerª{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 160, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 80]}}
¼	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+Ä&call_and_return_all_conditional_losses
Å__call__"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 160}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 160]}}
í
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+Æ&call_and_return_all_conditional_losses
Ç__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
â	

<kernel
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+È&call_and_return_all_conditional_losses
É__call__"Å
_tf_keras_layer«{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 160}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 160]}}
º	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 80]}}
í
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
è
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ñ

Rkernel
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6400]}}
¯
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\momentum_cachem£m¤m¥*m¦0m§1m¨<m©BmªCm«Rm¬v­v®v¯*v°0v±1v²<v³Bv´CvµRv¶"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
*3
04
15
<6
B7
C8
R9"
trackable_list_wrapper

0
1
2
3
4
*5
06
17
28
39
<10
B11
C12
D13
E14
R15"
trackable_list_wrapper
Î
regularization_losses
]metrics
trainable_variables
^non_trainable_variables
	variables

_layers
`layer_regularization_losses
alayer_metrics
¹__call__
·_default_save_signature
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
-
Òserving_default"
signature_map
!:	22dense_9/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
regularization_losses
bmetrics
trainable_variables
cnon_trainable_variables
	variables

dlayers
elayer_regularization_losses
flayer_metrics
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)22batch_normalization_11/gamma
*:(22batch_normalization_11/beta
3:12 (2"batch_normalization_11/moving_mean
7:52 (2&batch_normalization_11/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
°
regularization_losses
gmetrics
trainable_variables
hnon_trainable_variables
 	variables

ilayers
jlayer_regularization_losses
klayer_metrics
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
"regularization_losses
lmetrics
#trainable_variables
mnon_trainable_variables
$	variables

nlayers
olayer_regularization_losses
player_metrics
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
&regularization_losses
qmetrics
'trainable_variables
rnon_trainable_variables
(	variables

slayers
tlayer_regularization_losses
ulayer_metrics
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
&:$P 2conv1d_2/kernel
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
°
+regularization_losses
vmetrics
,trainable_variables
wnon_trainable_variables
-	variables

xlayers
ylayer_regularization_losses
zlayer_metrics
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_12/gamma
*:( 2batch_normalization_12/beta
3:1  (2"batch_normalization_12/moving_mean
7:5  (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
°
4regularization_losses
{metrics
5trainable_variables
|non_trainable_variables
6	variables

}layers
~layer_regularization_losses
layer_metrics
Å__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
8regularization_losses
metrics
9trainable_variables
non_trainable_variables
:	variables
layers
 layer_regularization_losses
layer_metrics
Ç__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
&:$ P2conv1d_3/kernel
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
µ
=regularization_losses
metrics
>trainable_variables
non_trainable_variables
?	variables
layers
 layer_regularization_losses
layer_metrics
É__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(P2batch_normalization_13/gamma
):'P2batch_normalization_13/beta
2:0P (2"batch_normalization_13/moving_mean
6:4P (2&batch_normalization_13/moving_variance
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
µ
Fregularization_losses
metrics
Gtrainable_variables
non_trainable_variables
H	variables
layers
 layer_regularization_losses
layer_metrics
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Jregularization_losses
metrics
Ktrainable_variables
non_trainable_variables
L	variables
layers
 layer_regularization_losses
layer_metrics
Í__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Nregularization_losses
metrics
Otrainable_variables
non_trainable_variables
P	variables
layers
 layer_regularization_losses
layer_metrics
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
": 	22dense_10/kernel
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
µ
Sregularization_losses
metrics
Ttrainable_variables
non_trainable_variables
U	variables
layers
 layer_regularization_losses
layer_metrics
Ñ__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
(
0"
trackable_list_wrapper
J
0
1
22
33
D4
E5"
trackable_list_wrapper
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
0
1"
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
20
31"
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
D0
E1"
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
¿

total

 count
¡	variables
¢	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
¡	variables"
_generic_user_object
':%	22Nadam/dense_9/kernel/m
1:/22$Nadam/batch_normalization_11/gamma/m
0:.22#Nadam/batch_normalization_11/beta/m
,:*P 2Nadam/conv1d_2/kernel/m
1:/ 2$Nadam/batch_normalization_12/gamma/m
0:. 2#Nadam/batch_normalization_12/beta/m
,:* P2Nadam/conv1d_3/kernel/m
0:.P2$Nadam/batch_normalization_13/gamma/m
/:-P2#Nadam/batch_normalization_13/beta/m
(:&	22Nadam/dense_10/kernel/m
':%	22Nadam/dense_9/kernel/v
1:/22$Nadam/batch_normalization_11/gamma/v
0:.22#Nadam/batch_normalization_11/beta/v
,:*P 2Nadam/conv1d_2/kernel/v
1:/ 2$Nadam/batch_normalization_12/gamma/v
0:. 2#Nadam/batch_normalization_12/beta/v
,:* P2Nadam/conv1d_3/kernel/v
0:.P2$Nadam/batch_normalization_13/gamma/v
/:-P2#Nadam/batch_normalization_13/beta/v
(:&	22Nadam/dense_10/kernel/v
à2Ý
"__inference__wrapped_model_1075711¶
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
input_4ÿÿÿÿÿÿÿÿÿ
ö2ó
J__inference_Discriminator_layer_call_and_return_conditional_losses_1077006
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076562
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076917
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076514À
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
2
/__inference_Discriminator_layer_call_fn_1077080
/__inference_Discriminator_layer_call_fn_1076648
/__inference_Discriminator_layer_call_fn_1077043
/__inference_Discriminator_layer_call_fn_1076733À
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
î2ë
D__inference_dense_9_layer_call_and_return_conditional_losses_1077087¢
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
Ó2Ð
)__inference_dense_9_layer_call_fn_1077094¢
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
ä2á
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077150
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077130´
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
®2«
8__inference_batch_normalization_11_layer_call_fn_1077163
8__inference_batch_normalization_11_layer_call_fn_1077176´
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
î2ë
D__inference_re_lu_6_layer_call_and_return_conditional_losses_1077181¢
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
Ó2Ð
)__inference_re_lu_6_layer_call_fn_1077186¢
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
F__inference_reshape_2_layer_call_and_return_conditional_losses_1077199¢
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
+__inference_reshape_2_layer_call_fn_1077204¢
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
ï2ì
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1077216¢
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
Ô2Ñ
*__inference_conv1d_2_layer_call_fn_1077223¢
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
2
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077279
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077341
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077259
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077361´
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
¢2
8__inference_batch_normalization_12_layer_call_fn_1077387
8__inference_batch_normalization_12_layer_call_fn_1077305
8__inference_batch_normalization_12_layer_call_fn_1077292
8__inference_batch_normalization_12_layer_call_fn_1077374´
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
î2ë
D__inference_re_lu_7_layer_call_and_return_conditional_losses_1077392¢
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
Ó2Ð
)__inference_re_lu_7_layer_call_fn_1077397¢
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
ï2ì
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1077409¢
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
Ô2Ñ
*__inference_conv1d_3_layer_call_fn_1077416¢
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
2
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077534
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077452
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077472
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077554´
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
¢2
8__inference_batch_normalization_13_layer_call_fn_1077485
8__inference_batch_normalization_13_layer_call_fn_1077498
8__inference_batch_normalization_13_layer_call_fn_1077567
8__inference_batch_normalization_13_layer_call_fn_1077580´
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
î2ë
D__inference_re_lu_8_layer_call_and_return_conditional_losses_1077585¢
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
Ó2Ð
)__inference_re_lu_8_layer_call_fn_1077590¢
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
F__inference_flatten_2_layer_call_and_return_conditional_losses_1077596¢
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
+__inference_flatten_2_layer_call_fn_1077601¢
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
ï2ì
E__inference_dense_10_layer_call_and_return_conditional_losses_1077609¢
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
Ô2Ñ
*__inference_dense_10_layer_call_fn_1077616¢
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
ÌBÉ
%__inference_signature_wrapper_1076780input_4"
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
 Á
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076514s*2301<DEBCR8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076562s*3021<EBDCR8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_Discriminator_layer_call_and_return_conditional_losses_1076917r*2301<DEBCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_Discriminator_layer_call_and_return_conditional_losses_1077006r*3021<EBDCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Discriminator_layer_call_fn_1076648f*2301<DEBCR8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_Discriminator_layer_call_fn_1076733f*3021<EBDCR8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_Discriminator_layer_call_fn_1077043e*2301<DEBCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_Discriminator_layer_call_fn_1077080e*3021<EBDCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
"__inference__wrapped_model_1075711y*3021<EBDCR0¢-
&¢#
!
input_4ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_10"
dense_10ÿÿÿÿÿÿÿÿÿ»
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077130d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ2
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ2
 »
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1077150d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ2
 
8__inference_batch_normalization_11_layer_call_fn_1077163W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ2
p
ª "ÿÿÿÿÿÿÿÿÿ2
8__inference_batch_normalization_11_layer_call_fn_1077176W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª "ÿÿÿÿÿÿÿÿÿ2Õ
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077259~2301A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Õ
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077279~3021A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ã
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077341l23018¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿP 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿP 
 Ã
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1077361l30218¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿP 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿP 
 ­
8__inference_batch_normalization_12_layer_call_fn_1077292q2301A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ­
8__inference_batch_normalization_12_layer_call_fn_1077305q3021A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
8__inference_batch_normalization_12_layer_call_fn_1077374_23018¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿP 
p
ª "ÿÿÿÿÿÿÿÿÿP 
8__inference_batch_normalization_12_layer_call_fn_1077387_30218¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿP 
p 
ª "ÿÿÿÿÿÿÿÿÿP Ó
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077452|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 Ó
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077472|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
 Á
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077534jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿPP
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿPP
 Á
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1077554jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿPP
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿPP
 «
8__inference_batch_normalization_13_layer_call_fn_1077485oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP«
8__inference_batch_normalization_13_layer_call_fn_1077498oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿP
8__inference_batch_normalization_13_layer_call_fn_1077567]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿPP
p
ª "ÿÿÿÿÿÿÿÿÿPP
8__inference_batch_normalization_13_layer_call_fn_1077580]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿPP
p 
ª "ÿÿÿÿÿÿÿÿÿPP­
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1077216d*3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿP 
 
*__inference_conv1d_2_layer_call_fn_1077223W*3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª "ÿÿÿÿÿÿÿÿÿP ­
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1077409d<4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿP 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿPP
 
*__inference_conv1d_3_layer_call_fn_1077416W<4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿP 
ª "ÿÿÿÿÿÿÿÿÿPP¥
E__inference_dense_10_layer_call_and_return_conditional_losses_1077609\R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_10_layer_call_fn_1077616OR0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_9_layer_call_and_return_conditional_losses_1077087\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ2
 |
)__inference_dense_9_layer_call_fn_1077094O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ2§
F__inference_flatten_2_layer_call_and_return_conditional_losses_1077596]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ2
 
+__inference_flatten_2_layer_call_fn_1077601P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª "ÿÿÿÿÿÿÿÿÿ2¢
D__inference_re_lu_6_layer_call_and_return_conditional_losses_1077181Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ2
 z
)__inference_re_lu_6_layer_call_fn_1077186M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2ª
D__inference_re_lu_7_layer_call_and_return_conditional_losses_1077392b4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿP 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿP 
 
)__inference_re_lu_7_layer_call_fn_1077397U4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿP 
ª "ÿÿÿÿÿÿÿÿÿP ¨
D__inference_re_lu_8_layer_call_and_return_conditional_losses_1077585`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª ")¢&

0ÿÿÿÿÿÿÿÿÿPP
 
)__inference_re_lu_8_layer_call_fn_1077590S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿPP
ª "ÿÿÿÿÿÿÿÿÿPP§
F__inference_reshape_2_layer_call_and_return_conditional_losses_1077199]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿPP
 
+__inference_reshape_2_layer_call_fn_1077204P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿPP®
%__inference_signature_wrapper_1076780*3021<EBDCR;¢8
¢ 
1ª.
,
input_4!
input_4ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_10"
dense_10ÿÿÿÿÿÿÿÿÿ