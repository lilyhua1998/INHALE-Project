í´
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8
z
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_45/kernel
s
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes

:@*
dtype0

batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_55/gamma

0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes
:@*
dtype0

batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_55/beta

/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes
:@*
dtype0

"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_55/moving_mean

6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_55/moving_variance

:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes
:@*
dtype0

conv1d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_10/kernel
y
$conv1d_10/kernel/Read/ReadVariableOpReadVariableOpconv1d_10/kernel*"
_output_shapes
:*
dtype0

batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_56/gamma

0batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_56/gamma*
_output_shapes
:*
dtype0

batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_56/beta

/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_56/beta*
_output_shapes
:*
dtype0

"batch_normalization_56/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_56/moving_mean

6batch_normalization_56/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_56/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_56/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_56/moving_variance

:batch_normalization_56/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_56/moving_variance*
_output_shapes
:*
dtype0

conv1d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_11/kernel
y
$conv1d_11/kernel/Read/ReadVariableOpReadVariableOpconv1d_11/kernel*"
_output_shapes
:*
dtype0

batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_57/gamma

0batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_57/gamma*
_output_shapes
:*
dtype0

batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_57/beta

/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_57/beta*
_output_shapes
:*
dtype0

"batch_normalization_57/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_57/moving_mean

6batch_normalization_57/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_57/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_57/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_57/moving_variance

:batch_normalization_57/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_57/moving_variance*
_output_shapes
:*
dtype0
z
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_46/kernel
s
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes

:@*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_45/kernel/m

*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m*
_output_shapes

:@*
dtype0

#Adam/batch_normalization_55/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_55/gamma/m

7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_55/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_55/beta/m

6Adam/batch_normalization_55/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/m*
_output_shapes
:@*
dtype0

Adam/conv1d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_10/kernel/m

+Adam/conv1d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_10/kernel/m*"
_output_shapes
:*
dtype0

#Adam/batch_normalization_56/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_56/gamma/m

7Adam/batch_normalization_56/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_56/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_56/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_56/beta/m

6Adam/batch_normalization_56/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_56/beta/m*
_output_shapes
:*
dtype0

Adam/conv1d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_11/kernel/m

+Adam/conv1d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_11/kernel/m*"
_output_shapes
:*
dtype0

#Adam/batch_normalization_57/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_57/gamma/m

7Adam/batch_normalization_57/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_57/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_57/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_57/beta/m

6Adam/batch_normalization_57/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_57/beta/m*
_output_shapes
:*
dtype0

Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_46/kernel/m

*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_45/kernel/v

*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v*
_output_shapes

:@*
dtype0

#Adam/batch_normalization_55/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_55/gamma/v

7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_55/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_55/beta/v

6Adam/batch_normalization_55/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/v*
_output_shapes
:@*
dtype0

Adam/conv1d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_10/kernel/v

+Adam/conv1d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_10/kernel/v*"
_output_shapes
:*
dtype0

#Adam/batch_normalization_56/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_56/gamma/v

7Adam/batch_normalization_56/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_56/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_56/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_56/beta/v

6Adam/batch_normalization_56/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_56/beta/v*
_output_shapes
:*
dtype0

Adam/conv1d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_11/kernel/v

+Adam/conv1d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_11/kernel/v*"
_output_shapes
:*
dtype0

#Adam/batch_normalization_57/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_57/gamma/v

7Adam/batch_normalization_57/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_57/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_57/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_57/beta/v

6Adam/batch_normalization_57/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_57/beta/v*
_output_shapes
:*
dtype0

Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_46/kernel/v

*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v*
_output_shapes

:@*
dtype0

NoOpNoOp
©P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*äO
valueÚOB×O BÐO
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

Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem¢m£m¤*m¥0m¦1m§<m¨Bm©CmªRm«v¬v­v®*v¯0v°1v±<v²Bv³Cv´Rvµ
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
\layer_metrics
]metrics
^non_trainable_variables
trainable_variables
	variables
_layer_regularization_losses

`layers
 
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses
alayer_metrics
bmetrics
cnon_trainable_variables
trainable_variables
	variables
dlayer_regularization_losses

elayers
 
ge
VARIABLE_VALUEbatch_normalization_55/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_55/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_55/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_55/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
flayer_metrics
gmetrics
hnon_trainable_variables
trainable_variables
 	variables
ilayer_regularization_losses

jlayers
 
 
 
­
"regularization_losses
klayer_metrics
lmetrics
mnon_trainable_variables
#trainable_variables
$	variables
nlayer_regularization_losses

olayers
 
 
 
­
&regularization_losses
player_metrics
qmetrics
rnon_trainable_variables
'trainable_variables
(	variables
slayer_regularization_losses

tlayers
\Z
VARIABLE_VALUEconv1d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0
­
+regularization_losses
ulayer_metrics
vmetrics
wnon_trainable_variables
,trainable_variables
-	variables
xlayer_regularization_losses

ylayers
 
ge
VARIABLE_VALUEbatch_normalization_56/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_56/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_56/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_56/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
zlayer_metrics
{metrics
|non_trainable_variables
5trainable_variables
6	variables
}layer_regularization_losses

~layers
 
 
 
±
8regularization_losses
layer_metrics
metrics
non_trainable_variables
9trainable_variables
:	variables
 layer_regularization_losses
layers
\Z
VARIABLE_VALUEconv1d_11/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

<0

<0
²
=regularization_losses
layer_metrics
metrics
non_trainable_variables
>trainable_variables
?	variables
 layer_regularization_losses
layers
 
ge
VARIABLE_VALUEbatch_normalization_57/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_57/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_57/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_57/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
layer_metrics
metrics
non_trainable_variables
Gtrainable_variables
H	variables
 layer_regularization_losses
layers
 
 
 
²
Jregularization_losses
layer_metrics
metrics
non_trainable_variables
Ktrainable_variables
L	variables
 layer_regularization_losses
layers
 
 
 
²
Nregularization_losses
layer_metrics
metrics
non_trainable_variables
Otrainable_variables
P	variables
 layer_regularization_losses
layers
[Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

R0

R0
²
Sregularization_losses
layer_metrics
metrics
non_trainable_variables
Ttrainable_variables
U	variables
 layer_regularization_losses
layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
*
0
1
22
33
D4
E5
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
8

total

count
 	variables
¡	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

 	variables
~|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_55/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_55/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_56/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_56/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_11/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_57/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_57/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_55/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_55/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_56/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_56/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_11/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_57/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_57/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_18Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
°
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18dense_45/kernel&batch_normalization_55/moving_variancebatch_normalization_55/gamma"batch_normalization_55/moving_meanbatch_normalization_55/betaconv1d_10/kernel&batch_normalization_56/moving_variancebatch_normalization_56/gamma"batch_normalization_56/moving_meanbatch_normalization_56/betaconv1d_11/kernel&batch_normalization_57/moving_variancebatch_normalization_57/gamma"batch_normalization_57/moving_meanbatch_normalization_57/betadense_46/kernel*
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
GPU 2J 8 */
f*R(
&__inference_signature_wrapper_41916797
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_45/kernel/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp$conv1d_10/kernel/Read/ReadVariableOp0batch_normalization_56/gamma/Read/ReadVariableOp/batch_normalization_56/beta/Read/ReadVariableOp6batch_normalization_56/moving_mean/Read/ReadVariableOp:batch_normalization_56/moving_variance/Read/ReadVariableOp$conv1d_11/kernel/Read/ReadVariableOp0batch_normalization_57/gamma/Read/ReadVariableOp/batch_normalization_57/beta/Read/ReadVariableOp6batch_normalization_57/moving_mean/Read/ReadVariableOp:batch_normalization_57/moving_variance/Read/ReadVariableOp#dense_46/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_55/beta/m/Read/ReadVariableOp+Adam/conv1d_10/kernel/m/Read/ReadVariableOp7Adam/batch_normalization_56/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_56/beta/m/Read/ReadVariableOp+Adam/conv1d_11/kernel/m/Read/ReadVariableOp7Adam/batch_normalization_57/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_57/beta/m/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_55/beta/v/Read/ReadVariableOp+Adam/conv1d_10/kernel/v/Read/ReadVariableOp7Adam/batch_normalization_56/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_56/beta/v/Read/ReadVariableOp+Adam/conv1d_11/kernel/v/Read/ReadVariableOp7Adam/batch_normalization_57/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_57/beta/v/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
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
!__inference__traced_save_41917785
ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_45/kernelbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv1d_10/kernelbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv1d_11/kernelbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_variancedense_46/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_45/kernel/m#Adam/batch_normalization_55/gamma/m"Adam/batch_normalization_55/beta/mAdam/conv1d_10/kernel/m#Adam/batch_normalization_56/gamma/m"Adam/batch_normalization_56/beta/mAdam/conv1d_11/kernel/m#Adam/batch_normalization_57/gamma/m"Adam/batch_normalization_57/beta/mAdam/dense_46/kernel/mAdam/dense_45/kernel/v#Adam/batch_normalization_55/gamma/v"Adam/batch_normalization_55/beta/vAdam/conv1d_10/kernel/v#Adam/batch_normalization_56/gamma/v"Adam/batch_normalization_56/beta/vAdam/conv1d_11/kernel/v#Adam/batch_normalization_57/gamma/v"Adam/batch_normalization_57/beta/vAdam/dense_46/kernel/v*7
Tin0
.2,*
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
$__inference__traced_restore_41917924§·
Ñ
¡
F__inference_dense_46_layer_call_and_return_conditional_losses_41916518

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û0
Ï
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41916299

inputs
assignmovingavg_41916274
assignmovingavg_1_41916280)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916274*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41916274*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916274*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916274*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41916274AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916274*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916280*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41916280*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916280*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916280*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41916280AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916280*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»

Þ
0__inference_Discriminator_layer_call_fn_41916750
input_18
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
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *T
fORM
K__inference_Discriminator_layer_call_and_return_conditional_losses_419167152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
ó
¡
F__inference_dense_45_layer_call_and_return_conditional_losses_41917104

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ï
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917469

inputs
assignmovingavg_41917444
assignmovingavg_1_41917450)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917444*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41917444*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917444*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917444*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41917444AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917444*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917450*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41917450*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917450*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917450*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41917450AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917450*
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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô:
¥
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916715

inputs
dense_45_41916670#
batch_normalization_55_41916673#
batch_normalization_55_41916675#
batch_normalization_55_41916677#
batch_normalization_55_41916679
conv1d_10_41916684#
batch_normalization_56_41916687#
batch_normalization_56_41916689#
batch_normalization_56_41916691#
batch_normalization_56_41916693
conv1d_11_41916697#
batch_normalization_57_41916700#
batch_normalization_57_41916702#
batch_normalization_57_41916704#
batch_normalization_57_41916706
dense_46_41916711
identity¢.batch_normalization_55/StatefulPartitionedCall¢.batch_normalization_56/StatefulPartitionedCall¢.batch_normalization_57/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall
 dense_45/StatefulPartitionedCallStatefulPartitionedCallinputsdense_45_41916670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_419161592"
 dense_45/StatefulPartitionedCallÉ
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0batch_normalization_55_41916673batch_normalization_55_41916675batch_normalization_55_41916677batch_normalization_55_41916679*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_4191585720
.batch_normalization_55/StatefulPartitionedCall
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_30_layer_call_and_return_conditional_losses_419162112
re_lu_30/PartitionedCallû
reshape_10/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_419162322
reshape_10/PartitionedCallª
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv1d_10_41916684*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_10_layer_call_and_return_conditional_losses_419162522#
!conv1d_10/StatefulPartitionedCallÎ
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0batch_normalization_56_41916687batch_normalization_56_41916689batch_normalization_56_41916691batch_normalization_56_41916693*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_4191631920
.batch_normalization_56/StatefulPartitionedCall
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_31_layer_call_and_return_conditional_losses_419163602
re_lu_31/PartitionedCall¨
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_31/PartitionedCall:output:0conv1d_11_41916697*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_11_layer_call_and_return_conditional_losses_419163802#
!conv1d_11/StatefulPartitionedCallÎ
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_57_41916700batch_normalization_57_41916702batch_normalization_57_41916704batch_normalization_57_41916706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_4191644720
.batch_normalization_57/StatefulPartitionedCall
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_32_layer_call_and_return_conditional_losses_419164882
re_lu_32/PartitionedCall÷
flatten_10/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_419165022
flatten_10/PartitionedCall¢
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_46_41916711*
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
GPU 2J 8 *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_419165182"
 dense_46/StatefulPartitionedCall
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
¬
9__inference_batch_normalization_57_layer_call_fn_41917502

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
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_419161042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
b
F__inference_re_lu_30_layer_call_and_return_conditional_losses_41917198

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¼
G__inference_conv1d_10_layer_call_and_return_conditional_losses_41917233

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
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¬
9__inference_batch_normalization_56_layer_call_fn_41917391

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
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_419162992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
G__inference_conv1d_11_layer_call_and_return_conditional_losses_41917426

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
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917571

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
r
,__inference_conv1d_10_layer_call_fn_41917240

inputs
unknown
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_10_layer_call_and_return_conditional_losses_419162522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¬
9__inference_batch_normalization_57_layer_call_fn_41917584

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
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_419164272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ï
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917276

inputs
assignmovingavg_41917251
assignmovingavg_1_41917257)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917251*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41917251*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917251*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917251*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41917251AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917251*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917257*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41917257*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917257*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917257*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41917257AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917257*
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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡­
ä
#__inference__wrapped_model_41915728
input_189
5discriminator_dense_45_matmul_readvariableop_resourceJ
Fdiscriminator_batch_normalization_55_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_55_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_55_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_55_batchnorm_readvariableop_2_resourceG
Cdiscriminator_conv1d_10_conv1d_expanddims_1_readvariableop_resourceJ
Fdiscriminator_batch_normalization_56_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_56_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_56_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_56_batchnorm_readvariableop_2_resourceG
Cdiscriminator_conv1d_11_conv1d_expanddims_1_readvariableop_resourceJ
Fdiscriminator_batch_normalization_57_batchnorm_readvariableop_resourceN
Jdiscriminator_batch_normalization_57_batchnorm_mul_readvariableop_resourceL
Hdiscriminator_batch_normalization_57_batchnorm_readvariableop_1_resourceL
Hdiscriminator_batch_normalization_57_batchnorm_readvariableop_2_resource9
5discriminator_dense_46_matmul_readvariableop_resource
identity¢=Discriminator/batch_normalization_55/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOp¢=Discriminator/batch_normalization_56/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOp¢=Discriminator/batch_normalization_57/batchnorm/ReadVariableOp¢?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_1¢?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_2¢ADiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOp¢:Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢:Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢,Discriminator/dense_45/MatMul/ReadVariableOp¢,Discriminator/dense_46/MatMul/ReadVariableOpÒ
,Discriminator/dense_45/MatMul/ReadVariableOpReadVariableOp5discriminator_dense_45_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,Discriminator/dense_45/MatMul/ReadVariableOpº
Discriminator/dense_45/MatMulMatMulinput_184Discriminator/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Discriminator/dense_45/MatMul
=Discriminator/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02?
=Discriminator/batch_normalization_55/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_55/batchnorm/add/y
2Discriminator/batch_normalization_55/batchnorm/addAddV2EDiscriminator/batch_normalization_55/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:@24
2Discriminator/batch_normalization_55/batchnorm/addÒ
4Discriminator/batch_normalization_55/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:@26
4Discriminator/batch_normalization_55/batchnorm/Rsqrt
ADiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02C
ADiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_55/batchnorm/mulMul8Discriminator/batch_normalization_55/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@24
2Discriminator/batch_normalization_55/batchnorm/mul
4Discriminator/batch_normalization_55/batchnorm/mul_1Mul'Discriminator/dense_45/MatMul:product:06Discriminator/batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@26
4Discriminator/batch_normalization_55/batchnorm/mul_1
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_55_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_55/batchnorm/mul_2MulGDiscriminator/batch_normalization_55/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:@26
4Discriminator/batch_normalization_55/batchnorm/mul_2
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_55_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02A
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_55/batchnorm/subSubGDiscriminator/batch_normalization_55/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@24
2Discriminator/batch_normalization_55/batchnorm/sub
4Discriminator/batch_normalization_55/batchnorm/add_1AddV28Discriminator/batch_normalization_55/batchnorm/mul_1:z:06Discriminator/batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@26
4Discriminator/batch_normalization_55/batchnorm/add_1®
Discriminator/re_lu_30/ReluRelu8Discriminator/batch_normalization_55/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Discriminator/re_lu_30/Relu
Discriminator/reshape_10/ShapeShape)Discriminator/re_lu_30/Relu:activations:0*
T0*
_output_shapes
:2 
Discriminator/reshape_10/Shape¦
,Discriminator/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Discriminator/reshape_10/strided_slice/stackª
.Discriminator/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Discriminator/reshape_10/strided_slice/stack_1ª
.Discriminator/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Discriminator/reshape_10/strided_slice/stack_2ø
&Discriminator/reshape_10/strided_sliceStridedSlice'Discriminator/reshape_10/Shape:output:05Discriminator/reshape_10/strided_slice/stack:output:07Discriminator/reshape_10/strided_slice/stack_1:output:07Discriminator/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Discriminator/reshape_10/strided_slice
(Discriminator/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Discriminator/reshape_10/Reshape/shape/1
(Discriminator/reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Discriminator/reshape_10/Reshape/shape/2
&Discriminator/reshape_10/Reshape/shapePack/Discriminator/reshape_10/strided_slice:output:01Discriminator/reshape_10/Reshape/shape/1:output:01Discriminator/reshape_10/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&Discriminator/reshape_10/Reshape/shapeá
 Discriminator/reshape_10/ReshapeReshape)Discriminator/re_lu_30/Relu:activations:0/Discriminator/reshape_10/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 Discriminator/reshape_10/Reshape©
-Discriminator/conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-Discriminator/conv1d_10/conv1d/ExpandDims/dim
)Discriminator/conv1d_10/conv1d/ExpandDims
ExpandDims)Discriminator/reshape_10/Reshape:output:06Discriminator/conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)Discriminator/conv1d_10/conv1d/ExpandDims
:Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCdiscriminator_conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¤
/Discriminator/conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/Discriminator/conv1d_10/conv1d/ExpandDims_1/dim
+Discriminator/conv1d_10/conv1d/ExpandDims_1
ExpandDimsBDiscriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:08Discriminator/conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+Discriminator/conv1d_10/conv1d/ExpandDims_1
Discriminator/conv1d_10/conv1dConv2D2Discriminator/conv1d_10/conv1d/ExpandDims:output:04Discriminator/conv1d_10/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2 
Discriminator/conv1d_10/conv1dÚ
&Discriminator/conv1d_10/conv1d/SqueezeSqueeze'Discriminator/conv1d_10/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&Discriminator/conv1d_10/conv1d/Squeeze
=Discriminator/batch_normalization_56/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_56_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=Discriminator/batch_normalization_56/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_56/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_56/batchnorm/add/y
2Discriminator/batch_normalization_56/batchnorm/addAddV2EDiscriminator/batch_normalization_56/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_56/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_56/batchnorm/addÒ
4Discriminator/batch_normalization_56/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_56/batchnorm/add:z:0*
T0*
_output_shapes
:26
4Discriminator/batch_normalization_56/batchnorm/Rsqrt
ADiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_56_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
ADiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_56/batchnorm/mulMul8Discriminator/batch_normalization_56/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_56/batchnorm/mul
4Discriminator/batch_normalization_56/batchnorm/mul_1Mul/Discriminator/conv1d_10/conv1d/Squeeze:output:06Discriminator/batch_normalization_56/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4Discriminator/batch_normalization_56/batchnorm/mul_1
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_56_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_56/batchnorm/mul_2MulGDiscriminator/batch_normalization_56/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_56/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4Discriminator/batch_normalization_56/batchnorm/mul_2
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_56_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_56/batchnorm/subSubGDiscriminator/batch_normalization_56/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_56/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_56/batchnorm/sub
4Discriminator/batch_normalization_56/batchnorm/add_1AddV28Discriminator/batch_normalization_56/batchnorm/mul_1:z:06Discriminator/batch_normalization_56/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4Discriminator/batch_normalization_56/batchnorm/add_1²
Discriminator/re_lu_31/ReluRelu8Discriminator/batch_normalization_56/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Discriminator/re_lu_31/Relu©
-Discriminator/conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-Discriminator/conv1d_11/conv1d/ExpandDims/dim
)Discriminator/conv1d_11/conv1d/ExpandDims
ExpandDims)Discriminator/re_lu_31/Relu:activations:06Discriminator/conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)Discriminator/conv1d_11/conv1d/ExpandDims
:Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCdiscriminator_conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¤
/Discriminator/conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/Discriminator/conv1d_11/conv1d/ExpandDims_1/dim
+Discriminator/conv1d_11/conv1d/ExpandDims_1
ExpandDimsBDiscriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:08Discriminator/conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+Discriminator/conv1d_11/conv1d/ExpandDims_1
Discriminator/conv1d_11/conv1dConv2D2Discriminator/conv1d_11/conv1d/ExpandDims:output:04Discriminator/conv1d_11/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2 
Discriminator/conv1d_11/conv1dÚ
&Discriminator/conv1d_11/conv1d/SqueezeSqueeze'Discriminator/conv1d_11/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&Discriminator/conv1d_11/conv1d/Squeeze
=Discriminator/batch_normalization_57/batchnorm/ReadVariableOpReadVariableOpFdiscriminator_batch_normalization_57_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=Discriminator/batch_normalization_57/batchnorm/ReadVariableOp±
4Discriminator/batch_normalization_57/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4Discriminator/batch_normalization_57/batchnorm/add/y
2Discriminator/batch_normalization_57/batchnorm/addAddV2EDiscriminator/batch_normalization_57/batchnorm/ReadVariableOp:value:0=Discriminator/batch_normalization_57/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_57/batchnorm/addÒ
4Discriminator/batch_normalization_57/batchnorm/RsqrtRsqrt6Discriminator/batch_normalization_57/batchnorm/add:z:0*
T0*
_output_shapes
:26
4Discriminator/batch_normalization_57/batchnorm/Rsqrt
ADiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOpReadVariableOpJdiscriminator_batch_normalization_57_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
ADiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOp
2Discriminator/batch_normalization_57/batchnorm/mulMul8Discriminator/batch_normalization_57/batchnorm/Rsqrt:y:0IDiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_57/batchnorm/mul
4Discriminator/batch_normalization_57/batchnorm/mul_1Mul/Discriminator/conv1d_11/conv1d/Squeeze:output:06Discriminator/batch_normalization_57/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4Discriminator/batch_normalization_57/batchnorm/mul_1
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_1ReadVariableOpHdiscriminator_batch_normalization_57_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_1
4Discriminator/batch_normalization_57/batchnorm/mul_2MulGDiscriminator/batch_normalization_57/batchnorm/ReadVariableOp_1:value:06Discriminator/batch_normalization_57/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4Discriminator/batch_normalization_57/batchnorm/mul_2
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_2ReadVariableOpHdiscriminator_batch_normalization_57_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_2
2Discriminator/batch_normalization_57/batchnorm/subSubGDiscriminator/batch_normalization_57/batchnorm/ReadVariableOp_2:value:08Discriminator/batch_normalization_57/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2Discriminator/batch_normalization_57/batchnorm/sub
4Discriminator/batch_normalization_57/batchnorm/add_1AddV28Discriminator/batch_normalization_57/batchnorm/mul_1:z:06Discriminator/batch_normalization_57/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4Discriminator/batch_normalization_57/batchnorm/add_1²
Discriminator/re_lu_32/ReluRelu8Discriminator/batch_normalization_57/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Discriminator/re_lu_32/Relu
Discriminator/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2 
Discriminator/flatten_10/ConstÕ
 Discriminator/flatten_10/ReshapeReshape)Discriminator/re_lu_32/Relu:activations:0'Discriminator/flatten_10/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 Discriminator/flatten_10/ReshapeÒ
,Discriminator/dense_46/MatMul/ReadVariableOpReadVariableOp5discriminator_dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,Discriminator/dense_46/MatMul/ReadVariableOpÛ
Discriminator/dense_46/MatMulMatMul)Discriminator/flatten_10/Reshape:output:04Discriminator/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Discriminator/dense_46/MatMul¦
Discriminator/dense_46/SigmoidSigmoid'Discriminator/dense_46/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
Discriminator/dense_46/Sigmoidæ
IdentityIdentity"Discriminator/dense_46/Sigmoid:y:0>^Discriminator/batch_normalization_55/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_55/batchnorm/mul/ReadVariableOp>^Discriminator/batch_normalization_56/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_56/batchnorm/mul/ReadVariableOp>^Discriminator/batch_normalization_57/batchnorm/ReadVariableOp@^Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_1@^Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_2B^Discriminator/batch_normalization_57/batchnorm/mul/ReadVariableOp;^Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp;^Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp-^Discriminator/dense_45/MatMul/ReadVariableOp-^Discriminator/dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2~
=Discriminator/batch_normalization_55/batchnorm/ReadVariableOp=Discriminator/batch_normalization_55/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_55/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_55/batchnorm/mul/ReadVariableOp2~
=Discriminator/batch_normalization_56/batchnorm/ReadVariableOp=Discriminator/batch_normalization_56/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_56/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_56/batchnorm/mul/ReadVariableOp2~
=Discriminator/batch_normalization_57/batchnorm/ReadVariableOp=Discriminator/batch_normalization_57/batchnorm/ReadVariableOp2
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_1?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_12
?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_2?Discriminator/batch_normalization_57/batchnorm/ReadVariableOp_22
ADiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOpADiscriminator/batch_normalization_57/batchnorm/mul/ReadVariableOp2x
:Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:Discriminator/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2x
:Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:Discriminator/conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2\
,Discriminator/dense_45/MatMul/ReadVariableOp,Discriminator/dense_45/MatMul/ReadVariableOp2\
,Discriminator/dense_46/MatMul/ReadVariableOp,Discriminator/dense_46/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
Ò
r
,__inference_conv1d_11_layer_call_fn_41917433

inputs
unknown
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_11_layer_call_and_return_conditional_losses_419163802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917147

inputs
assignmovingavg_41917122
assignmovingavg_1_41917128)
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
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

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917122*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41917122*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917122*
_output_shapes
:@2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917122*
_output_shapes
:@2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41917122AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917122*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917128*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41917128*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917128*
_output_shapes
:@2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917128*
_output_shapes
:@2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41917128AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917128*
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
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û0
Ï
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917551

inputs
assignmovingavg_41917526
assignmovingavg_1_41917532)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917526*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41917526*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917526*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917526*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41917526AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917526*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917532*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41917532*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917532*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917532*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41917532AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917532*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41915857

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ú:
§
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916579
input_18
dense_45_41916534#
batch_normalization_55_41916537#
batch_normalization_55_41916539#
batch_normalization_55_41916541#
batch_normalization_55_41916543
conv1d_10_41916548#
batch_normalization_56_41916551#
batch_normalization_56_41916553#
batch_normalization_56_41916555#
batch_normalization_56_41916557
conv1d_11_41916561#
batch_normalization_57_41916564#
batch_normalization_57_41916566#
batch_normalization_57_41916568#
batch_normalization_57_41916570
dense_46_41916575
identity¢.batch_normalization_55/StatefulPartitionedCall¢.batch_normalization_56/StatefulPartitionedCall¢.batch_normalization_57/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall
 dense_45/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_45_41916534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_419161592"
 dense_45/StatefulPartitionedCallÉ
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0batch_normalization_55_41916537batch_normalization_55_41916539batch_normalization_55_41916541batch_normalization_55_41916543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_4191585720
.batch_normalization_55/StatefulPartitionedCall
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_30_layer_call_and_return_conditional_losses_419162112
re_lu_30/PartitionedCallû
reshape_10/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_419162322
reshape_10/PartitionedCallª
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv1d_10_41916548*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_10_layer_call_and_return_conditional_losses_419162522#
!conv1d_10/StatefulPartitionedCallÎ
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0batch_normalization_56_41916551batch_normalization_56_41916553batch_normalization_56_41916555batch_normalization_56_41916557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_4191631920
.batch_normalization_56/StatefulPartitionedCall
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_31_layer_call_and_return_conditional_losses_419163602
re_lu_31/PartitionedCall¨
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_31/PartitionedCall:output:0conv1d_11_41916561*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_11_layer_call_and_return_conditional_losses_419163802#
!conv1d_11/StatefulPartitionedCallÎ
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_57_41916564batch_normalization_57_41916566batch_normalization_57_41916568batch_normalization_57_41916570*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_4191644720
.batch_normalization_57/StatefulPartitionedCall
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_32_layer_call_and_return_conditional_losses_419164882
re_lu_32/PartitionedCall÷
flatten_10/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_419165022
flatten_10/PartitionedCall¢
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_46_41916575*
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
GPU 2J 8 *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_419165182"
 dense_46/StatefulPartitionedCall
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
Ä
b
F__inference_re_lu_32_layer_call_and_return_conditional_losses_41917602

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ô
&__inference_signature_wrapper_41916797
input_18
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *,
f'R%
#__inference__wrapped_model_419157282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
¦
G
+__inference_re_lu_31_layer_call_fn_41917414

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_31_layer_call_and_return_conditional_losses_419163602
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¬
9__inference_batch_normalization_56_layer_call_fn_41917404

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
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_419163192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
b
F__inference_re_lu_31_layer_call_and_return_conditional_losses_41917409

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ï
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41916104

inputs
assignmovingavg_41916079
assignmovingavg_1_41916085)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916079*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41916079*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916079*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916079*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41916079AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916079*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916085*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41916085*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916085*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916085*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41916085AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916085*
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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

Ü
0__inference_Discriminator_layer_call_fn_41917097

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
identity¢StatefulPartitionedCall¶
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
GPU 2J 8 *T
fORM
K__inference_Discriminator_layer_call_and_return_conditional_losses_419167152
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
ô:
§
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916531
input_18
dense_45_41916168#
batch_normalization_55_41916197#
batch_normalization_55_41916199#
batch_normalization_55_41916201#
batch_normalization_55_41916203
conv1d_10_41916261#
batch_normalization_56_41916346#
batch_normalization_56_41916348#
batch_normalization_56_41916350#
batch_normalization_56_41916352
conv1d_11_41916389#
batch_normalization_57_41916474#
batch_normalization_57_41916476#
batch_normalization_57_41916478#
batch_normalization_57_41916480
dense_46_41916527
identity¢.batch_normalization_55/StatefulPartitionedCall¢.batch_normalization_56/StatefulPartitionedCall¢.batch_normalization_57/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall
 dense_45/StatefulPartitionedCallStatefulPartitionedCallinput_18dense_45_41916168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_419161592"
 dense_45/StatefulPartitionedCallÇ
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0batch_normalization_55_41916197batch_normalization_55_41916199batch_normalization_55_41916201batch_normalization_55_41916203*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_4191582420
.batch_normalization_55/StatefulPartitionedCall
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_30_layer_call_and_return_conditional_losses_419162112
re_lu_30/PartitionedCallû
reshape_10/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_419162322
reshape_10/PartitionedCallª
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv1d_10_41916261*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_10_layer_call_and_return_conditional_losses_419162522#
!conv1d_10/StatefulPartitionedCallÌ
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0batch_normalization_56_41916346batch_normalization_56_41916348batch_normalization_56_41916350batch_normalization_56_41916352*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_4191629920
.batch_normalization_56/StatefulPartitionedCall
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_31_layer_call_and_return_conditional_losses_419163602
re_lu_31/PartitionedCall¨
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_31/PartitionedCall:output:0conv1d_11_41916389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_11_layer_call_and_return_conditional_losses_419163802#
!conv1d_11/StatefulPartitionedCallÌ
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_57_41916474batch_normalization_57_41916476batch_normalization_57_41916478batch_normalization_57_41916480*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_4191642720
.batch_normalization_57/StatefulPartitionedCall
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_32_layer_call_and_return_conditional_losses_419164882
re_lu_32/PartitionedCall÷
flatten_10/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_419165022
flatten_10/PartitionedCall¢
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_46_41916527*
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
GPU 2J 8 *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_419165182"
 dense_46/StatefulPartitionedCall
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
´
b
F__inference_re_lu_30_layer_call_and_return_conditional_losses_41916211

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»
¬
9__inference_batch_normalization_55_layer_call_fn_41917180

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
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_419158242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë

T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917378

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
¬
9__inference_batch_normalization_56_layer_call_fn_41917309

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
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_419159642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917167

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
Ê
K__inference_Discriminator_layer_call_and_return_conditional_losses_41917023

inputs+
'dense_45_matmul_readvariableop_resource<
8batch_normalization_55_batchnorm_readvariableop_resource@
<batch_normalization_55_batchnorm_mul_readvariableop_resource>
:batch_normalization_55_batchnorm_readvariableop_1_resource>
:batch_normalization_55_batchnorm_readvariableop_2_resource9
5conv1d_10_conv1d_expanddims_1_readvariableop_resource<
8batch_normalization_56_batchnorm_readvariableop_resource@
<batch_normalization_56_batchnorm_mul_readvariableop_resource>
:batch_normalization_56_batchnorm_readvariableop_1_resource>
:batch_normalization_56_batchnorm_readvariableop_2_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource<
8batch_normalization_57_batchnorm_readvariableop_resource@
<batch_normalization_57_batchnorm_mul_readvariableop_resource>
:batch_normalization_57_batchnorm_readvariableop_1_resource>
:batch_normalization_57_batchnorm_readvariableop_2_resource+
'dense_46_matmul_readvariableop_resource
identity¢/batch_normalization_55/batchnorm/ReadVariableOp¢1batch_normalization_55/batchnorm/ReadVariableOp_1¢1batch_normalization_55/batchnorm/ReadVariableOp_2¢3batch_normalization_55/batchnorm/mul/ReadVariableOp¢/batch_normalization_56/batchnorm/ReadVariableOp¢1batch_normalization_56/batchnorm/ReadVariableOp_1¢1batch_normalization_56/batchnorm/ReadVariableOp_2¢3batch_normalization_56/batchnorm/mul/ReadVariableOp¢/batch_normalization_57/batchnorm/ReadVariableOp¢1batch_normalization_57/batchnorm/ReadVariableOp_1¢1batch_normalization_57/batchnorm/ReadVariableOp_2¢3batch_normalization_57/batchnorm/mul/ReadVariableOp¢,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢dense_45/MatMul/ReadVariableOp¢dense_46/MatMul/ReadVariableOp¨
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_45/MatMul/ReadVariableOp
dense_45/MatMulMatMulinputs&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_45/MatMul×
/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_55/batchnorm/ReadVariableOp
&batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_55/batchnorm/add/yä
$batch_normalization_55/batchnorm/addAddV27batch_normalization_55/batchnorm/ReadVariableOp:value:0/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/add¨
&batch_normalization_55/batchnorm/RsqrtRsqrt(batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_55/batchnorm/Rsqrtã
3batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_55/batchnorm/mul/ReadVariableOpá
$batch_normalization_55/batchnorm/mulMul*batch_normalization_55/batchnorm/Rsqrt:y:0;batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/mulÎ
&batch_normalization_55/batchnorm/mul_1Muldense_45/MatMul:product:0(batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_55/batchnorm/mul_1Ý
1batch_normalization_55/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_55_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1batch_normalization_55/batchnorm/ReadVariableOp_1á
&batch_normalization_55/batchnorm/mul_2Mul9batch_normalization_55/batchnorm/ReadVariableOp_1:value:0(batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_55/batchnorm/mul_2Ý
1batch_normalization_55/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_55_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype023
1batch_normalization_55/batchnorm/ReadVariableOp_2ß
$batch_normalization_55/batchnorm/subSub9batch_normalization_55/batchnorm/ReadVariableOp_2:value:0*batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/subá
&batch_normalization_55/batchnorm/add_1AddV2*batch_normalization_55/batchnorm/mul_1:z:0(batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_55/batchnorm/add_1
re_lu_30/ReluRelu*batch_normalization_55/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_30/Reluo
reshape_10/ShapeShapere_lu_30/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2¤
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1z
reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/2×
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0#reshape_10/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shape©
reshape_10/ReshapeReshapere_lu_30/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape_10/Reshape
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_10/conv1d/ExpandDims/dimÉ
conv1d_10/conv1d/ExpandDims
ExpandDimsreshape_10/Reshape:output:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/ExpandDimsÖ
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimß
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_10/conv1d/ExpandDims_1Þ
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_10/conv1d°
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/Squeeze×
/batch_normalization_56/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_56_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_56/batchnorm/ReadVariableOp
&batch_normalization_56/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_56/batchnorm/add/yä
$batch_normalization_56/batchnorm/addAddV27batch_normalization_56/batchnorm/ReadVariableOp:value:0/batch_normalization_56/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/add¨
&batch_normalization_56/batchnorm/RsqrtRsqrt(batch_normalization_56/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_56/batchnorm/Rsqrtã
3batch_normalization_56/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_56_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_56/batchnorm/mul/ReadVariableOpá
$batch_normalization_56/batchnorm/mulMul*batch_normalization_56/batchnorm/Rsqrt:y:0;batch_normalization_56/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/mulÚ
&batch_normalization_56/batchnorm/mul_1Mul!conv1d_10/conv1d/Squeeze:output:0(batch_normalization_56/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_56/batchnorm/mul_1Ý
1batch_normalization_56/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_56_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_56/batchnorm/ReadVariableOp_1á
&batch_normalization_56/batchnorm/mul_2Mul9batch_normalization_56/batchnorm/ReadVariableOp_1:value:0(batch_normalization_56/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_56/batchnorm/mul_2Ý
1batch_normalization_56/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_56_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_56/batchnorm/ReadVariableOp_2ß
$batch_normalization_56/batchnorm/subSub9batch_normalization_56/batchnorm/ReadVariableOp_2:value:0*batch_normalization_56/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/subå
&batch_normalization_56/batchnorm/add_1AddV2*batch_normalization_56/batchnorm/mul_1:z:0(batch_normalization_56/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_56/batchnorm/add_1
re_lu_31/ReluRelu*batch_normalization_56/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_31/Relu
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_11/conv1d/ExpandDims/dimÉ
conv1d_11/conv1d/ExpandDims
ExpandDimsre_lu_31/Relu:activations:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/ExpandDimsÖ
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimß
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_11/conv1d/ExpandDims_1Þ
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_11/conv1d°
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/Squeeze×
/batch_normalization_57/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_57_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_57/batchnorm/ReadVariableOp
&batch_normalization_57/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_57/batchnorm/add/yä
$batch_normalization_57/batchnorm/addAddV27batch_normalization_57/batchnorm/ReadVariableOp:value:0/batch_normalization_57/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/add¨
&batch_normalization_57/batchnorm/RsqrtRsqrt(batch_normalization_57/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_57/batchnorm/Rsqrtã
3batch_normalization_57/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_57_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_57/batchnorm/mul/ReadVariableOpá
$batch_normalization_57/batchnorm/mulMul*batch_normalization_57/batchnorm/Rsqrt:y:0;batch_normalization_57/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/mulÚ
&batch_normalization_57/batchnorm/mul_1Mul!conv1d_11/conv1d/Squeeze:output:0(batch_normalization_57/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_57/batchnorm/mul_1Ý
1batch_normalization_57/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_57_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_57/batchnorm/ReadVariableOp_1á
&batch_normalization_57/batchnorm/mul_2Mul9batch_normalization_57/batchnorm/ReadVariableOp_1:value:0(batch_normalization_57/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_57/batchnorm/mul_2Ý
1batch_normalization_57/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_57_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_57/batchnorm/ReadVariableOp_2ß
$batch_normalization_57/batchnorm/subSub9batch_normalization_57/batchnorm/ReadVariableOp_2:value:0*batch_normalization_57/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/subå
&batch_normalization_57/batchnorm/add_1AddV2*batch_normalization_57/batchnorm/mul_1:z:0(batch_normalization_57/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_57/batchnorm/add_1
re_lu_32/ReluRelu*batch_normalization_57/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_32/Reluu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
flatten_10/Const
flatten_10/ReshapeReshapere_lu_32/Relu:activations:0flatten_10/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
flatten_10/Reshape¨
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_46/MatMul/ReadVariableOp£
dense_46/MatMulMatMulflatten_10/Reshape:output:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_46/MatMul|
dense_46/SigmoidSigmoiddense_46/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_46/Sigmoidø
IdentityIdentitydense_46/Sigmoid:y:00^batch_normalization_55/batchnorm/ReadVariableOp2^batch_normalization_55/batchnorm/ReadVariableOp_12^batch_normalization_55/batchnorm/ReadVariableOp_24^batch_normalization_55/batchnorm/mul/ReadVariableOp0^batch_normalization_56/batchnorm/ReadVariableOp2^batch_normalization_56/batchnorm/ReadVariableOp_12^batch_normalization_56/batchnorm/ReadVariableOp_24^batch_normalization_56/batchnorm/mul/ReadVariableOp0^batch_normalization_57/batchnorm/ReadVariableOp2^batch_normalization_57/batchnorm/ReadVariableOp_12^batch_normalization_57/batchnorm/ReadVariableOp_24^batch_normalization_57/batchnorm/mul/ReadVariableOp-^conv1d_10/conv1d/ExpandDims_1/ReadVariableOp-^conv1d_11/conv1d/ExpandDims_1/ReadVariableOp^dense_45/MatMul/ReadVariableOp^dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2b
/batch_normalization_55/batchnorm/ReadVariableOp/batch_normalization_55/batchnorm/ReadVariableOp2f
1batch_normalization_55/batchnorm/ReadVariableOp_11batch_normalization_55/batchnorm/ReadVariableOp_12f
1batch_normalization_55/batchnorm/ReadVariableOp_21batch_normalization_55/batchnorm/ReadVariableOp_22j
3batch_normalization_55/batchnorm/mul/ReadVariableOp3batch_normalization_55/batchnorm/mul/ReadVariableOp2b
/batch_normalization_56/batchnorm/ReadVariableOp/batch_normalization_56/batchnorm/ReadVariableOp2f
1batch_normalization_56/batchnorm/ReadVariableOp_11batch_normalization_56/batchnorm/ReadVariableOp_12f
1batch_normalization_56/batchnorm/ReadVariableOp_21batch_normalization_56/batchnorm/ReadVariableOp_22j
3batch_normalization_56/batchnorm/mul/ReadVariableOp3batch_normalization_56/batchnorm/mul/ReadVariableOp2b
/batch_normalization_57/batchnorm/ReadVariableOp/batch_normalization_57/batchnorm/ReadVariableOp2f
1batch_normalization_57/batchnorm/ReadVariableOp_11batch_normalization_57/batchnorm/ReadVariableOp_12f
1batch_normalization_57/batchnorm/ReadVariableOp_21batch_normalization_57/batchnorm/ReadVariableOp_22j
3batch_normalization_57/batchnorm/mul/ReadVariableOp3batch_normalization_57/batchnorm/mul/ReadVariableOp2\
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2\
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
 
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916934

inputs+
'dense_45_matmul_readvariableop_resource3
/batch_normalization_55_assignmovingavg_419168115
1batch_normalization_55_assignmovingavg_1_41916817@
<batch_normalization_55_batchnorm_mul_readvariableop_resource<
8batch_normalization_55_batchnorm_readvariableop_resource9
5conv1d_10_conv1d_expanddims_1_readvariableop_resource3
/batch_normalization_56_assignmovingavg_419168615
1batch_normalization_56_assignmovingavg_1_41916867@
<batch_normalization_56_batchnorm_mul_readvariableop_resource<
8batch_normalization_56_batchnorm_readvariableop_resource9
5conv1d_11_conv1d_expanddims_1_readvariableop_resource3
/batch_normalization_57_assignmovingavg_419169025
1batch_normalization_57_assignmovingavg_1_41916908@
<batch_normalization_57_batchnorm_mul_readvariableop_resource<
8batch_normalization_57_batchnorm_readvariableop_resource+
'dense_46_matmul_readvariableop_resource
identity¢:batch_normalization_55/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_55/AssignMovingAvg/ReadVariableOp¢<batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_55/batchnorm/ReadVariableOp¢3batch_normalization_55/batchnorm/mul/ReadVariableOp¢:batch_normalization_56/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_56/AssignMovingAvg/ReadVariableOp¢<batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_56/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_56/batchnorm/ReadVariableOp¢3batch_normalization_56/batchnorm/mul/ReadVariableOp¢:batch_normalization_57/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_57/AssignMovingAvg/ReadVariableOp¢<batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_57/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_57/batchnorm/ReadVariableOp¢3batch_normalization_57/batchnorm/mul/ReadVariableOp¢,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp¢,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp¢dense_45/MatMul/ReadVariableOp¢dense_46/MatMul/ReadVariableOp¨
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_45/MatMul/ReadVariableOp
dense_45/MatMulMatMulinputs&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_45/MatMul¸
5batch_normalization_55/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_55/moments/mean/reduction_indicesç
#batch_normalization_55/moments/meanMeandense_45/MatMul:product:0>batch_normalization_55/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2%
#batch_normalization_55/moments/meanÁ
+batch_normalization_55/moments/StopGradientStopGradient,batch_normalization_55/moments/mean:output:0*
T0*
_output_shapes

:@2-
+batch_normalization_55/moments/StopGradientü
0batch_normalization_55/moments/SquaredDifferenceSquaredDifferencedense_45/MatMul:product:04batch_normalization_55/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@22
0batch_normalization_55/moments/SquaredDifferenceÀ
9batch_normalization_55/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_55/moments/variance/reduction_indices
'batch_normalization_55/moments/varianceMean4batch_normalization_55/moments/SquaredDifference:z:0Bbatch_normalization_55/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2)
'batch_normalization_55/moments/varianceÅ
&batch_normalization_55/moments/SqueezeSqueeze,batch_normalization_55/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2(
&batch_normalization_55/moments/SqueezeÍ
(batch_normalization_55/moments/Squeeze_1Squeeze0batch_normalization_55/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2*
(batch_normalization_55/moments/Squeeze_1
,batch_normalization_55/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_55/AssignMovingAvg/41916811*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_55/AssignMovingAvg/decayÚ
5batch_normalization_55/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_55_assignmovingavg_41916811*
_output_shapes
:@*
dtype027
5batch_normalization_55/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_55/AssignMovingAvg/subSub=batch_normalization_55/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_55/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_55/AssignMovingAvg/41916811*
_output_shapes
:@2,
*batch_normalization_55/AssignMovingAvg/subÝ
*batch_normalization_55/AssignMovingAvg/mulMul.batch_normalization_55/AssignMovingAvg/sub:z:05batch_normalization_55/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_55/AssignMovingAvg/41916811*
_output_shapes
:@2,
*batch_normalization_55/AssignMovingAvg/mul½
:batch_normalization_55/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_55_assignmovingavg_41916811.batch_normalization_55/AssignMovingAvg/mul:z:06^batch_normalization_55/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_55/AssignMovingAvg/41916811*
_output_shapes
 *
dtype02<
:batch_normalization_55/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_55/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_55/AssignMovingAvg_1/41916817*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_55/AssignMovingAvg_1/decayà
7batch_normalization_55/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_55_assignmovingavg_1_41916817*
_output_shapes
:@*
dtype029
7batch_normalization_55/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_55/AssignMovingAvg_1/subSub?batch_normalization_55/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_55/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_55/AssignMovingAvg_1/41916817*
_output_shapes
:@2.
,batch_normalization_55/AssignMovingAvg_1/subç
,batch_normalization_55/AssignMovingAvg_1/mulMul0batch_normalization_55/AssignMovingAvg_1/sub:z:07batch_normalization_55/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_55/AssignMovingAvg_1/41916817*
_output_shapes
:@2.
,batch_normalization_55/AssignMovingAvg_1/mulÉ
<batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_55_assignmovingavg_1_419168170batch_normalization_55/AssignMovingAvg_1/mul:z:08^batch_normalization_55/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_55/AssignMovingAvg_1/41916817*
_output_shapes
 *
dtype02>
<batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_55/batchnorm/add/yÞ
$batch_normalization_55/batchnorm/addAddV21batch_normalization_55/moments/Squeeze_1:output:0/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/add¨
&batch_normalization_55/batchnorm/RsqrtRsqrt(batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_55/batchnorm/Rsqrtã
3batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3batch_normalization_55/batchnorm/mul/ReadVariableOpá
$batch_normalization_55/batchnorm/mulMul*batch_normalization_55/batchnorm/Rsqrt:y:0;batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/mulÎ
&batch_normalization_55/batchnorm/mul_1Muldense_45/MatMul:product:0(batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_55/batchnorm/mul_1×
&batch_normalization_55/batchnorm/mul_2Mul/batch_normalization_55/moments/Squeeze:output:0(batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:@2(
&batch_normalization_55/batchnorm/mul_2×
/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/batch_normalization_55/batchnorm/ReadVariableOpÝ
$batch_normalization_55/batchnorm/subSub7batch_normalization_55/batchnorm/ReadVariableOp:value:0*batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2&
$batch_normalization_55/batchnorm/subá
&batch_normalization_55/batchnorm/add_1AddV2*batch_normalization_55/batchnorm/mul_1:z:0(batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&batch_normalization_55/batchnorm/add_1
re_lu_30/ReluRelu*batch_normalization_55/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_30/Reluo
reshape_10/ShapeShapere_lu_30/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2¤
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1z
reshape_10/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/2×
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0#reshape_10/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shape©
reshape_10/ReshapeReshapere_lu_30/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape_10/Reshape
conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_10/conv1d/ExpandDims/dimÉ
conv1d_10/conv1d/ExpandDims
ExpandDimsreshape_10/Reshape:output:0(conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/ExpandDimsÖ
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_10/conv1d/ExpandDims_1/dimß
conv1d_10/conv1d/ExpandDims_1
ExpandDims4conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_10/conv1d/ExpandDims_1Þ
conv1d_10/conv1dConv2D$conv1d_10/conv1d/ExpandDims:output:0&conv1d_10/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_10/conv1d°
conv1d_10/conv1d/SqueezeSqueezeconv1d_10/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_10/conv1d/Squeeze¿
5batch_normalization_56/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_56/moments/mean/reduction_indicesó
#batch_normalization_56/moments/meanMean!conv1d_10/conv1d/Squeeze:output:0>batch_normalization_56/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_56/moments/meanÅ
+batch_normalization_56/moments/StopGradientStopGradient,batch_normalization_56/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_56/moments/StopGradient
0batch_normalization_56/moments/SquaredDifferenceSquaredDifference!conv1d_10/conv1d/Squeeze:output:04batch_normalization_56/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_56/moments/SquaredDifferenceÇ
9batch_normalization_56/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_56/moments/variance/reduction_indices
'batch_normalization_56/moments/varianceMean4batch_normalization_56/moments/SquaredDifference:z:0Bbatch_normalization_56/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_56/moments/varianceÆ
&batch_normalization_56/moments/SqueezeSqueeze,batch_normalization_56/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_56/moments/SqueezeÎ
(batch_normalization_56/moments/Squeeze_1Squeeze0batch_normalization_56/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_56/moments/Squeeze_1
,batch_normalization_56/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_56/AssignMovingAvg/41916861*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_56/AssignMovingAvg/decayÚ
5batch_normalization_56/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_56_assignmovingavg_41916861*
_output_shapes
:*
dtype027
5batch_normalization_56/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_56/AssignMovingAvg/subSub=batch_normalization_56/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_56/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_56/AssignMovingAvg/41916861*
_output_shapes
:2,
*batch_normalization_56/AssignMovingAvg/subÝ
*batch_normalization_56/AssignMovingAvg/mulMul.batch_normalization_56/AssignMovingAvg/sub:z:05batch_normalization_56/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_56/AssignMovingAvg/41916861*
_output_shapes
:2,
*batch_normalization_56/AssignMovingAvg/mul½
:batch_normalization_56/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_56_assignmovingavg_41916861.batch_normalization_56/AssignMovingAvg/mul:z:06^batch_normalization_56/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_56/AssignMovingAvg/41916861*
_output_shapes
 *
dtype02<
:batch_normalization_56/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_56/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_56/AssignMovingAvg_1/41916867*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_56/AssignMovingAvg_1/decayà
7batch_normalization_56/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_56_assignmovingavg_1_41916867*
_output_shapes
:*
dtype029
7batch_normalization_56/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_56/AssignMovingAvg_1/subSub?batch_normalization_56/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_56/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_56/AssignMovingAvg_1/41916867*
_output_shapes
:2.
,batch_normalization_56/AssignMovingAvg_1/subç
,batch_normalization_56/AssignMovingAvg_1/mulMul0batch_normalization_56/AssignMovingAvg_1/sub:z:07batch_normalization_56/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_56/AssignMovingAvg_1/41916867*
_output_shapes
:2.
,batch_normalization_56/AssignMovingAvg_1/mulÉ
<batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_56_assignmovingavg_1_419168670batch_normalization_56/AssignMovingAvg_1/mul:z:08^batch_normalization_56/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_56/AssignMovingAvg_1/41916867*
_output_shapes
 *
dtype02>
<batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_56/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_56/batchnorm/add/yÞ
$batch_normalization_56/batchnorm/addAddV21batch_normalization_56/moments/Squeeze_1:output:0/batch_normalization_56/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/add¨
&batch_normalization_56/batchnorm/RsqrtRsqrt(batch_normalization_56/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_56/batchnorm/Rsqrtã
3batch_normalization_56/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_56_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_56/batchnorm/mul/ReadVariableOpá
$batch_normalization_56/batchnorm/mulMul*batch_normalization_56/batchnorm/Rsqrt:y:0;batch_normalization_56/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/mulÚ
&batch_normalization_56/batchnorm/mul_1Mul!conv1d_10/conv1d/Squeeze:output:0(batch_normalization_56/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_56/batchnorm/mul_1×
&batch_normalization_56/batchnorm/mul_2Mul/batch_normalization_56/moments/Squeeze:output:0(batch_normalization_56/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_56/batchnorm/mul_2×
/batch_normalization_56/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_56_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_56/batchnorm/ReadVariableOpÝ
$batch_normalization_56/batchnorm/subSub7batch_normalization_56/batchnorm/ReadVariableOp:value:0*batch_normalization_56/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_56/batchnorm/subå
&batch_normalization_56/batchnorm/add_1AddV2*batch_normalization_56/batchnorm/mul_1:z:0(batch_normalization_56/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_56/batchnorm/add_1
re_lu_31/ReluRelu*batch_normalization_56/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_31/Relu
conv1d_11/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_11/conv1d/ExpandDims/dimÉ
conv1d_11/conv1d/ExpandDims
ExpandDimsre_lu_31/Relu:activations:0(conv1d_11/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/ExpandDimsÖ
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_11_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_11/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_11/conv1d/ExpandDims_1/dimß
conv1d_11/conv1d/ExpandDims_1
ExpandDims4conv1d_11/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_11/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_11/conv1d/ExpandDims_1Þ
conv1d_11/conv1dConv2D$conv1d_11/conv1d/ExpandDims:output:0&conv1d_11/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_11/conv1d°
conv1d_11/conv1d/SqueezeSqueezeconv1d_11/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_11/conv1d/Squeeze¿
5batch_normalization_57/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_57/moments/mean/reduction_indicesó
#batch_normalization_57/moments/meanMean!conv1d_11/conv1d/Squeeze:output:0>batch_normalization_57/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_57/moments/meanÅ
+batch_normalization_57/moments/StopGradientStopGradient,batch_normalization_57/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_57/moments/StopGradient
0batch_normalization_57/moments/SquaredDifferenceSquaredDifference!conv1d_11/conv1d/Squeeze:output:04batch_normalization_57/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_57/moments/SquaredDifferenceÇ
9batch_normalization_57/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_57/moments/variance/reduction_indices
'batch_normalization_57/moments/varianceMean4batch_normalization_57/moments/SquaredDifference:z:0Bbatch_normalization_57/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_57/moments/varianceÆ
&batch_normalization_57/moments/SqueezeSqueeze,batch_normalization_57/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_57/moments/SqueezeÎ
(batch_normalization_57/moments/Squeeze_1Squeeze0batch_normalization_57/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_57/moments/Squeeze_1
,batch_normalization_57/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_57/AssignMovingAvg/41916902*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_57/AssignMovingAvg/decayÚ
5batch_normalization_57/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_57_assignmovingavg_41916902*
_output_shapes
:*
dtype027
5batch_normalization_57/AssignMovingAvg/ReadVariableOpæ
*batch_normalization_57/AssignMovingAvg/subSub=batch_normalization_57/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_57/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_57/AssignMovingAvg/41916902*
_output_shapes
:2,
*batch_normalization_57/AssignMovingAvg/subÝ
*batch_normalization_57/AssignMovingAvg/mulMul.batch_normalization_57/AssignMovingAvg/sub:z:05batch_normalization_57/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_57/AssignMovingAvg/41916902*
_output_shapes
:2,
*batch_normalization_57/AssignMovingAvg/mul½
:batch_normalization_57/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_57_assignmovingavg_41916902.batch_normalization_57/AssignMovingAvg/mul:z:06^batch_normalization_57/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_57/AssignMovingAvg/41916902*
_output_shapes
 *
dtype02<
:batch_normalization_57/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_57/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_57/AssignMovingAvg_1/41916908*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_57/AssignMovingAvg_1/decayà
7batch_normalization_57/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_57_assignmovingavg_1_41916908*
_output_shapes
:*
dtype029
7batch_normalization_57/AssignMovingAvg_1/ReadVariableOpð
,batch_normalization_57/AssignMovingAvg_1/subSub?batch_normalization_57/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_57/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_57/AssignMovingAvg_1/41916908*
_output_shapes
:2.
,batch_normalization_57/AssignMovingAvg_1/subç
,batch_normalization_57/AssignMovingAvg_1/mulMul0batch_normalization_57/AssignMovingAvg_1/sub:z:07batch_normalization_57/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_57/AssignMovingAvg_1/41916908*
_output_shapes
:2.
,batch_normalization_57/AssignMovingAvg_1/mulÉ
<batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_57_assignmovingavg_1_419169080batch_normalization_57/AssignMovingAvg_1/mul:z:08^batch_normalization_57/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_57/AssignMovingAvg_1/41916908*
_output_shapes
 *
dtype02>
<batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_57/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_57/batchnorm/add/yÞ
$batch_normalization_57/batchnorm/addAddV21batch_normalization_57/moments/Squeeze_1:output:0/batch_normalization_57/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/add¨
&batch_normalization_57/batchnorm/RsqrtRsqrt(batch_normalization_57/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_57/batchnorm/Rsqrtã
3batch_normalization_57/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_57_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_57/batchnorm/mul/ReadVariableOpá
$batch_normalization_57/batchnorm/mulMul*batch_normalization_57/batchnorm/Rsqrt:y:0;batch_normalization_57/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/mulÚ
&batch_normalization_57/batchnorm/mul_1Mul!conv1d_11/conv1d/Squeeze:output:0(batch_normalization_57/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_57/batchnorm/mul_1×
&batch_normalization_57/batchnorm/mul_2Mul/batch_normalization_57/moments/Squeeze:output:0(batch_normalization_57/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_57/batchnorm/mul_2×
/batch_normalization_57/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_57_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_57/batchnorm/ReadVariableOpÝ
$batch_normalization_57/batchnorm/subSub7batch_normalization_57/batchnorm/ReadVariableOp:value:0*batch_normalization_57/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_57/batchnorm/subå
&batch_normalization_57/batchnorm/add_1AddV2*batch_normalization_57/batchnorm/mul_1:z:0(batch_normalization_57/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_57/batchnorm/add_1
re_lu_32/ReluRelu*batch_normalization_57/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_32/Reluu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
flatten_10/Const
flatten_10/ReshapeReshapere_lu_32/Relu:activations:0flatten_10/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
flatten_10/Reshape¨
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_46/MatMul/ReadVariableOp£
dense_46/MatMulMatMulflatten_10/Reshape:output:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_46/MatMul|
dense_46/SigmoidSigmoiddense_46/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_46/Sigmoid

IdentityIdentitydense_46/Sigmoid:y:0;^batch_normalization_55/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_55/AssignMovingAvg/ReadVariableOp=^batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_55/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_55/batchnorm/ReadVariableOp4^batch_normalization_55/batchnorm/mul/ReadVariableOp;^batch_normalization_56/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_56/AssignMovingAvg/ReadVariableOp=^batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_56/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_56/batchnorm/ReadVariableOp4^batch_normalization_56/batchnorm/mul/ReadVariableOp;^batch_normalization_57/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_57/AssignMovingAvg/ReadVariableOp=^batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_57/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_57/batchnorm/ReadVariableOp4^batch_normalization_57/batchnorm/mul/ReadVariableOp-^conv1d_10/conv1d/ExpandDims_1/ReadVariableOp-^conv1d_11/conv1d/ExpandDims_1/ReadVariableOp^dense_45/MatMul/ReadVariableOp^dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2x
:batch_normalization_55/AssignMovingAvg/AssignSubVariableOp:batch_normalization_55/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_55/AssignMovingAvg/ReadVariableOp5batch_normalization_55/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_55/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_55/batchnorm/ReadVariableOp/batch_normalization_55/batchnorm/ReadVariableOp2j
3batch_normalization_55/batchnorm/mul/ReadVariableOp3batch_normalization_55/batchnorm/mul/ReadVariableOp2x
:batch_normalization_56/AssignMovingAvg/AssignSubVariableOp:batch_normalization_56/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_56/AssignMovingAvg/ReadVariableOp5batch_normalization_56/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_56/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_56/AssignMovingAvg_1/ReadVariableOp7batch_normalization_56/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_56/batchnorm/ReadVariableOp/batch_normalization_56/batchnorm/ReadVariableOp2j
3batch_normalization_56/batchnorm/mul/ReadVariableOp3batch_normalization_56/batchnorm/mul/ReadVariableOp2x
:batch_normalization_57/AssignMovingAvg/AssignSubVariableOp:batch_normalization_57/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_57/AssignMovingAvg/ReadVariableOp5batch_normalization_57/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_57/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_57/AssignMovingAvg_1/ReadVariableOp7batch_normalization_57/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_57/batchnorm/ReadVariableOp/batch_normalization_57/batchnorm/ReadVariableOp2j
3batch_normalization_57/batchnorm/mul/ReadVariableOp3batch_normalization_57/batchnorm/mul/ReadVariableOp2\
,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp,conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2\
,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp,conv1d_11/conv1d/ExpandDims_1/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
d
H__inference_flatten_10_layer_call_and_return_conditional_losses_41916502

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

Ü
0__inference_Discriminator_layer_call_fn_41917060

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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Discriminator_layer_call_and_return_conditional_losses_419166302
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
À
q
+__inference_dense_45_layer_call_fn_41917111

inputs
unknown
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_419161592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
¡
F__inference_dense_45_layer_call_and_return_conditional_losses_41916159

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í^
Ë
!__inference__traced_save_41917785
file_prefix.
*savev2_dense_45_kernel_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop/
+savev2_conv1d_10_kernel_read_readvariableop;
7savev2_batch_normalization_56_gamma_read_readvariableop:
6savev2_batch_normalization_56_beta_read_readvariableopA
=savev2_batch_normalization_56_moving_mean_read_readvariableopE
Asavev2_batch_normalization_56_moving_variance_read_readvariableop/
+savev2_conv1d_11_kernel_read_readvariableop;
7savev2_batch_normalization_57_gamma_read_readvariableop:
6savev2_batch_normalization_57_beta_read_readvariableopA
=savev2_batch_normalization_57_moving_mean_read_readvariableopE
Asavev2_batch_normalization_57_moving_variance_read_readvariableop.
*savev2_dense_46_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_m_read_readvariableop6
2savev2_adam_conv1d_10_kernel_m_read_readvariableopB
>savev2_adam_batch_normalization_56_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_56_beta_m_read_readvariableop6
2savev2_adam_conv1d_11_kernel_m_read_readvariableopB
>savev2_adam_batch_normalization_57_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_57_beta_m_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_v_read_readvariableop6
2savev2_adam_conv1d_10_kernel_v_read_readvariableopB
>savev2_adam_batch_normalization_56_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_56_beta_v_read_readvariableop6
2savev2_adam_conv1d_11_kernel_v_read_readvariableopB
>savev2_adam_batch_normalization_57_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_57_beta_v_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*£
valueB,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesà
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_45_kernel_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop+savev2_conv1d_10_kernel_read_readvariableop7savev2_batch_normalization_56_gamma_read_readvariableop6savev2_batch_normalization_56_beta_read_readvariableop=savev2_batch_normalization_56_moving_mean_read_readvariableopAsavev2_batch_normalization_56_moving_variance_read_readvariableop+savev2_conv1d_11_kernel_read_readvariableop7savev2_batch_normalization_57_gamma_read_readvariableop6savev2_batch_normalization_57_beta_read_readvariableop=savev2_batch_normalization_57_moving_mean_read_readvariableopAsavev2_batch_normalization_57_moving_variance_read_readvariableop*savev2_dense_46_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop>savev2_adam_batch_normalization_55_gamma_m_read_readvariableop=savev2_adam_batch_normalization_55_beta_m_read_readvariableop2savev2_adam_conv1d_10_kernel_m_read_readvariableop>savev2_adam_batch_normalization_56_gamma_m_read_readvariableop=savev2_adam_batch_normalization_56_beta_m_read_readvariableop2savev2_adam_conv1d_11_kernel_m_read_readvariableop>savev2_adam_batch_normalization_57_gamma_m_read_readvariableop=savev2_adam_batch_normalization_57_beta_m_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop>savev2_adam_batch_normalization_55_gamma_v_read_readvariableop=savev2_adam_batch_normalization_55_beta_v_read_readvariableop2savev2_adam_conv1d_10_kernel_v_read_readvariableop>savev2_adam_batch_normalization_56_gamma_v_read_readvariableop=savev2_adam_batch_normalization_56_beta_v_read_readvariableop2savev2_adam_conv1d_11_kernel_v_read_readvariableop>savev2_adam_batch_normalization_57_gamma_v_read_readvariableop=savev2_adam_batch_normalization_57_beta_v_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
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

identity_1Identity_1:output:0*Ç
_input_shapesµ
²: :@:@:@:@:@:::::::::::@: : : : : : : :@:@:@:::::::@:@:@:@:::::::@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@:
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
: :$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::  

_output_shapes
::$! 

_output_shapes

:@:$" 

_output_shapes

:@: #

_output_shapes
:@: $

_output_shapes
:@:(%$
"
_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::(($
"
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::$+ 

_output_shapes

:@:,

_output_shapes
: 
á
d
H__inference_reshape_10_layer_call_and_return_conditional_losses_41917216

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë

T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41916319

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
Ï
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41915964

inputs
assignmovingavg_41915939
assignmovingavg_1_41915945)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915939*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41915939*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915939*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915939*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41915939AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915939*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915945*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41915945*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915945*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915945*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41915945AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915945*
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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
q
+__inference_dense_46_layer_call_fn_41917633

inputs
unknown
identity¢StatefulPartitionedCallé
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
GPU 2J 8 *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_419165182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
I
-__inference_reshape_10_layer_call_fn_41917221

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_419162322
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
I
-__inference_flatten_10_layer_call_fn_41917618

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_419165022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
b
F__inference_re_lu_32_layer_call_and_return_conditional_losses_41916488

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41916447

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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
G
+__inference_re_lu_32_layer_call_fn_41917607

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_32_layer_call_and_return_conditional_losses_419164882
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
G__inference_conv1d_11_layer_call_and_return_conditional_losses_41916380

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
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917296

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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
¬
9__inference_batch_normalization_55_layer_call_fn_41917193

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
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_419158572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41915997

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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
G__inference_conv1d_10_layer_call_and_return_conditional_losses_41916252

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
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
IdentityIdentityconv1d/Squeeze:output:0#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýº

$__inference__traced_restore_41917924
file_prefix$
 assignvariableop_dense_45_kernel3
/assignvariableop_1_batch_normalization_55_gamma2
.assignvariableop_2_batch_normalization_55_beta9
5assignvariableop_3_batch_normalization_55_moving_mean=
9assignvariableop_4_batch_normalization_55_moving_variance'
#assignvariableop_5_conv1d_10_kernel3
/assignvariableop_6_batch_normalization_56_gamma2
.assignvariableop_7_batch_normalization_56_beta9
5assignvariableop_8_batch_normalization_56_moving_mean=
9assignvariableop_9_batch_normalization_56_moving_variance(
$assignvariableop_10_conv1d_11_kernel4
0assignvariableop_11_batch_normalization_57_gamma3
/assignvariableop_12_batch_normalization_57_beta:
6assignvariableop_13_batch_normalization_57_moving_mean>
:assignvariableop_14_batch_normalization_57_moving_variance'
#assignvariableop_15_dense_46_kernel!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count.
*assignvariableop_23_adam_dense_45_kernel_m;
7assignvariableop_24_adam_batch_normalization_55_gamma_m:
6assignvariableop_25_adam_batch_normalization_55_beta_m/
+assignvariableop_26_adam_conv1d_10_kernel_m;
7assignvariableop_27_adam_batch_normalization_56_gamma_m:
6assignvariableop_28_adam_batch_normalization_56_beta_m/
+assignvariableop_29_adam_conv1d_11_kernel_m;
7assignvariableop_30_adam_batch_normalization_57_gamma_m:
6assignvariableop_31_adam_batch_normalization_57_beta_m.
*assignvariableop_32_adam_dense_46_kernel_m.
*assignvariableop_33_adam_dense_45_kernel_v;
7assignvariableop_34_adam_batch_normalization_55_gamma_v:
6assignvariableop_35_adam_batch_normalization_55_beta_v/
+assignvariableop_36_adam_conv1d_10_kernel_v;
7assignvariableop_37_adam_batch_normalization_56_gamma_v:
6assignvariableop_38_adam_batch_normalization_56_beta_v/
+assignvariableop_39_adam_conv1d_11_kernel_v;
7assignvariableop_40_adam_batch_normalization_57_gamma_v:
6assignvariableop_41_adam_batch_normalization_57_beta_v.
*assignvariableop_42_adam_dense_46_kernel_v
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*£
valueB,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesæ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_45_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1´
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_55_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_55_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3º
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_55_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¾
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_55_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv1d_10_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_56_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_56_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8º
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_56_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_56_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv1d_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¸
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_57_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_57_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¾
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_57_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Â
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_57_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_46_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¦
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_45_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¿
AssignVariableOp_24AssignVariableOp7assignvariableop_24_adam_batch_normalization_55_gamma_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¾
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_batch_normalization_55_beta_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26³
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_conv1d_10_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¿
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_batch_normalization_56_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¾
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_batch_normalization_56_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1d_11_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¿
AssignVariableOp_30AssignVariableOp7assignvariableop_30_adam_batch_normalization_57_gamma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¾
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_57_beta_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32²
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_46_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_45_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¿
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_batch_normalization_55_gamma_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_55_beta_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36³
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_conv1d_10_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¿
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_56_gamma_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¾
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_56_beta_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39³
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_11_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¿
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_batch_normalization_57_gamma_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¾
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_57_beta_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42²
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_46_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_42AssignVariableOp_422(
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
ñ
¬
9__inference_batch_normalization_57_layer_call_fn_41917515

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
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_419161372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¬
9__inference_batch_normalization_57_layer_call_fn_41917597

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
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_419164472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41916137

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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

Þ
0__inference_Discriminator_layer_call_fn_41916665
input_18
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
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *T
fORM
K__inference_Discriminator_layer_call_and_return_conditional_losses_419166302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_18
Û0
Ï
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917358

inputs
assignmovingavg_41917333
assignmovingavg_1_41917339)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917333*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41917333*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917333*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41917333*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41917333AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41917333*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917339*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41917339*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917339*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41917339*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41917339AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41917339*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
d
H__inference_flatten_10_layer_call_and_return_conditional_losses_41917613

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦0
Ï
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41915824

inputs
assignmovingavg_41915799
assignmovingavg_1_41915805)
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
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

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915799*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41915799*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915799*
_output_shapes
:@2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41915799*
_output_shapes
:@2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41915799AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41915799*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915805*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41915805*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915805*
_output_shapes
:@2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41915805*
_output_shapes
:@2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41915805AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41915805*
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
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

G
+__inference_re_lu_30_layer_call_fn_41917203

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_30_layer_call_and_return_conditional_losses_419162112
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û0
Ï
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41916427

inputs
assignmovingavg_41916402
assignmovingavg_1_41916408)
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
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Î
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916402*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_41916402*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpó
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916402*
_output_shapes
:2
AssignMovingAvg/subê
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/41916402*
_output_shapes
:2
AssignMovingAvg/mul³
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_41916402AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/41916402*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÔ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916408*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_41916408*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpý
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916408*
_output_shapes
:2
AssignMovingAvg_1/subô
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/41916408*
_output_shapes
:2
AssignMovingAvg_1/mul¿
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_41916408AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/41916408*
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
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
d
H__inference_reshape_10_layer_call_and_return_conditional_losses_41916232

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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ
¬
9__inference_batch_normalization_56_layer_call_fn_41917322

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
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_419159972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
b
F__inference_re_lu_31_layer_call_and_return_conditional_losses_41916360

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
¡
F__inference_dense_46_layer_call_and_return_conditional_losses_41917626

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917489

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
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î:
¥
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916630

inputs
dense_45_41916585#
batch_normalization_55_41916588#
batch_normalization_55_41916590#
batch_normalization_55_41916592#
batch_normalization_55_41916594
conv1d_10_41916599#
batch_normalization_56_41916602#
batch_normalization_56_41916604#
batch_normalization_56_41916606#
batch_normalization_56_41916608
conv1d_11_41916612#
batch_normalization_57_41916615#
batch_normalization_57_41916617#
batch_normalization_57_41916619#
batch_normalization_57_41916621
dense_46_41916626
identity¢.batch_normalization_55/StatefulPartitionedCall¢.batch_normalization_56/StatefulPartitionedCall¢.batch_normalization_57/StatefulPartitionedCall¢!conv1d_10/StatefulPartitionedCall¢!conv1d_11/StatefulPartitionedCall¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall
 dense_45/StatefulPartitionedCallStatefulPartitionedCallinputsdense_45_41916585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_419161592"
 dense_45/StatefulPartitionedCallÇ
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0batch_normalization_55_41916588batch_normalization_55_41916590batch_normalization_55_41916592batch_normalization_55_41916594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_4191582420
.batch_normalization_55/StatefulPartitionedCall
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_30_layer_call_and_return_conditional_losses_419162112
re_lu_30/PartitionedCallû
reshape_10/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_419162322
reshape_10/PartitionedCallª
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall#reshape_10/PartitionedCall:output:0conv1d_10_41916599*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_10_layer_call_and_return_conditional_losses_419162522#
!conv1d_10/StatefulPartitionedCallÌ
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0batch_normalization_56_41916602batch_normalization_56_41916604batch_normalization_56_41916606batch_normalization_56_41916608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_4191629920
.batch_normalization_56/StatefulPartitionedCall
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_31_layer_call_and_return_conditional_losses_419163602
re_lu_31/PartitionedCall¨
!conv1d_11/StatefulPartitionedCallStatefulPartitionedCall!re_lu_31/PartitionedCall:output:0conv1d_11_41916612*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv1d_11_layer_call_and_return_conditional_losses_419163802#
!conv1d_11/StatefulPartitionedCallÌ
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall*conv1d_11/StatefulPartitionedCall:output:0batch_normalization_57_41916615batch_normalization_57_41916617batch_normalization_57_41916619batch_normalization_57_41916621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_4191642720
.batch_normalization_57/StatefulPartitionedCall
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_re_lu_32_layer_call_and_return_conditional_losses_419164882
re_lu_32/PartitionedCall÷
flatten_10/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_419165022
flatten_10/PartitionedCall¢
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_46_41916626*
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
GPU 2J 8 *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_419165182"
 dense_46/StatefulPartitionedCall
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall"^conv1d_11/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::::2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2F
!conv1d_11/StatefulPartitionedCall!conv1d_11/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
input_181
serving_default_input_18:0ÿÿÿÿÿÿÿÿÿ<
dense_460
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:©
g
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
+¶&call_and_return_all_conditional_losses
·__call__
¸_default_save_signature"êb
_tf_keras_networkÎb{"class_name": "Functional", "name": "Discriminator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}, "name": "input_18", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["input_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["dense_45", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_30", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_30", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_10", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8]}}, "name": "reshape_10", "inbound_nodes": [[["re_lu_30", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_10", "inbound_nodes": [[["reshape_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_56", "inbound_nodes": [[["conv1d_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_31", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_31", "inbound_nodes": [[["batch_normalization_56", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_11", "inbound_nodes": [[["re_lu_31", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv1d_11", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_32", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_32", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["re_lu_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_46", "inbound_nodes": [[["flatten_10", 0, 0, {}]]]}], "input_layers": [["input_18", 0, 0]], "output_layers": [["dense_46", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Discriminator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}, "name": "input_18", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["input_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["dense_45", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_30", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_30", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_10", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8]}}, "name": "reshape_10", "inbound_nodes": [[["re_lu_30", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_10", "inbound_nodes": [[["reshape_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_56", "inbound_nodes": [[["conv1d_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_31", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_31", "inbound_nodes": [[["batch_normalization_56", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_11", "inbound_nodes": [[["re_lu_31", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_57", "inbound_nodes": [[["conv1d_11", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_32", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_32", "inbound_nodes": [[["batch_normalization_57", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["re_lu_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_46", "inbound_nodes": [[["flatten_10", 0, 0, {}]]]}], "input_layers": [["input_18", 0, 0]], "output_layers": [["dense_46", 0, 0]]}}, "training_config": {"loss": "wasserstein_loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ë"è
_tf_keras_input_layerÈ{"class_name": "InputLayer", "name": "input_18", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}}
ë

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
¶	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"à
_tf_keras_layerÆ{"class_name": "BatchNormalization", "name": "batch_normalization_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_55", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ï
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_30", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ø
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"ç
_tf_keras_layerÍ{"class_name": "Reshape", "name": "reshape_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_10", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8]}}}
ß	

*kernel
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8]}}
¹	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_56", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_56", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 16]}}
ï
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_31", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
à	

<kernel
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"Ã
_tf_keras_layer©{"class_name": "Conv1D", "name": "conv1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_11", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 16]}}
·	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"á
_tf_keras_layerÇ{"class_name": "BatchNormalization", "name": "batch_normalization_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_57", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8]}}
ï
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_32", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ê
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
í

Rkernel
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}

Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratem¢m£m¤*m¥0m¦1m§<m¨Bm©CmªRm«v¬v­v®*v¯0v°1v±<v²Bv³Cv´Rvµ"
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
\layer_metrics
]metrics
^non_trainable_variables
trainable_variables
	variables
_layer_regularization_losses

`layers
·__call__
¸_default_save_signature
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
-
Ñserving_default"
signature_map
!:@2dense_45/kernel
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
alayer_metrics
bmetrics
cnon_trainable_variables
trainable_variables
	variables
dlayer_regularization_losses

elayers
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_55/gamma
):'@2batch_normalization_55/beta
2:0@ (2"batch_normalization_55/moving_mean
6:4@ (2&batch_normalization_55/moving_variance
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
flayer_metrics
gmetrics
hnon_trainable_variables
trainable_variables
 	variables
ilayer_regularization_losses

jlayers
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
"regularization_losses
klayer_metrics
lmetrics
mnon_trainable_variables
#trainable_variables
$	variables
nlayer_regularization_losses

olayers
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
&regularization_losses
player_metrics
qmetrics
rnon_trainable_variables
'trainable_variables
(	variables
slayer_regularization_losses

tlayers
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_10/kernel
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
ulayer_metrics
vmetrics
wnon_trainable_variables
,trainable_variables
-	variables
xlayer_regularization_losses

ylayers
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_56/gamma
):'2batch_normalization_56/beta
2:0 (2"batch_normalization_56/moving_mean
6:4 (2&batch_normalization_56/moving_variance
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
zlayer_metrics
{metrics
|non_trainable_variables
5trainable_variables
6	variables
}layer_regularization_losses

~layers
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
´
8regularization_losses
layer_metrics
metrics
non_trainable_variables
9trainable_variables
:	variables
 layer_regularization_losses
layers
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_11/kernel
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
layer_metrics
metrics
non_trainable_variables
>trainable_variables
?	variables
 layer_regularization_losses
layers
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_57/gamma
):'2batch_normalization_57/beta
2:0 (2"batch_normalization_57/moving_mean
6:4 (2&batch_normalization_57/moving_variance
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
layer_metrics
metrics
non_trainable_variables
Gtrainable_variables
H	variables
 layer_regularization_losses
layers
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Jregularization_losses
layer_metrics
metrics
non_trainable_variables
Ktrainable_variables
L	variables
 layer_regularization_losses
layers
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Nregularization_losses
layer_metrics
metrics
non_trainable_variables
Otrainable_variables
P	variables
 layer_regularization_losses
layers
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_46/kernel
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
layer_metrics
metrics
non_trainable_variables
Ttrainable_variables
U	variables
 layer_regularization_losses
layers
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
J
0
1
22
33
D4
E5"
trackable_list_wrapper
 "
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
¿

total

count
 	variables
¡	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
&:$@2Adam/dense_45/kernel/m
/:-@2#Adam/batch_normalization_55/gamma/m
.:,@2"Adam/batch_normalization_55/beta/m
+:)2Adam/conv1d_10/kernel/m
/:-2#Adam/batch_normalization_56/gamma/m
.:,2"Adam/batch_normalization_56/beta/m
+:)2Adam/conv1d_11/kernel/m
/:-2#Adam/batch_normalization_57/gamma/m
.:,2"Adam/batch_normalization_57/beta/m
&:$@2Adam/dense_46/kernel/m
&:$@2Adam/dense_45/kernel/v
/:-@2#Adam/batch_normalization_55/gamma/v
.:,@2"Adam/batch_normalization_55/beta/v
+:)2Adam/conv1d_10/kernel/v
/:-2#Adam/batch_normalization_56/gamma/v
.:,2"Adam/batch_normalization_56/beta/v
+:)2Adam/conv1d_11/kernel/v
/:-2#Adam/batch_normalization_57/gamma/v
.:,2"Adam/batch_normalization_57/beta/v
&:$@2Adam/dense_46/kernel/v
ú2÷
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916531
K__inference_Discriminator_layer_call_and_return_conditional_losses_41917023
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916934
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916579À
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
2
0__inference_Discriminator_layer_call_fn_41916750
0__inference_Discriminator_layer_call_fn_41916665
0__inference_Discriminator_layer_call_fn_41917060
0__inference_Discriminator_layer_call_fn_41917097À
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
â2ß
#__inference__wrapped_model_41915728·
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
annotationsª *'¢$
"
input_18ÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_dense_45_layer_call_and_return_conditional_losses_41917104¢
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
+__inference_dense_45_layer_call_fn_41917111¢
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
æ2ã
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917147
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917167´
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
°2­
9__inference_batch_normalization_55_layer_call_fn_41917193
9__inference_batch_normalization_55_layer_call_fn_41917180´
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
ð2í
F__inference_re_lu_30_layer_call_and_return_conditional_losses_41917198¢
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
+__inference_re_lu_30_layer_call_fn_41917203¢
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
ò2ï
H__inference_reshape_10_layer_call_and_return_conditional_losses_41917216¢
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
×2Ô
-__inference_reshape_10_layer_call_fn_41917221¢
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
ñ2î
G__inference_conv1d_10_layer_call_and_return_conditional_losses_41917233¢
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
Ö2Ó
,__inference_conv1d_10_layer_call_fn_41917240¢
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
2
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917296
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917358
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917276
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917378´
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
¦2£
9__inference_batch_normalization_56_layer_call_fn_41917404
9__inference_batch_normalization_56_layer_call_fn_41917322
9__inference_batch_normalization_56_layer_call_fn_41917309
9__inference_batch_normalization_56_layer_call_fn_41917391´
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
ð2í
F__inference_re_lu_31_layer_call_and_return_conditional_losses_41917409¢
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
+__inference_re_lu_31_layer_call_fn_41917414¢
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
ñ2î
G__inference_conv1d_11_layer_call_and_return_conditional_losses_41917426¢
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
Ö2Ó
,__inference_conv1d_11_layer_call_fn_41917433¢
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
2
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917489
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917469
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917551
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917571´
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
¦2£
9__inference_batch_normalization_57_layer_call_fn_41917515
9__inference_batch_normalization_57_layer_call_fn_41917597
9__inference_batch_normalization_57_layer_call_fn_41917584
9__inference_batch_normalization_57_layer_call_fn_41917502´
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
ð2í
F__inference_re_lu_32_layer_call_and_return_conditional_losses_41917602¢
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
+__inference_re_lu_32_layer_call_fn_41917607¢
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
ò2ï
H__inference_flatten_10_layer_call_and_return_conditional_losses_41917613¢
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
×2Ô
-__inference_flatten_10_layer_call_fn_41917618¢
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
F__inference_dense_46_layer_call_and_return_conditional_losses_41917626¢
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
+__inference_dense_46_layer_call_fn_41917633¢
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
ÎBË
&__inference_signature_wrapper_41916797input_18"
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
 Ã
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916531t*2301<DEBCR9¢6
/¢,
"
input_18ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916579t*3021<EBDCR9¢6
/¢,
"
input_18ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_Discriminator_layer_call_and_return_conditional_losses_41916934r*2301<DEBCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_Discriminator_layer_call_and_return_conditional_losses_41917023r*3021<EBDCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Discriminator_layer_call_fn_41916665g*2301<DEBCR9¢6
/¢,
"
input_18ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Discriminator_layer_call_fn_41916750g*3021<EBDCR9¢6
/¢,
"
input_18ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Discriminator_layer_call_fn_41917060e*2301<DEBCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Discriminator_layer_call_fn_41917097e*3021<EBDCR7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
#__inference__wrapped_model_41915728z*3021<EBDCR1¢.
'¢$
"
input_18ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_46"
dense_46ÿÿÿÿÿÿÿÿÿº
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917147b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 º
T__inference_batch_normalization_55_layer_call_and_return_conditional_losses_41917167b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
9__inference_batch_normalization_55_layer_call_fn_41917180U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@
9__inference_batch_normalization_55_layer_call_fn_41917193U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@Ô
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917276|2301@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917296|3021@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917358j23017¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Â
T__inference_batch_normalization_56_layer_call_and_return_conditional_losses_41917378j30217¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¬
9__inference_batch_normalization_56_layer_call_fn_41917309o2301@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
9__inference_batch_normalization_56_layer_call_fn_41917322o3021@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_56_layer_call_fn_41917391]23017¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_56_layer_call_fn_41917404]30217¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÔ
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917469|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917489|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917551jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Â
T__inference_batch_normalization_57_layer_call_and_return_conditional_losses_41917571jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¬
9__inference_batch_normalization_57_layer_call_fn_41917502oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
9__inference_batch_normalization_57_layer_call_fn_41917515oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_57_layer_call_fn_41917584]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_57_layer_call_fn_41917597]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ®
G__inference_conv1d_10_layer_call_and_return_conditional_losses_41917233c*3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_conv1d_10_layer_call_fn_41917240V*3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
G__inference_conv1d_11_layer_call_and_return_conditional_losses_41917426c<3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_conv1d_11_layer_call_fn_41917433V<3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
F__inference_dense_45_layer_call_and_return_conditional_losses_41917104[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
+__inference_dense_45_layer_call_fn_41917111N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¥
F__inference_dense_46_layer_call_and_return_conditional_losses_41917626[R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
+__inference_dense_46_layer_call_fn_41917633NR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¨
H__inference_flatten_10_layer_call_and_return_conditional_losses_41917613\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_flatten_10_layer_call_fn_41917618O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¢
F__inference_re_lu_30_layer_call_and_return_conditional_losses_41917198X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
+__inference_re_lu_30_layer_call_fn_41917203K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@ª
F__inference_re_lu_31_layer_call_and_return_conditional_losses_41917409`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_re_lu_31_layer_call_fn_41917414S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
F__inference_re_lu_32_layer_call_and_return_conditional_losses_41917602`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_re_lu_32_layer_call_fn_41917607S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
H__inference_reshape_10_layer_call_and_return_conditional_losses_41917216\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_reshape_10_layer_call_fn_41917221O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ±
&__inference_signature_wrapper_41916797*3021<EBDCR=¢:
¢ 
3ª0
.
input_18"
input_18ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_46"
dense_46ÿÿÿÿÿÿÿÿÿ